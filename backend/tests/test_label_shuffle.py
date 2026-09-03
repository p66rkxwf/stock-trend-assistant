"""打亂標籤測試：標籤一打亂，表現就必須崩塌到基線水準。

沒崩塌 → 管線裡有洩漏。設計理由與「抓得到/抓不到什麼」寫在 stockta/ml/leakage.py。

本檔刻意包含兩類測試：
- 崩塌測試（打亂後應回到基線）——真正的把關。
- **對照測試**（已知有洩漏時應判定為紅）——證明把關有牙齒。一條永遠綠的檢查
  等於沒有檢查；沒有這幾條，整條管線壞掉時崩塌測試也會過。

合成資料刻意帶 AR(1) 動能，使真標籤**學得到東西**——否則「打亂後掉到 0.5」
只是因為本來就沒有訊號，證明不了任何事（test_real_labels_beat_shuffled 把關這點）。
"""

import numpy as np
import pandas as pd
import pytest

from stockta.features.market import build_market_context
from stockta.ml.dataset import build_dataset
from stockta.ml.leakage import (
    inject_duplicate_rows,
    inject_label_derived_feature,
    permute_labels,
    score_dataset,
    ticker_seed,
)
from stockta.ml.models.baselines import RandomForestModel

pytestmark = pytest.mark.leakage

# 固定結束日（非 "now"）——切分點與樣本數才不會隨執行日期漂移，測試才可重現
_END = "2026-06-30"
_PERIODS = 1500
_N_TICKERS = 12

# 切分點：測試期留約 15 個月，使 n_test 夠大、AUC 的標準誤壓在 0.012 附近，
# 遠小於 leakage.AUC_TOLERANCE（0.03），崩塌測試才不會偶發性閃紅。
_SPLIT = {"train_end": "2024-06-30", "val_end": "2025-03-31", "stride": 2}

_PERMUTATION_SEED = 20260816


def _ar1_ohlcv(seed: int, phi: float = 0.4, sigma: float = 0.02) -> pd.DataFrame:
    """帶 AR(1) 動能的合成 OHLCV——報酬有序列相關，模型學得到真訊號。

    sigma 逐檔不同是刻意的：真實股票池的標籤分佈差異極大（實測「觀望」占比
    0.222–0.852），而波動率本來就寫在特徵裡。合成資料若每檔同質，就測不出
    「逐檔置換不是乾淨虛無假設」這個坑——而那正是全尺度跑法上實際踩到的坑。
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp(_END), periods=_PERIODS)
    eps = rng.normal(0, sigma, len(idx))
    ret = np.empty(len(idx))
    ret[0] = eps[0]
    for i in range(1, len(idx)):
        ret[i] = phi * ret[i - 1] + eps[i]

    close = 100.0 * np.exp(np.cumsum(ret))
    spread = np.abs(rng.normal(0, 0.01, len(idx)))
    return pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.005, len(idx))),
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": rng.integers(500_000, 5_000_000, len(idx)).astype(float),
        },
        index=idx,
    )


@pytest.fixture(scope="module")
def pool() -> dict[str, pd.DataFrame]:
    # 波動率由 0.010 掃到 0.035：低波動檔幾乎全是「觀望」、高波動檔多為漲/跌，
    # 重現真實池「各檔標籤分佈差很多」的性質
    sigmas = np.linspace(0.010, 0.035, _N_TICKERS)
    return {
        f"{2000 + i}.TW": _ar1_ohlcv(seed=100 + i, sigma=float(sigmas[i]))
        for i in range(_N_TICKERS)
    }


@pytest.fixture(scope="module")
def context(pool) -> pd.DataFrame:
    # min_tickers 降為 2 以配合小池（正式管線為 BREADTH_MIN_TICKERS=30）
    return build_market_context(_ar1_ohlcv(seed=7), pool, min_tickers=2)


@pytest.fixture(scope="module")
def real_dataset(pool, context):
    return build_dataset(pool, context, **_SPLIT)


@pytest.fixture(scope="module")
def shuffled_dataset(pool, context):
    """預設的 pooled 置換——全池匯總後再洗，才是乾淨的虛無假設。"""
    return build_dataset(pool, context, label_permutation_seed=_PERMUTATION_SEED, **_SPLIT)


@pytest.fixture(scope="module")
def per_ticker_shuffled_dataset(pool, context):
    """逐檔置換——刻意保留的**錯誤**虛無假設，供對照測試釘住這個坑。"""
    return build_dataset(
        pool,
        context,
        label_permutation_seed=_PERMUTATION_SEED,
        label_permutation_scope="per_ticker",
        **_SPLIT,
    )


def _model() -> RandomForestModel:
    # 少樹淺樹：這裡要測的是「有沒有訊號」，不是模型調得多好；CI 速度優先
    return RandomForestModel(n_estimators=40, min_samples_leaf=20, random_state=0)


@pytest.fixture(scope="module")
def real_score(real_dataset):
    return score_dataset(real_dataset, _model)


@pytest.fixture(scope="module")
def shuffled_score(shuffled_dataset):
    return score_dataset(shuffled_dataset, _model)


# === 前置條件：兩次跑法必須只差「標籤對應」一個變因 =========================


def test_permutation_changes_only_the_label_alignment(real_dataset, shuffled_dataset):
    """打亂版與真實版的樣本數、切分邊界、整體類別組成必須相同。

    若樣本數也跟著變，兩者的分數差異就不再只來自標籤，崩塌測試也就失去意義。
    """
    for attr in ("X_train", "X_val", "X_test"):
        assert getattr(real_dataset, attr).shape == getattr(shuffled_dataset, attr).shape

    np.testing.assert_array_equal(real_dataset.dates_test, shuffled_dataset.dates_test)

    # 各期的類別組成會漂移，這是設計使然而非瑕疵，有兩個來源：
    # (1) Dataset 只保留全部標籤的一個子集（stride 子取樣 + embargo 剔除標籤日跨切分者），
    #     置換會改變哪些標籤落在被保留的位置上；
    # (2) 標籤本身有時間趨勢（多頭年「漲」較多），全域置換等於把各期的組成拉向全期平均，
    #     期間越短、與全期平均差越多，漂移越明顯（驗證期最短，實測約 4%）。
    # 這無害——evaluate() 的多數類基線是以實際測試標籤重算的，比較仍然公平。
    # 置換本身的嚴格守恆由 test_permute_labels_keeps_nan_in_place 逐值把關，此處只需
    # 確認沒有退化：三類都還在、且沒有哪一類被洗到接近消失（否則 macro AUC 會無定義）。
    for attr in ("y_train", "y_val", "y_test"):
        fake_share = np.bincount(getattr(shuffled_dataset, attr).astype(int), minlength=3)
        assert (fake_share / fake_share.sum() > 0.05).all(), (
            f"{attr} 有類別被洗到幾乎消失：{fake_share}"
        )

    # 對應關係確實被打斷了——否則這根本不是打亂
    assert not np.array_equal(real_dataset.y_test, shuffled_dataset.y_test)


def test_permute_labels_keeps_nan_in_place():
    """NaN 必須留在原位：它決定哪些樣本有效，洗掉就會改變切分邊界。"""
    labels = pd.Series([0.0, 1.0, np.nan, 2.0, np.nan, 1.0, 0.0, 2.0])
    out = permute_labels(labels, seed=1)

    np.testing.assert_array_equal(out.isna().to_numpy(), labels.isna().to_numpy())
    assert sorted(out.dropna()) == sorted(labels.dropna())


def test_ticker_seed_is_stable_across_processes():
    """種子不得依賴內建 hash()——str 的 hash 受 PYTHONHASHSEED 影響會變。"""
    assert ticker_seed("2330.TW", 42) == ticker_seed("2330.TW", 42)
    assert ticker_seed("2330.TW", 42) != ticker_seed("2317.TW", 42)
    assert ticker_seed("2330.TW", 42) != ticker_seed("2330.TW", 43)


# === 主測試：打亂標籤後必須崩塌 =============================================


def test_shuffled_labels_collapse_to_baseline(shuffled_score):
    assert not shuffled_score.leaked(), (
        f"打亂標籤後仍學得到東西，管線有洩漏：{shuffled_score.summary()}"
    )
    # 明確釘住上下界：太高是洩漏，太低則代表評估本身壞了（例如標籤順序被錯置）
    assert 0.45 <= shuffled_score.macro_auc <= 0.53, shuffled_score.summary()


def test_real_labels_beat_shuffled(real_score, shuffled_score):
    """量表對照：真標籤必須明顯優於打亂版。

    沒有這條，「管線整條壞掉、什麼都學不到」也會讓崩塌測試通過——那是綠燈假象。
    """
    assert real_score.macro_auc > shuffled_score.macro_auc + 0.03, (
        f"真標籤 {real_score.summary()}／打亂 {shuffled_score.summary()}"
    )
    assert real_score.leaked() is True  # 有真訊號時，同一組判準本來就該回報「超出基線」


# === 對照測試：已知有洩漏時必須判定為紅 =====================================


def test_per_ticker_permutation_is_not_a_clean_null(per_ticker_shuffled_dataset):
    """逐檔置換**留得下**可學的訊號——所以它不能當主測試。

    這條測試釘住的是 2026-08-16 全尺度跑法上實際踩到的坑：逐檔置換只打斷同一檔內
    的時間對應，卻保留各檔之間標籤分佈的差異（低波動檔幾乎全「觀望」、高波動檔多
    漲/跌），而波動率本來就寫在特徵裡。當時虛無分佈停在 0.577±0.004 而非 0.5，
    看起來像洩漏，其實是虛無假設沒設乾淨。

    若哪天有人把預設改回 per_ticker，這條會亮紅燈提醒他先讀完這段。
    """
    score = score_dataset(per_ticker_shuffled_dataset, _model)

    assert score.leaked(), (
        "逐檔置換竟然崩塌到基線——合成池的各檔標籤分佈可能變得太接近，"
        f"這條對照因此失去意義，請檢查 pool fixture 的波動率設定：{score.summary()}"
    )


def test_detects_duplicate_rows_across_splits(shuffled_dataset):
    """測試集樣本混進訓練集（去重沒做乾淨的典型後果）必須被抓到。"""
    leaked_dataset = inject_duplicate_rows(shuffled_dataset, n=400)
    score = score_dataset(leaked_dataset, _model)

    assert score.leaked(), f"注入重複樣本後仍未被判定為洩漏：{score.summary()}"


def test_detects_label_derived_feature(shuffled_dataset):
    """任何一欄帶著標籤影子的特徵都必須被抓到。"""
    leaked_dataset = inject_label_derived_feature(shuffled_dataset)
    score = score_dataset(leaked_dataset, _model)

    assert score.leaked(), f"注入標籤衍生特徵後仍未被判定為洩漏：{score.summary()}"
