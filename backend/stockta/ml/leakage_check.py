"""全尺度打亂標籤測試：python -m stockta.ml.leakage_check

CI 那條（tests/test_label_shuffle.py）用小合成池換取速度——3 秒跑完，每次 push 都擋。
這裡用**正式股票池與正式超參數**跑同一套判準，把數字寫進 docs/leakage_report.md
供報告與評審引用。

與 CI 版的兩個差別，都是為了統計效力：

1. **跑多個置換種子**（--repeats），得到打亂標籤下的**經驗虛無分佈**，而不是單一數字。
   單跑一次只能得到一個點，落在 0.53 還是 0.55 完全可能是抽樣噪音；有了分佈才
   說得出「真標籤的成績是否超出虛無分佈」。
2. **預設不使用正式切分**。正式切分（SPLIT_VAL_END=2026-06-30）之後只剩約一個半月
   交易日，測試樣本僅數百筆，macro AUC 的標準誤達 0.03~0.04——跟判定門檻同一個量級，
   跑出來的紅綠燈沒有意義。故預設把切分點前移，換一個夠長的測試期。
   要看正式切分的結果就傳 --production-split（報告會標注樣本數不足的限制）。

判準與設計理由（抓得到什麼、抓不到什麼）見 stockta/ml/leakage.py。
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from stockta.config import (
    HISTORY_YEARS,
    SPLIT_TRAIN_END,
    SPLIT_VAL_END,
    STOCK_POOL,
)
from stockta.ml.dataset import build_dataset
from stockta.ml.leakage import (
    ACCURACY_TOLERANCE,
    AUC_TOLERANCE,
    PipelineScore,
    inject_duplicate_rows,
    inject_label_derived_feature,
    score_dataset,
)
from stockta.ml.train import create_model, load_market_context, load_pool_ohlcv

DEFAULT_SEED = 20260816

# 預設切分：把切分點前移，換一個夠長（約兩年）的測試期。正式切分的測試期目前
# 只有約一個半月，樣本數不足以支撐 0.03 的判定門檻（見模組 docstring）。
POWER_TRAIN_END = "2023-06-30"
POWER_VAL_END = "2024-06-30"


def _run(
    name: str, dataset, model_name: str, expect: str
) -> tuple[str, PipelineScore, float, str]:
    t0 = time.perf_counter()
    score = score_dataset(dataset, lambda: create_model(model_name))
    return name, score, time.perf_counter() - t0, expect


def _verdict(score: PipelineScore, expect: str) -> str:
    """依該列的**預期**解讀 leaked()。

    同一個布林值在不同列代表不同意思：主測試那幾列「超出基線」＝洩漏；
    真標籤那列「超出基線」＝資料裡有訊號可學，本來就該如此；刻意注入那幾列
    「超出基線」＝判準成功抓到已知的洩漏。不分開講會讓表格自相矛盾。
    """
    beat_baseline = score.leaked()
    if expect == "null":  # 打亂標籤：不該學得到東西
        return "🟢 無洩漏跡象" if not beat_baseline else "🔴 有洩漏"
    if expect == "signal":  # 真標籤：本來就該學得到
        return "🟢 有訊號（預期）" if beat_baseline else "🔴 學不到訊號"
    return "🟢 已偵測（預期）" if beat_baseline else "🔴 判準失效，未偵測到"


def _table(rows: list[tuple[str, PipelineScore, float, str]]) -> list[str]:
    lines = [
        "| 跑法 | 訓練筆數 | 測試筆數 | macro AUC | 超出隨機 | 準確率 | 多數類基線 | 超出基線 | 判定 | 耗時 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s, seconds, expect in rows:
        lines.append(
            f"| {name} | {s.n_train:,} | {s.n_test:,} | {s.macro_auc:.4f} | {s.auc_excess:+.4f} "
            f"| {s.accuracy:.4f} | {s.majority_baseline:.4f} | {s.accuracy_excess:+.4f} "
            f"| {_verdict(s, expect)} | {seconds:.0f}s |"
        )
    return lines


def _null_summary(nulls: list[PipelineScore], real: PipelineScore) -> list[str]:
    """把多次置換的結果整理成虛無分佈，並用它來判斷真標籤成績。"""
    aucs = np.array([s.macro_auc for s in nulls])
    mean, sd = float(aucs.mean()), float(aucs.std(ddof=1)) if len(aucs) > 1 else 0.0
    # 真標籤成績在虛無分佈上的 z 分數；置換次數少時只作參考，不當成 p 值用
    z = (real.macro_auc - mean) / sd if sd > 0 else float("nan")
    return [
        "",
        f"- 虛無分佈（{len(nulls)} 個置換種子）：macro AUC 平均 **{mean:.4f}**、"
        f"標準差 {sd:.4f}、範圍 {aucs.min():.4f}–{aucs.max():.4f}",
        f"- 真標籤 macro AUC **{real.macro_auc:.4f}**，高出虛無分佈平均 "
        f"{real.macro_auc - mean:+.4f}（z ≈ {z:.1f}）",
        f"- 虛無分佈平均超出隨機猜測 {mean - 0.5:+.4f}"
        f"（判定門檻 {AUC_TOLERANCE}）→ "
        + ("🔴 **超出門檻，管線有洩漏**" if mean - 0.5 > AUC_TOLERANCE else "🟢 未超出門檻"),
        "",
    ]


def _write_report(
    results: dict[str, dict],
    n_tickers: int,
    stride: int,
    seed: int,
    repeats: int,
    train_end: str,
    val_end: str,
    production_split: bool,
) -> None:
    from stockta.config import BACKEND_ROOT

    lines = [
        "# 打亂標籤測試報告（全尺度）",
        "",
        f"- 產生時間：{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"- 股票池：{n_tickers} 檔（近 {HISTORY_YEARS} 年）、stride={stride}",
        f"- 切分：訓練 ≤{train_end}、驗證 ≤{val_end}、其後測試",
        f"- 置換種子 {seed} 起連續 {repeats} 個，構成虛無分佈",
        f"- 判定門檻：macro AUC 超出隨機 > {AUC_TOLERANCE}，或準確率超出多數類基線 "
        f"> {ACCURACY_TOLERANCE}，即判定洩漏",
        "",
    ]

    if production_split:
        lines += [
            "> ⚠️ **本次使用正式切分，統計效力不足。** 正式切分的測試期只剩約一個半月",
            f"> （SPLIT_VAL_END={SPLIT_VAL_END}，2026-08-14 實驗 #9 前移切分所致），",
            "> 測試樣本僅數百筆，macro AUC 的標準誤與判定門檻同一量級，紅綠燈不可信。",
            "> 結論請以預設切分（測試期約兩年）的那份報告為準。",
            "",
        ]
    else:
        lines += [
            f"> 本報告刻意**不使用正式切分**（訓練 ≤{SPLIT_TRAIN_END}、驗證 ≤{SPLIT_VAL_END}）：",
            "> 正式切分之後只剩約一個半月交易日，測試樣本數百筆，macro AUC 的標準誤",
            "> 達 0.03~0.04，與判定門檻同一量級，跑出來的紅綠燈沒有意義。洩漏是**管線的",
            "> 性質**、不是某個切分點的性質，改用較早的切分點換取足夠長的測試期，",
            "> 檢定力才夠。要看正式切分的結果：`--production-split`。",
            "",
        ]

    lines += [
        "## 這條測試抓得到什麼、抓不到什麼",
        "",
        "把標籤隨機打亂、整條管線原封不動重跑，表現必須崩塌到基線水準。沒崩塌，",
        "就代表管線裡有洩漏。但它的涵蓋範圍必須講清楚，否則綠燈只是自我安慰：",
        "",
        "- **抓得到**：標籤資訊回流管線的洩漏——訓練/測試樣本重複或高度重疊、",
        "  預處理在全樣本上 fit、任何由標籤衍生出來的特徵。",
        "- **抓不到**：純前視特徵（特徵含 t 之後的資訊）。標籤一打亂，前視特徵一樣",
        "  派不上用場，這條測試對它是盲的——那由 `tests/test_no_lookahead.py` 的",
        "  「竄改未來、逐值比對」負責。**兩者互補，缺一不可。**",
        "",
        "置換綁在 (ticker, 日期) 鍵上、**在切分之前**、且**跨全池匯總後**完成，",
        "train/val/test 共用同一份假標籤。只打亂訓練標籤就抓不到「跨切分重複樣本」",
        "——而重疊視窗（相鄰樣本共用 59/60 歷史）正是本專案最可能的洩漏形態。",
        "為什麼必須跨全池匯總，見下文「一個實際踩過的坑」。",
        "",
        "判定門檻的兩個指標中，**macro AUC 是主指標**。準確率會被「打亂後多數類占比",
        "改變」干擾（表中打亂各列的多數類基線約 0.49，真標籤列約 0.46），單看準確率",
        "容易誤讀；AUC 對類別占比不敏感，虛無水準固定在 0.5。",
        "",
    ]

    for model_name, payload in results.items():
        lines += [
            f"## {model_name}",
            *_null_summary(payload["nulls"], payload["real"]),
            *_table(payload["rows"]),
            "",
        ]

    lines += [
        "## 為什麼跑多個置換種子",
        "",
        "單跑一次只能得到一個點。打亂標籤後拿到 0.52 還是 0.55，很可能純粹是抽樣噪音——",
        "沒有分佈就無法分辨「殘餘訊號」與「運氣」。跑多個種子得到經驗虛無分佈後，",
        "才說得出真標籤的成績究竟超出虛無多少個標準差。",
        "",
        "## 為什麼置換要跨全池匯總（一個實際踩過的坑）",
        "",
        "第一版把標籤**逐檔**各洗各的，虛無分佈停在 macro AUC 0.577±0.004（5 個種子、",
        "n_test≈5,000，約 10 個標準誤），看起來像嚴重洩漏。追下去發現不是洩漏，",
        "是**虛無假設沒設乾淨**：",
        "",
        "逐檔置換只打斷「同一檔內特徵與標籤的時間對應」，卻保留了**各檔之間標籤分佈",
        "的差異**。實測全池 49 檔的「觀望」占比從 0.222（世芯-KY，高波動）到 0.852",
        "（中華電，牛皮）差近 4 倍，而波動率、通道寬度、量能這些特徵本來就認得出",
        "是哪一類股票——模型於是學得到「這種特徵的股票，標籤分佈長這樣」。",
        "那是真實且合理的訊號，不是洩漏。",
        "",
        "改成全池匯總後再洗，所有樣本可交換，虛無分佈才落回 0.5。表中「逐檔置換」",
        "那一列保留下來作為此判斷的證據。",
        "",
        "**教訓**：虛無假設設錯，紅燈會讓人去追一個不存在的 bug。而這個坑用同質的",
        "合成資料是驗不出來的——CI 測試的合成池因此刻意讓各檔波動率不同。",
        "",
        "## 對照組為什麼必須存在",
        "",
        "表中「刻意注入」兩列是**已知有洩漏**的對照：一列把測試樣本複製進訓練集",
        "（去重失效），一列加入由標籤直接算出的特徵（目標洩漏）。它們必須被判為紅。",
        "",
        "一條永遠綠的檢查等於沒有檢查——若管線整條壞掉、模型什麼都學不到，",
        "「打亂後掉到基線」一樣會通過。對照組證明這套判準真的抓得到東西；",
        "「真標籤」那列則證明資料裡本來就有訊號可學。三者要一起看才有意義。",
        "",
        "> 由 `python -m stockta.ml.leakage_check` 自動產生。",
        "> CI 版本（小合成池、3 秒）在 `tests/test_label_shuffle.py`，由 pre-push hook 執行。",
        "",
    ]

    out = BACKEND_ROOT.parent / "docs" / "leakage_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n報告已寫入 {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="全尺度打亂標籤測試")
    parser.add_argument("--models", nargs="+", default=["rf"], help="要跑的模型（預設 rf）")
    parser.add_argument("--stride", type=int, default=2, help="視窗取樣間隔，與正式訓練一致")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="標籤置換種子（起點）")
    parser.add_argument(
        "--repeats", type=int, default=5, help="置換次數（構成虛無分佈，預設 5）"
    )
    parser.add_argument(
        "--production-split",
        action="store_true",
        help="改用 config 的正式切分（測試期過短、統計效力不足，報告會標注）",
    )
    # 注意：市場寬度需要當日至少 BREADTH_MIN_TICKERS（30）檔有資料，否則整段 context
    # 為 NaN、特徵被 dropna 清空。故 --tickers 少於 30 檔會得到空資料集，這是
    # build_market_context() 的既有設計，不是這裡的 bug。
    parser.add_argument("--tickers", nargs="*", default=None, help="只用指定股票（需 ≥30 檔）")
    args = parser.parse_args()

    if args.production_split:
        train_end, val_end = SPLIT_TRAIN_END, SPLIT_VAL_END
    else:
        train_end, val_end = POWER_TRAIN_END, POWER_VAL_END
    split = {"train_end": train_end, "val_end": val_end, "stride": args.stride}

    print(f"載入 {len(args.tickers or STOCK_POOL)} 檔股票資料…")
    ohlcv = load_pool_ohlcv(args.tickers)
    context = load_market_context(ohlcv)

    print(f"建立資料集（stride={args.stride}、訓練 ≤{train_end}、驗證 ≤{val_end}）…")
    real = build_dataset(ohlcv, context, **split)
    fakes = [
        build_dataset(ohlcv, context, label_permutation_seed=args.seed + i, **split)
        for i in range(args.repeats)
    ]
    # 錯誤虛無假設的對照：留在報告裡當證據，說明為什麼預設是 pooled
    per_ticker = build_dataset(
        ohlcv,
        context,
        label_permutation_seed=args.seed,
        label_permutation_scope="per_ticker",
        **split,
    )
    print(f"train={len(real.y_train)} val={len(real.y_val)} test={len(real.y_test)}")
    if len(real.y_test) < 1000:
        print(
            f"⚠️ 測試樣本只有 {len(real.y_test)} 筆，macro AUC 標準誤與判定門檻同量級，"
            "紅綠燈參考價值有限"
        )

    results: dict[str, dict] = {}
    for model_name in args.models:
        rows: list[tuple[str, PipelineScore, float, str]] = []
        nulls: list[PipelineScore] = []

        print(f"  [{model_name}] 真標籤（對照：應學得到）…", flush=True)
        real_row = _run("真標籤（對照：應學得到）", real, model_name, "signal")
        print(f"      {real_row[1].summary()}")
        rows.append(real_row)

        for i, fake in enumerate(fakes):
            label = f"**打亂標籤 #{i + 1}（主測試：應崩塌）**"
            print(f"  [{model_name}] {label} …", flush=True)
            row = _run(label, fake, model_name, "null")
            print(f"      {row[1].summary()}  → 超出基線={row[1].leaked()}")
            rows.append(row)
            nulls.append(row[1])

        # 對照組固定用第一個置換，證明判準抓得到已知的洩漏
        for label, dataset in (
            ("對照 — 逐檔置換（錯誤的虛無假設）", per_ticker),
            ("刻意注入 — 跨切分重複樣本", inject_duplicate_rows(fakes[0], n=2000)),
            ("刻意注入 — 標籤衍生特徵", inject_label_derived_feature(fakes[0])),
        ):
            print(f"  [{model_name}] {label} …", flush=True)
            row = _run(label, dataset, model_name, "detect")
            print(f"      {row[1].summary()}  → 超出基線={row[1].leaked()}")
            rows.append(row)

        results[model_name] = {"rows": rows, "nulls": nulls, "real": real_row[1]}

    _write_report(
        results,
        len(ohlcv),
        args.stride,
        args.seed,
        args.repeats,
        train_end,
        val_end,
        args.production_split,
    )


if __name__ == "__main__":
    main()
