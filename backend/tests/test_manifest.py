"""資料快照雜湊：內容一樣就同雜湊，改一個數字就不同雜湊。

沒有這條，manifest 只是一份看起來很專業但抓不到任何東西的 JSON。
"""

import pandas as pd
import pytest

from stockta.data.manifest import build, content_sha, describe, diff, read, verify, write

pytestmark = pytest.mark.leakage


@pytest.fixture
def cache_dir(tmp_path, random_walk_ohlcv):
    (tmp_path / "2330.TW.parquet").parent.mkdir(parents=True, exist_ok=True)
    random_walk_ohlcv.to_parquet(tmp_path / "2330.TW.parquet")
    random_walk_ohlcv.iloc[:300].to_parquet(tmp_path / "2317.TW.parquet")
    return tmp_path


def test_content_sha_ignores_row_order(random_walk_ohlcv):
    """雜湊算的是內容不是位元組——列順序被打亂後排序回來，雜湊必須相同。"""
    shuffled = random_walk_ohlcv.sample(frac=1.0, random_state=0)
    assert content_sha(shuffled) == content_sha(random_walk_ohlcv)


def test_content_sha_detects_a_single_changed_price(random_walk_ohlcv):
    """改一格收盤價就必須換一個雜湊——還原股價被回頭改寫就是長這樣。"""
    tampered = random_walk_ohlcv.copy()
    tampered.iloc[100, tampered.columns.get_loc("close")] *= 1.001

    assert content_sha(tampered) != content_sha(random_walk_ohlcv)


def test_build_and_verify_roundtrip(cache_dir):
    write(build(cache_dir), cache_dir)

    assert verify(cache_dir) == []
    recorded = read(cache_dir)
    assert recorded["n_files"] == 2
    assert set(recorded["files"]) == {"2330.TW", "2317.TW"}


def test_verify_reports_rewritten_history(cache_dir, random_walk_ohlcv):
    """模擬 yfinance 除權息後回頭改寫歷史：列數不變、數值變了。"""
    write(build(cache_dir), cache_dir)

    readjusted = random_walk_ohlcv.copy()
    readjusted[["open", "high", "low", "close"]] *= 0.97  # 配息還原
    readjusted.to_parquet(cache_dir / "2330.TW.parquet")

    differences = verify(cache_dir)
    assert len(differences) == 1
    assert "2330.TW" in differences[0]
    assert "內容雜湊改變" in differences[0]


def test_verify_reports_appended_rows(cache_dir, random_walk_ohlcv):
    write(build(cache_dir), cache_dir)

    extended = random_walk_ohlcv.iloc[:400]
    extended.to_parquet(cache_dir / "2317.TW.parquet")

    differences = verify(cache_dir)
    assert any("2317.TW" in line and "rows 300 → 400" in line for line in differences)


def test_diff_reports_added_and_removed_tickers():
    old = {"files": {"2330.TW": {"sha256": "a"}}}
    new = {"files": {"2454.TW": {"sha256": "b"}}}

    messages = diff(old, new)
    assert any("2330.TW" in m and "消失" in m for m in messages)
    assert any("2454.TW" in m and "新增" in m for m in messages)


def test_manifest_sha_changes_when_any_file_changes(cache_dir, random_walk_ohlcv):
    """彙總雜湊是 artifact metadata 記的那一個——任一檔變動它就必須變。"""
    before = build(cache_dir)["manifest_sha"]

    tampered = random_walk_ohlcv.copy()
    tampered.iloc[5, tampered.columns.get_loc("volume")] += 1
    tampered.to_parquet(cache_dir / "2330.TW.parquet")

    assert build(cache_dir)["manifest_sha"] != before


def test_describe_summarises_content(random_walk_ohlcv):
    summary = describe(random_walk_ohlcv)

    assert summary["rows"] == len(random_walk_ohlcv)
    assert summary["first_date"] == str(random_walk_ohlcv.index.min().date())
    assert summary["na_rows"] == 0


def test_verify_without_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify(tmp_path)


def test_empty_cache_builds_a_manifest(tmp_path):
    """空快取不該爆——CI 機器上就是沒有 parquet。"""
    manifest = build(tmp_path)

    assert manifest["n_files"] == 0
    assert manifest["files"] == {}
    assert isinstance(manifest["manifest_sha"], str)


def test_parquet_roundtrip_preserves_the_hash(cache_dir, random_walk_ohlcv):
    """寫進 parquet 再讀回來，雜湊必須不變，否則每次重寫快取都會誤報漂移。"""
    reloaded = pd.read_parquet(cache_dir / "2330.TW.parquet")

    assert content_sha(reloaded) == content_sha(random_walk_ohlcv)
