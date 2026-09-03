"""資料快照與雜湊：讓「能重跑到同一份數字」這句話有東西可查。

CLI：python -m stockta.data.manifest build|verify|show

**要解決的問題**：`data_cache/*.parquet` 不進版控（太大），而 yfinance 的
adjusted close 會隨新的配息／分割**回頭改寫歷史**——同一段歷史，今天抓到的值
和上個月抓到的值不一樣，而且 provider 的增量合併是 `keep="last"`，新值直接蓋掉舊值，
不留痕跡。沒有快照與雜湊，「這個數字是用哪一份資料跑出來的」就永遠答不出來。

**作法**：對快取內每個 parquet 算內容雜湊與摘要統計，寫成 `data_cache/MANIFEST.json`
（**這一個檔進版控**），再由 `ml/registry.save()` 把彙總雜湊寫進 artifact metadata。
於是每個模型都指得出它是用哪一份資料訓練的。

雜湊算的是**內容**不是檔案位元組：parquet 的壓縮參數與 metadata 會隨套件版本變動，
位元組雜湊會無謂地變。改以正規化後的數值序列化計算，同樣的數字就得到同樣的雜湊。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stockta.config import AUTO_ADJUST, DATA_CACHE_DIR

MANIFEST_NAME = "MANIFEST.json"

# 價格四捨五入到小數 6 位再算雜湊：避開浮點數字面表示在不同 pandas/pyarrow
# 版本間的差異，同時仍足以偵測還原股價被回頭改寫（改寫幅度遠大於 1e-6）。
_ROUND_DP = 6
_COLUMNS = ["open", "high", "low", "close", "volume"]


def content_sha(df: pd.DataFrame) -> str:
    """對 OHLCV 內容算 sha256（與檔案位元組無關，只看數字）。"""
    canon = df.sort_index()[_COLUMNS].round(_ROUND_DP)
    payload = canon.to_csv(float_format=f"%.{_ROUND_DP}f").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def describe(df: pd.DataFrame) -> dict:
    """單一標的的內容摘要——雜湊對不上時，靠這些欄位一眼看出差在哪。"""
    close = df["close"].astype("float64")
    return {
        "sha256": content_sha(df),
        "rows": int(len(df)),
        "first_date": str(df.index.min().date()),
        "last_date": str(df.index.max().date()),
        "first_close": round(float(close.iloc[0]), _ROUND_DP),
        "last_close": round(float(close.iloc[-1]), _ROUND_DP),
        "na_rows": int(df.isna().any(axis=1).sum()),
    }


def _package_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def build(cache_dir: Path = DATA_CACHE_DIR) -> dict:
    """掃描快取目錄，產出完整 manifest（不寫檔）。"""
    files = sorted(p for p in cache_dir.glob("*.parquet"))
    entries = {p.stem: describe(pd.read_parquet(p)) for p in files}

    # 彙總雜湊：對「標的 → 內容雜湊」的正規化 JSON 再取一次 sha256。
    # 任何一檔的任何一個數字變了，這個值就會變——artifact 記的就是它。
    digest_payload = json.dumps(
        {k: v["sha256"] for k, v in entries.items()}, sort_keys=True
    ).encode("utf-8")

    return {
        "manifest_sha": hashlib.sha256(digest_payload).hexdigest(),
        "snapshot_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "auto_adjust": AUTO_ADJUST,
        "n_files": len(entries),
        "versions": {
            "yfinance": _package_version("yfinance"),
            "pandas": _package_version("pandas"),
            "pyarrow": _package_version("pyarrow"),
        },
        "files": entries,
    }


def manifest_path(cache_dir: Path = DATA_CACHE_DIR) -> Path:
    return cache_dir / MANIFEST_NAME


def write(manifest: dict, cache_dir: Path = DATA_CACHE_DIR) -> Path:
    path = manifest_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path


def read(cache_dir: Path = DATA_CACHE_DIR) -> dict | None:
    path = manifest_path(cache_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def current_sha(cache_dir: Path = DATA_CACHE_DIR) -> str | None:
    """已記錄的 manifest 彙總雜湊；沒有 manifest 時回 None。

    registry.save() 用它把 artifact 綁到資料快照。刻意讀檔而不重算——重算要掃全部
    parquet，訓練流程不該為了記一個字串付這個成本。
    """
    manifest = read(cache_dir)
    return manifest["manifest_sha"] if manifest else None


def diff(old: dict, new: dict) -> list[str]:
    """比對兩份 manifest，回傳人看得懂的差異描述（空 list = 完全一致）。"""
    old_files, new_files = old.get("files", {}), new.get("files", {})
    messages: list[str] = []

    for ticker in sorted(set(old_files) - set(new_files)):
        messages.append(f"{ticker}：快取中已消失")
    for ticker in sorted(set(new_files) - set(old_files)):
        messages.append(f"{ticker}：新增（manifest 中原本沒有）")

    for ticker in sorted(set(old_files) & set(new_files)):
        a, b = old_files[ticker], new_files[ticker]
        if a["sha256"] == b["sha256"]:
            continue
        changes = [
            f"{key} {a[key]} → {b[key]}"
            for key in ("rows", "first_date", "last_date", "first_close", "last_close")
            if a[key] != b[key]
        ]
        detail = "；".join(changes) if changes else "列數與起訖值相同，但中間有數值被改寫"
        messages.append(f"{ticker}：內容雜湊改變（{detail}）")

    return messages


def verify(cache_dir: Path = DATA_CACHE_DIR) -> list[str]:
    """以目前快取重算並與已記錄的 manifest 比對。回傳差異描述。"""
    recorded = read(cache_dir)
    if recorded is None:
        raise FileNotFoundError(
            f"找不到 {manifest_path(cache_dir)}，請先執行 python -m stockta.data.manifest build"
        )
    return diff(recorded, build(cache_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="資料快照雜湊")
    parser.add_argument("command", choices=["build", "verify", "show"])
    args = parser.parse_args()

    if args.command == "build":
        manifest = build()
        path = write(manifest)
        print(f"已寫入 {path}")
        print(f"  {manifest['n_files']} 檔、彙總雜湊 {manifest['manifest_sha'][:16]}…")
        return

    if args.command == "show":
        manifest = read()
        if manifest is None:
            raise SystemExit("尚無 manifest，請先 build")
        print(json.dumps(manifest, ensure_ascii=False, indent=2)[:4000])
        return

    differences = verify()
    if not differences:
        print("快取與 manifest 完全一致。")
        return
    print(f"與 manifest 有 {len(differences)} 處差異：")
    for line in differences:
        print(f"  - {line}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
