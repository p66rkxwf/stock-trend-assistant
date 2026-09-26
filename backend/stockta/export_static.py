"""靜態站匯出：python -m stockta.export_static --out ../frontend/public/data

雲端部署（Cloudflare Pages）沒有常駐後端：每日排程跑完 record_* 之後，以 TestClient
逐一呼叫現有 API，把回應原樣寫成 JSON，前端 staticApi（frontend/lib/api.ts）改讀這些檔案。
走 API 本身的程式路徑＝靜態站與本機 uvicorn 的數字同源，不另寫一套計算。

帶參數的端點改為「匯出最大範圍、前端切片」：
- candles：匯出近 5 年，前端依 range 或自訂區間切
- history：匯出近 HISTORY_EXPORT_DAYS 天，前端依 (start, end] 篩選並重算命中率
- scan / rank：匯出即時 + 最近 SCAN_HISTORY_SESSIONS 個交易日，其餘日期前端回報超出範圍

匯出期間資料源換成「只讀快取」的 provider：匯出內容＝record 步驟當下看到的資料，
不會在匯出途中又抓到一根新 K 棒。/prediction 的落地副作用也關掉——線上實證只由
record_predictions 寫入（與 /scan 的設計原則一致）。

關卡（任一不過 → exit 1；workflow 不部署，線上維持上一版）：
- 模型必須真的載入（is_mock=false）：找不到 artifact 時 API 會退回 mock，不能把假預測發布出去
- 全站端點必須 200
- 個股檔案覆蓋率 ≥ MIN_TICKER_COVERAGE
- 資料日期不得早於上一版（--prev）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

CANDLE_EXPORT_RANGE = "5y"
HISTORY_EXPORT_DAYS = 730
SCAN_HISTORY_SESSIONS = 60
MIN_TICKER_COVERAGE = 0.9

# 每檔匯出的端點：(API 路徑樣板, 檔名)；{t}=ticker、{history_start}/{last_day} 由匯出時代入
TICKER_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("/api/stocks/{t}/prediction", "prediction.json"),
    ("/api/stocks/{t}/indicators", "indicators.json"),
    ("/api/stocks/{t}/predictions", "predictions.json"),
    (f"/api/stocks/{{t}}/candles?range={CANDLE_EXPORT_RANGE}", "candles.json"),
    ("/api/stocks/{t}/history?start={history_start}&end={last_day}", "history.json"),
)


class ExportError(Exception):
    """關卡未通過：不得部署。"""


@dataclass
class ExportResult:
    written: int = 0
    failures: list[str] = field(default_factory=list)


class _Dumper:
    def __init__(self, client, out: Path, log):
        self._client = client
        self._out = out
        self._log = log
        self.result = ExportResult()

    def dump(self, url: str, relpath: str, required: bool = False) -> dict | None:
        resp = self._client.get(url)
        if resp.status_code != 200:
            msg = f"{url} → HTTP {resp.status_code}: {resp.text[:200]}"
            if required:
                raise ExportError(msg)
            self.result.failures.append(msg)
            self._log(f"[略過] {msg}")
            return None
        path = self._out / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(resp.content)
        self.result.written += 1
        return resp.json()


def _require_real(payload: dict, what: str) -> None:
    if payload.get("is_mock"):
        raise ExportError(f"{what} 為 mock 回應（模型未載入）——拒絕發布假資料")


def run_export(
    client,
    out: Path,
    *,
    tickers: Iterable[str],
    last_day: date,
    data_as_of: date,
    scan_dates: list[date],
    ticker_endpoints: tuple[tuple[str, str], ...] = TICKER_ENDPOINTS,
    prev_meta: dict | None = None,
    log=print,
) -> dict:
    """逐一呼叫 API 寫出 JSON；回傳 meta。關卡不過拋 ExportError。"""
    from stockta.api.routers.stocks import _RANGE_TO_DAYS

    if prev_meta and prev_meta.get("data_as_of", "") > data_as_of.isoformat():
        raise ExportError(
            f"資料日期倒退：本次 {data_as_of}、上一版 {prev_meta['data_as_of']}（快取還原失敗？）"
        )

    d = _Dumper(client, out, log)
    tickers = list(tickers)

    model = d.dump("/api/model", "model.json", required=True)
    _require_real(model, "/api/model")
    d.dump("/api/stocks", "stocks.json", required=True)
    d.dump("/api/market", "market.json", required=True)
    d.dump("/api/track-record", "track-record.json", required=True)
    d.dump("/api/rank/summary", "rank/summary.json", required=True)

    scan = d.dump("/api/scan", "scan/latest.json", required=True)
    _require_real(scan, "/api/scan")
    rank = d.dump("/api/rank", "rank/latest.json", required=True)
    _require_real(rank, "/api/rank")

    t0 = time.monotonic()
    for kind in ("scan", "rank"):
        done = []
        for day in scan_dates:
            if d.dump(f"/api/{kind}?date={day.isoformat()}", f"{kind}/{day.isoformat()}.json"):
                done.append(day.isoformat())
        # sessions＝交易日曆、dates＝實際匯出成功的日子；前端先把查詢日對齊到 ≤ 它的交易日
        # （同 API 的 point-in-time 規則），對齊到的日子若沒匯出就回報錯誤，而不是改給別天
        _write_json(out / kind / "index.json", {
            "latest": (scan if kind == "scan" else rank)["base_date"],
            "sessions": [day.isoformat() for day in scan_dates],
            "dates": done,
        })
        log(f"{kind}：即時 + 歷史 {len(done)}/{len(scan_dates)} 日（{time.monotonic() - t0:.0f}s）")

    history_start = last_day - timedelta(days=HISTORY_EXPORT_DAYS)
    ticker_ok = 0
    ticker_total = 0
    for i, t in enumerate(tickers, 1):
        for tmpl, name in ticker_endpoints:
            url = tmpl.format(t=t, history_start=history_start.isoformat(), last_day=last_day.isoformat())
            ticker_total += 1
            ticker_ok += d.dump(url, f"stocks/{t}/{name}") is not None
        if i % 10 == 0 or i == len(tickers):
            log(f"個股 {i}/{len(tickers)}（{time.monotonic() - t0:.0f}s）")

    coverage = ticker_ok / ticker_total if ticker_total else 1.0
    if coverage < MIN_TICKER_COVERAGE:
        raise ExportError(
            f"個股檔案覆蓋率 {coverage:.0%} < {MIN_TICKER_COVERAGE:.0%}：\n  " + "\n  ".join(d.result.failures[:20])
        )

    candle_days = _RANGE_TO_DAYS[CANDLE_EXPORT_RANGE]
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # API 的推論錨點（last_completed_trading_day）；前端切 candles 區間以此為終點
        "last_trading_day": last_day.isoformat(),
        # 實際最後一根 K 棒（^TWII）；國定假日時會早於 last_trading_day
        "data_as_of": data_as_of.isoformat(),
        "model_version": model["model_version"],
        "cs_model": rank.get("model"),
        "range_days": _RANGE_TO_DAYS,
        "candles_start": (last_day - timedelta(days=candle_days)).isoformat(),
        "history_start": history_start.isoformat(),
        "coverage": round(coverage, 4),
        "failures": d.result.failures,
        "commit": os.environ.get("GITHUB_SHA"),
        "run_url": _run_url(),
    }
    _write_json(out / "meta.json", meta)
    log(f"完成：{d.result.written} 檔，覆蓋率 {coverage:.1%}，資料截至 {data_as_of}")
    return meta


def _run_url() -> str | None:
    server, repo, run = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{server}/{repo}/actions/runs/{run}" if server and repo and run else None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _trading_dates(last_day: date, sessions: int) -> tuple[date, list[date]]:
    """以 ^TWII 快取索引為交易日曆：回傳 (最後一根 K 棒日, last_day 之前最近 sessions 個交易日)。

    ≥ last_day 的日期 API 一律走即時掃描，所以歷史檔只收嚴格早於 last_day 的交易日。
    """
    from stockta.config import DATA_CACHE_DIR, MARKET_INDEX_TICKER
    from stockta.data.cache import ParquetCache

    market = ParquetCache(DATA_CACHE_DIR).read(MARKET_INDEX_TICKER)
    if market is None or market.empty:
        raise ExportError(f"找不到 {MARKET_INDEX_TICKER} 快取")
    days = [ts.date() for ts in market.index if ts.date() <= last_day]
    if not days:
        raise ExportError(f"{MARKET_INDEX_TICKER} 快取沒有 {last_day} 以前的資料")
    past = [x for x in days if x < last_day]
    return days[-1], past[-sessions:]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="匯出靜態站 JSON")
    parser.add_argument("--out", type=Path, required=True, help="輸出目錄（frontend/public/data）")
    parser.add_argument("--prev", type=Path, help="上一版 meta.json；資料日期倒退則拒絕")
    parser.add_argument("--sessions", type=int, default=SCAN_HISTORY_SESSIONS, help="scan/rank 歷史交易日數")
    parser.add_argument("--tickers", nargs="+", help="只匯出指定標的（除錯用）")
    args = parser.parse_args(argv)

    from fastapi.testclient import TestClient

    from stockta.api import main as api_main
    from stockta.config import AUTO_ADJUST, DATA_CACHE_DIR, STOCK_POOL
    from stockta.data.cache import ParquetCache
    from stockta.data.calendar import last_completed_trading_day
    from stockta.data.provider import YFinanceProvider
    from stockta.inference.market_context import MarketContextService

    prev_meta = None
    if args.prev and args.prev.exists():
        prev_meta = json.loads(args.prev.read_text(encoding="utf-8"))

    last_day = last_completed_trading_day()
    try:
        data_as_of, scan_dates = _trading_dates(last_day, args.sessions)
        with TestClient(api_main.app) as client:
            state = api_main.app.state
            state.limiter.enabled = False  # 同一個 "testclient" IP 連打數百次，會誤觸每分鐘限流
            cache = ParquetCache(DATA_CACHE_DIR)
            offline = YFinanceProvider(cache=cache, auto_adjust=AUTO_ADJUST, max_cache_age_days=9999)
            state.data_provider = offline
            state.market_context = MarketContextService(offline, cache, list(STOCK_POOL))
            state.prediction_store = None
            run_export(
                client,
                args.out,
                tickers=args.tickers or list(STOCK_POOL),
                last_day=last_day,
                data_as_of=data_as_of,
                scan_dates=scan_dates,
                prev_meta=prev_meta,
            )
    except ExportError as exc:
        print(f"[匯出失敗] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
