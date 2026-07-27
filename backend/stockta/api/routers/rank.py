"""Cross-sectional 相對強弱排名：GET /api/rank（可選 ?date= 歷史某日）

以 CS 模型對全池評分（P(未來 5 日贏過中位數)），排序回傳。無 date=即時；
帶 date=歷史 point-in-time 重算。GET /api/rank/summary 回傳回測摘要供前端策略卡。
唯讀分析端點，與 3 類 /prediction、/scan 並存。
"""

import json
from datetime import date, timedelta

from fastapi import APIRouter, Query, Request

from stockta.api.errors import DataSourceUnavailableError, InvalidRangeError
from stockta.api.schemas import RankResponse, RankResult, RankSummaryResponse
from stockta.config import (
    ARTIFACTS_CS_DIR,
    AUTO_ADJUST,
    CS_TOP_FRACTION,
    DATA_CACHE_DIR,
    INDICATOR_WARMUP_DAYS,
    STOCK_POOL,
    WINDOW_LENGTH_DAYS,
)
from stockta.data.cache import ParquetCache
from stockta.data.calendar import calendar_lookback_days, last_completed_trading_day
from stockta.data.provider import DataProvider, DataProviderError, YFinanceProvider
from stockta.inference.replay import context_fetch_start, load_pool_context
from stockta.ml.cross_sectional import score_asof

router = APIRouter(tags=["rank"])


def _rank_results(scored: list[tuple[str, float, date]]) -> list[RankResult]:
    """(ticker, score, base_date) 列表 → 排名結果（分數高=相對強=rank 1）。"""
    scored = sorted(scored, key=lambda x: x[1], reverse=True)
    n = len(scored)
    out = []
    for i, (ticker, score, _bd) in enumerate(scored):
        pct = 1 - (i / (n - 1)) if n > 1 else 1.0
        q = "top" if pct >= 1 - CS_TOP_FRACTION else "bottom" if pct <= CS_TOP_FRACTION else "mid"
        out.append(
            RankResult(
                ticker=ticker, name=STOCK_POOL.get(ticker, ""), score=score,
                rank=i + 1, percentile=pct, quantile=q,
            )
        )
    return out


@router.get("/api/rank", response_model=RankResponse)
def rank(
    request: Request,
    date_param: str | None = Query(None, alias="date", description="歷史查詢日 YYYY-MM-DD；省略=即時"),
) -> RankResponse:
    cs = getattr(request.app.state, "cs_model", None)
    end = last_completed_trading_day()
    if cs is None:
        return RankResponse(base_date=end, model="mock", results=[], is_mock=True)
    model, scaler, meta = cs

    as_of = None
    historical = False
    if date_param is not None:
        try:
            as_of = date.fromisoformat(date_param)
        except ValueError as exc:
            raise InvalidRangeError(f"日期格式需為 YYYY-MM-DD：{date_param!r}") from exc
        if as_of < end:
            historical = True

    scored: list[tuple[str, float, date]] = []
    base_seen = end
    if not historical:
        provider: DataProvider = request.app.state.data_provider
        start = end - timedelta(days=calendar_lookback_days(WINDOW_LENGTH_DAYS, INDICATOR_WARMUP_DAYS))
        try:
            context = request.app.state.market_context.get(end)
        except DataProviderError as exc:
            raise DataSourceUnavailableError(str(exc)) from exc
        for ticker in STOCK_POOL:
            try:
                df = provider.get_ohlcv(ticker, start, end)
            except DataProviderError:
                continue
            r = score_asof(model, scaler, df, context, None)
            if r is not None:
                scored.append((ticker, r[0], r[1]))
                base_seen = r[1]
    else:
        cache = ParquetCache(DATA_CACHE_DIR)
        provider = YFinanceProvider(cache=cache, auto_adjust=AUTO_ADJUST, max_cache_age_days=9999)
        try:
            pool_ohlcv, context = load_pool_context(provider, context_fetch_start(as_of), as_of)
        except DataProviderError as exc:
            raise DataSourceUnavailableError(str(exc)) from exc
        for ticker, df in pool_ohlcv.items():
            r = score_asof(model, scaler, df, context, as_of)
            if r is not None:
                scored.append((ticker, r[0], r[1]))
                base_seen = r[1]

    if not scored:
        raise DataSourceUnavailableError("無足夠資料可排名（快取過舊或日期過早）")
    return RankResponse(
        base_date=base_seen, model=meta.get("model_name", "cs"),
        is_historical=historical, results=_rank_results(scored),
    )


@router.get("/api/rank/summary", response_model=RankSummaryResponse)
def rank_summary() -> RankSummaryResponse:
    path = ARTIFACTS_CS_DIR / "summary.json"
    if not path.exists():
        return RankSummaryResponse(available=False)
    s = json.loads(path.read_text(encoding="utf-8"))
    return RankSummaryResponse(
        available=True,
        model=s.get("model"),
        test_rank_ic=s.get("test_rank_ic"),
        test_rank_ic_t=s.get("test_rank_ic_t"),
        val_rank_ic=s.get("val_rank_ic"),
        holding_days=s.get("holding_days"),
        net_cum=s.get("net_cum"),
        bench_cum=s.get("bench_cum"),
        net_excess_cum=s.get("net_excess_cum"),
        win_rate=s.get("win_rate"),
        cost_bps=s.get("cost_bps"),
    )
