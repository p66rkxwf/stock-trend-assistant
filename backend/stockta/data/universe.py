"""Point-in-time 成分股：回答「在 d 這一天，台灣 50 的成分股是哪 50 檔？」

**要解決的問題**：`config.STOCK_POOL` 是 2026-07 的台灣 50 成分股快照，卻被拿來
訓練與回測 20 年。世芯-KY、緯穎、奇鋐這些近年才入榜的贏家，在 2010 年根本不在
指數裡——用今天的名單回測過去，等於預先知道誰會贏。這是倖存者偏誤的教科書形態。

**作法**：把指數的成分股**變動事件**（何時納入、何時剔除）存成可稽核的事件表，
再重放成任一日期的成分股集合。事件表的每一列都必須帶 `source_url`，
沒有出處的列不准進表——這份資料的價值完全建立在可查證上。

介面形狀刻意對齊 quant-trading-platform 的
`qtp/backtest/engine.py::run_backtest(universe: dict[date, set[str]])`，
兩個專案對「as-of universe」講同一種語言。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd

from stockta.config import BACKEND_ROOT

UNIVERSE_DIR = BACKEND_ROOT / "data" / "universe"
EVENTS_PATH = UNIVERSE_DIR / "tw50_events.csv"

# 台灣 50 指數固定 50 檔。這不是我們的假設，是指數規則明訂的
# （Ground Rules v3.5 §6.3.5「A constant number of constituents will be maintained」），
# 所以拿它當一致性檢查是站得住腳的：任一時點重放出來若不是 50 檔，就是事件表有缺漏。
INDEX_SIZE = 50

# 定期審核的緩衝區（Ground Rules v3.5 §6.3.2 / §6.3.3）：
#   納入——市值排名「升到第 40 名或以上」
#   剔除——市值排名「掉到第 61 名或以下」
#
# **這兩個數字是本模組存在的理由。** 因為納入與剔除的門檻不同，排名 41~60 之間
# 是一段緩衝帶：同樣排在第 50 名，原本在裡面的留著、原本在外面的進不來。
# 也就是說**成分資格取決於前一期的狀態**，不是當期市值的函數。
#
# 直接後果：**不能靠「把當時的市值排序取前 50 名」來重建歷史成分股**——
# 那會在緩衝帶裡系統性地猜錯。要知道 d 當天誰在裡面，只能從某個已知狀態
# 逐次套用每一次審核的異動，也就是本模組做的事。
REVIEW_INSERT_RANK = 40
REVIEW_DELETE_RANK = 61

ADD = "add"
DROP = "drop"


class UniverseError(RuntimeError):
    """成分股事件表不自洽（順序矛盾、規模不符等）。"""


@dataclass(frozen=True)
class MembershipEvent:
    effective_date: date
    ticker: str
    action: str  # add | drop
    review: str  # 定期審核 | 臨時調整
    source_url: str
    name: str = ""  # 公司名稱，讓事件表用肉眼就能核對
    note: str = ""  # 備註：公告日、疑義、資料來源的細節

    @property
    def label(self) -> str:
        return f"{self.name}({self.ticker})" if self.name else self.ticker


def load_events(path: Path = EVENTS_PATH) -> list[MembershipEvent]:
    """讀取事件表，依生效日排序。缺檔時拋出明確錯誤。"""
    if not path.exists():
        raise FileNotFoundError(
            f"找不到成分股事件表 {path}。\n"
            "PIT universe 需要人工策展的事件表，來源與格式見 data/universe/README.md。"
        )

    events: list[MembershipEvent] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            if not (row.get("ticker") or "").strip():
                continue  # 允許空行
            action = (row["action"] or "").strip().lower()
            if action not in (ADD, DROP):
                raise UniverseError(f"{path}:{line_no} action 必須是 add 或 drop，收到 {action!r}")
            if not (row.get("source_url") or "").strip():
                raise UniverseError(
                    f"{path}:{line_no} 缺 source_url。沒有出處的列不准進表——"
                    "這份資料的價值完全建立在可查證上。"
                )
            events.append(
                MembershipEvent(
                    effective_date=date.fromisoformat(row["effective_date"].strip()),
                    ticker=row["ticker"].strip(),
                    action=action,
                    review=(row.get("review") or "").strip(),
                    source_url=row["source_url"].strip(),
                    name=(row.get("name") or "").strip(),
                    note=(row.get("note") or "").strip(),
                )
            )

    events.sort(key=lambda e: (e.effective_date, e.ticker, e.action))
    return events


def _replay(events: list[MembershipEvent]) -> list[tuple[date, frozenset[str]]]:
    """重放事件，回傳 [(生效日, 該日起的成分股集合)]，依日期遞增。"""
    members: set[str] = set()
    timeline: list[tuple[date, frozenset[str]]] = []

    for effective_date, group in _group_by_date(events):
        for event in group:
            if event.action == ADD:
                if event.ticker in members:
                    raise UniverseError(
                        f"{effective_date} {event.ticker} 已在成分股內卻又被納入——事件表重複"
                    )
                members.add(event.ticker)
            else:
                if event.ticker not in members:
                    raise UniverseError(
                        f"{effective_date} {event.ticker} 不在成分股內卻被剔除——"
                        "事件表順序矛盾（可能漏了更早的納入事件）"
                    )
                members.discard(event.ticker)
        timeline.append((effective_date, frozenset(members)))

    return timeline


def _group_by_date(events: list[MembershipEvent]):
    """同一生效日的事件必須一起套用（同時有納入與剔除，中途規模不會是 50）。"""
    current_date: date | None = None
    bucket: list[MembershipEvent] = []
    for event in events:
        if event.effective_date != current_date:
            if bucket:
                yield current_date, bucket
            current_date, bucket = event.effective_date, []
        bucket.append(event)
    if bucket:
        yield current_date, bucket


@lru_cache(maxsize=4)
def _timeline(path: Path = EVENTS_PATH) -> tuple[tuple[date, frozenset[str]], ...]:
    return tuple(_replay(load_events(path)))


def members_asof(when: date, path: Path = EVENTS_PATH) -> set[str]:
    """when 當日（含）生效的成分股集合。早於事件表起點時回傳空集合。"""
    timeline = _timeline(path)
    result: frozenset[str] = frozenset()
    for effective_date, members in timeline:
        if effective_date > when:
            break
        result = members
    return set(result)


def third_friday(year: int, month: int) -> date:
    """該月的第三個星期五。"""
    first = date(year, month, 1)
    # weekday(): Mon=0 … Fri=4
    first_friday = 1 + (4 - first.weekday()) % 7
    return date(year, month, first_friday + 14)


def last_review_effective(on: date) -> date:
    """on（含）之前最近一次定期審核的生效日。

    台灣 50 指數每年 3/6/9/12 月定期審核，於該月**第三個星期五收盤後**生效
    （FTSE 公告用語："after the close of business on Friday, 20 March 2026,
    i.e. on Monday, 23 March 2026"）。

    這個函式的用途是**讓過期的股票池藏不住**：成分股每季會變，把名單寫死在
    config 裡的那一刻起它就開始腐壞，而腐壞不會有任何錯誤訊息。
    """
    candidates = [
        third_friday(year, month)
        for year in (on.year, on.year - 1)
        for month in (3, 6, 9, 12)
    ]
    return max(d for d in candidates if d <= on)


def pool_staleness(as_of: date, today: date | None = None) -> tuple[int, date]:
    """股票池自 as_of 之後錯過了幾次定期審核。回傳 (錯過次數, 最近一次生效日)。"""
    today = today or date.today()
    missed = 0
    cursor = last_review_effective(today)
    latest = cursor
    while cursor > as_of:
        missed += 1
        cursor = last_review_effective(cursor - timedelta(days=1))
    return missed, latest


def coverage_start(path: Path = EVENTS_PATH) -> date | None:
    """事件表能回答問題的最早日期（第一次達到 INDEX_SIZE 檔的那天）。

    在此之前重放出來的集合不完整，不該拿去做 PIT 回測——回傳 None 表示
    事件表從未湊滿過 INDEX_SIZE 檔。
    """
    for effective_date, members in _timeline(path):
        if len(members) == INDEX_SIZE:
            return effective_date
    return None


def all_tickers_ever(path: Path = EVENTS_PATH) -> set[str]:
    """曾經在指數內出現過的所有代號——含已下市與被併購者。

    這是抓價格資料時該用的清單：只抓現存成分股，倖存者偏誤就從資料源頭進來了。
    """
    return {event.ticker for event in load_events(path)}


def membership_frame(dates, path: Path = EVENTS_PATH) -> pd.DataFrame:
    """逐日成分股寬表（index=日期、columns=代號、值為 bool）。

    features/market.py 算市場寬度、ml/cross_sectional.py 取當日中位數時使用——
    兩者目前都用今天的股票池計算整段歷史，那本身就是被污染的特徵。
    """
    index = pd.DatetimeIndex(dates)
    tickers = sorted(all_tickers_ever(path))
    rows = [
        [ticker in members_asof(stamp.date(), path) for ticker in tickers] for stamp in index
    ]
    return pd.DataFrame(rows, index=index, columns=tickers, dtype=bool)


def validate(path: Path = EVENTS_PATH) -> list[str]:
    """檢查事件表自洽性，回傳問題描述（空 list = 通過）。

    最強的一條是規模檢查：台灣 50 永遠是 50 檔，重放後不是 50 檔就是事件表有缺漏。
    刻意允許事件表起點之前的「暖身期」（成分股還沒湊滿），只檢查湊滿之後。
    """
    problems: list[str] = []

    # 先做便宜的逐列檢查，再做需要完整重放的檢查。順序相反的話，重放一旦因為
    # 資料不全而中斷，後面所有問題就都看不到了——一次只能修一個問題最令人洩氣。
    try:
        events = load_events(path)
    except (UniverseError, FileNotFoundError) as exc:
        return [str(exc)]

    # 定期審核的生效日必然是 3/6/9/12 月的第三個星期五（臨時調整才會落在別的日子）。
    # 對不上就是抄錯了——財經報導把生效日寫差一天是常見的錯，靠肉眼看不出來。
    seen_dates: set[date] = set()
    for event in events:
        if event.review != "定期審核" or event.effective_date in seen_dates:
            continue
        seen_dates.add(event.effective_date)
        expected = third_friday(event.effective_date.year, event.effective_date.month)
        if event.effective_date != expected:
            problems.append(
                f"{event.effective_date}：定期審核的生效日應為當月第三個星期五"
                f"（{expected}），對不上——請回查原始公告"
            )

    try:
        timeline = _timeline(path)
    except UniverseError as exc:
        return [*problems, str(exc)]

    if not timeline:
        return [*problems, "事件表是空的"]

    started = False
    for effective_date, members in timeline:
        if len(members) == INDEX_SIZE:
            started = True
        elif started:
            problems.append(
                f"{effective_date}：成分股 {len(members)} 檔，應為 {INDEX_SIZE} 檔"
                f"（{'少' if len(members) < INDEX_SIZE else '多'}"
                f" {abs(len(members) - INDEX_SIZE)} 檔）"
            )

    if not started:
        problems.append(
            f"事件表從未湊滿 {INDEX_SIZE} 檔（最多 "
            f"{max(len(m) for _, m in timeline)} 檔）——起始成分股名單可能不完整"
        )
    return problems


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="PIT 成分股事件表")
    parser.add_argument("command", choices=["validate", "asof", "coverage"])
    parser.add_argument("--date", help="asof 查詢日期（YYYY-MM-DD）")
    args = parser.parse_args()

    if args.command == "validate":
        problems = validate()
        if not problems:
            events = load_events()
            start = coverage_start()
            print(
                f"事件表自洽：{len(events)} 個事件、"
                f"涵蓋 {len(all_tickers_ever())} 個曾入榜代號、"
                f"可回答日期起自 {start}"
            )
            return
        print(f"發現 {len(problems)} 個問題：")
        for line in problems[:40]:
            print(f"  - {line}")
        raise SystemExit(1)

    if args.command == "coverage":
        print(f"可回答的最早日期：{coverage_start()}")
        print(f"曾入榜代號數：{len(all_tickers_ever())}")
        return

    if not args.date:
        raise SystemExit("asof 需要 --date")
    members = members_asof(date.fromisoformat(args.date))
    print(f"{args.date} 成分股 {len(members)} 檔：")
    print("  " + "、".join(sorted(members)))


if __name__ == "__main__":
    main()
