"""股票池稽核：python -m stockta.data.pool_audit events|check

`config.STOCK_POOL` 自稱是台灣 50 成分股快照，但它是人工維護的，會過期。
過期的池子不只讓回測有倖存者偏誤，**還會讓線上每天對已被剔除的股票發訊號**。

兩個子指令：

- `events`：把 STOCK_POOL 與 `data/universe/tw50_events.csv` 裡已查證的指數調整
  對照，列出「應納入但池中沒有」與「應剔除但池中還在」。不需連網。
- `check`：吃一份**你信得過的**成分股代號清單，逐檔向 yfinance 查證代號與名稱
  是否對得上，再與 STOCK_POOL 比對，印出換上去會變動什麼。

為什麼 `check` 要重新查證名稱：整理成分股名單時最容易犯、也最難發現的錯，
是代號與名稱錯位（把 3034 寫成台光電、把 2311 寫成日月光投控）。錯位的池子
在程式裡完全不會報錯，只會安靜地抓錯一檔股票的資料。所以名稱一律以資料源為準，
不接受人工輸入的名稱。
"""

from __future__ import annotations

import argparse
import time

from stockta.config import STOCK_POOL

PAUSE_SECONDS = 0.4  # 對 Yahoo 溫柔一點


def _to_ticker(code: str) -> str:
    code = code.strip().upper()
    return code if code.endswith(".TW") else f"{code}.TW"


def audit_against_events() -> int:
    """以已查證的指數調整事件對照 STOCK_POOL。回傳發現的問題數。"""
    from datetime import date

    from stockta.config import STOCK_POOL_ASOF
    from stockta.data.universe import INDEX_SIZE, load_events, pool_staleness

    events = load_events()
    pool = set(STOCK_POOL)

    problems = 0
    by_date: dict = {}
    for event in events:
        by_date.setdefault(event.effective_date, []).append(event)

    print(f"STOCK_POOL 檔數：{len(STOCK_POOL)}（台灣 50 指數規模為 {INDEX_SIZE}）")
    if len(STOCK_POOL) != INDEX_SIZE:
        print(f"  ⚠️ 檔數就對不上，差 {INDEX_SIZE - len(STOCK_POOL)} 檔")
        problems += 1

    # 成分股每季調整，所以「多久沒核對」本身就是一個要監看的量
    as_of = date.fromisoformat(STOCK_POOL_ASOF)
    missed, latest = pool_staleness(as_of)
    print(f"最後核對日：{as_of}；最近一次定期審核生效日：{latest}")
    if missed:
        print(f"  ⚠️ 自最後核對以來已錯過 {missed} 次定期審核")
        problems += missed
    print()

    for effective_date in sorted(by_date):
        group = by_date[effective_date]
        missing = [e for e in group if e.action == "add" and e.ticker not in pool]
        stale = [e for e in group if e.action == "drop" and e.ticker in pool]
        print(f"[{effective_date} 生效]")
        print(
            "  應納入但池中沒有："
            + ("、".join(e.label for e in missing) or "無")
        )
        print(
            "  應剔除但池中還在："
            + ("、".join(e.label for e in stale) or "無")
        )
        problems += len(missing) + len(stale)

    print()
    if problems:
        print(f"共 {problems} 處不一致——股票池已過期。")
        print("線上影響：/api/scan 與 /api/rank 都在這個池子上跑。")
    else:
        print("股票池與已查證的調整事件一致。")
    return problems


def verify_and_diff(codes: list[str]) -> int:
    """逐檔向 yfinance 查證代號，再與 STOCK_POOL 比對。回傳查證失敗的檔數。"""
    import yfinance as yf

    tickers = [_to_ticker(c) for c in codes]
    if len(set(tickers)) != len(tickers):
        print("⚠️ 輸入清單有重複代號")

    print(f"查證 {len(tickers)} 檔代號（向 yfinance 取得正式名稱）…\n")
    resolved: dict[str, str] = {}
    failed: list[str] = []
    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or ""
        except Exception as exc:
            name = ""
            print(f"  [{i}/{len(tickers)}] {ticker} 查詢失敗：{exc}")
        if not name:
            failed.append(ticker)
        else:
            resolved[ticker] = name
            existing = STOCK_POOL.get(ticker)
            flag = "" if existing is None else f"（池中記為「{existing}」）"
            print(f"  [{i}/{len(tickers)}] {ticker} → {name} {flag}")
        time.sleep(PAUSE_SECONDS)

    if failed:
        print(f"\n⚠️ {len(failed)} 檔查不到名稱，代號可能有誤：{'、'.join(failed)}")

    pool = set(STOCK_POOL)
    incoming = set(resolved)
    print("\n=== 換成這份清單會變動什麼 ===")
    print(f"  新增 {len(incoming - pool)} 檔：")
    for ticker in sorted(incoming - pool):
        print(f"    + {ticker} {resolved[ticker]}")
    print(f"  移除 {len(pool - incoming)} 檔：")
    for ticker in sorted(pool - incoming):
        print(f"    - {ticker} {STOCK_POOL[ticker]}")
    print(f"  維持 {len(pool & incoming)} 檔")

    if incoming != pool:
        print(
            "\n**注意：換池子必須重訓。** 市場寬度（breadth_up / breadth_ma5）"
            "\n是用股票池算出來的，改了池子，整段歷史的 context 就跟著變，"
            "\n現有五個 artifact 都是用舊池子的寬度訓練的。"
            "\n特徵欄位名稱沒變，registry 的契約比對**擋不下這種變動**——"
            "\n這正是換池子時最容易忽略的地方。"
        )
    return len(failed)


def main() -> None:
    parser = argparse.ArgumentParser(description="股票池稽核")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("events", help="與已查證的指數調整事件對照（不連網）")
    check = sub.add_parser("check", help="查證一份代號清單並與 STOCK_POOL 比對")
    check.add_argument("--codes", nargs="+", help="代號清單，如 2330 2454 3711")
    check.add_argument("--from-file", help="每行一個代號的文字檔")

    args = parser.parse_args()

    if args.command == "events":
        raise SystemExit(1 if audit_against_events() else 0)

    codes = list(args.codes or [])
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as handle:
            codes += [line.split(",")[0].strip() for line in handle if line.strip()]
    if not codes:
        raise SystemExit("check 需要 --codes 或 --from-file")
    raise SystemExit(1 if verify_and_diff(codes) else 0)


if __name__ == "__main__":
    main()
