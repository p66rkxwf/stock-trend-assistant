"""每個特徵的「最早可用時間」——把「這個值在當時真的存在嗎」寫成可檢查的東西。

散文式的註解沒有約束力：寫「本特徵只用當日資料」不會在有人加了前視特徵時亮紅燈。
所以這裡把可用時點編碼成 (trading_days_offset, gate)，由
tests/test_feature_availability.py 逐項驗證：

- **trading_days_offset**：相對基準日 t 的交易日位移。特徵一律必須 ≤ 0；
  只要有人標了正數，就是自己承認用了未來資訊，測試直接擋下。
- **gate**：位移之外還需要什麼到位才算真的可用。這一欄回答的是清單上
  「三者差幾小時」那題——同樣是「t 日收盤後」，個股 K 線與**全池**寬度到位的
  時間差了幾十分鐘，而排程是在 15:00 跑的。

標籤刻意也放進同一套表示法：它的 offset 是 +LABEL_HORIZON_DAYS，與所有特徵的
≤ 0 形成對比，「特徵與標籤的時間隔離」因此變成一條算術上成立的敘述，而不是宣稱。
"""

from __future__ import annotations

from stockta.config import LABEL_HORIZON_DAYS

# (相對 t 的交易日位移, 還需要什麼到位)
Availability = tuple[int, str]

# 標籤：未來 LABEL_HORIZON_DAYS 個交易日的累積報酬，故位移為正。
# 所有特徵都必須嚴格早於它——這是整條管線的核心不變式。
LABEL_AVAILABILITY: Availability = (
    LABEL_HORIZON_DAYS,
    f"該檔 t+{LABEL_HORIZON_DAYS} 交易日的收盤 K 線到位",
)


def latest_offset(availabilities: dict[str, Availability]) -> int:
    """一組特徵中最晚的可用位移。"""
    return max(offset for offset, _ in availabilities.values())


def gates(availabilities: dict[str, Availability]) -> set[str]:
    """出現過的所有到位條件（產生文件時用來列出決策時點的組成）。"""
    return {gate for _, gate in availabilities.values()}


def _write_report() -> None:
    """產生 docs/feature_availability.md：python -m stockta.features.availability"""
    from stockta.config import (
        BACKEND_ROOT,
        MARKET_CLOSE_HOUR,
        MARKET_CLOSE_MINUTE,
        MARKET_INCLUDE_REGIME,
    )
    from stockta.features.pipeline import FEATURE_AVAILABILITY, FEATURE_COLUMNS

    close_hhmm = f"{MARKET_CLOSE_HOUR:02d}:{MARKET_CLOSE_MINUTE:02d}"
    in_production = set(FEATURE_COLUMNS)

    lines = [
        "# 特徵的最早可用時間",
        "",
        "「你的每一個特徵，在你宣稱可以使用它的那個時間點，真的已經存在嗎？」",
        "這份表格是那個問題的逐欄答案，由 `stockta/features/availability.py` 產生，",
        "並由 `tests/test_feature_availability.py` 逐項驗證——**新增特徵而沒有標註，",
        "測試會紅燈**。散文式的「本特徵只用當日資料」沒有這種約束力。",
        "",
        "- **位移**：相對基準日 t 的交易日位移。特徵一律 ≤ 0。",
        "- **到位條件**：位移之外，還需要什麼資料實際抵達才算真的可用。",
        "  同樣是「t 日收盤後」，個股 K 線與**全池**寬度到位的時間差很多——",
        "  這一欄就是清單上「三者差幾小時」那題的答案。",
        "",
        "| 特徵 | 位移（交易日） | 到位條件 | 正式管線使用中 |",
        "|---|---|---|---|",
    ]
    def fmt(offset: int) -> str:
        return f"{offset:+d}" if offset else "0"

    for column, (offset, gate) in FEATURE_AVAILABILITY.items():
        used = "✅" if column in in_production else "—"
        lines.append(f"| `{column}` | {fmt(offset)} | {gate} | {used} |")

    label_offset, label_gate = LABEL_AVAILABILITY
    lines += [
        f"| **標籤** | **{fmt(label_offset)}** | **{label_gate}** | ✅ |",
        "",
        f"未使用的欄位是 regime 特徵（`MARKET_INCLUDE_REGIME={MARKET_INCLUDE_REGIME}`，",
        "實驗 #9 判定不採用）。它們仍然標註，否則哪天打開旗標，就會有四個沒人",
        "檢查過可用時點的特徵直接進正式管線。",
        "",
        "## 特徵與標籤的時間隔離",
        "",
        f"所有特徵位移 ≤ 0、標籤位移 = {fmt(label_offset)}。兩者不重疊是**算術上成立**的，",
        "不是宣稱——`test_label_is_strictly_later_than_every_feature` 就在驗這件事。",
        "",
        "切分邊界上還有第二道：`ml/dataset.py` 的 embargo 會剔除「標籤日跨進下一個",
        "切分期」的樣本，避免訓練集的標籤落在驗證期之內。",
        "",
        "## 決策時點：宣告的可用時間 vs 實際到位時間",
        "",
        "上表的位移都是 0，看起來每個特徵一樣早。**實際上不是。**",
        "",
        f"- 台股 {close_hhmm} 收盤。`data/calendar.py` 的 `last_completed_trading_day()`",
        f"  在 {close_hhmm} 之前一律回退到前一個交易日——盤中呼叫不會拿到半根 K 棒。",
        "- 每日排程（`daily_predict.bat`）在收盤後執行。",
        "- 但**資料商的當日 K 線不保證那時已經到位**。",
        "",
        "### 這不是假設，是實測到的",
        "",
        "2026-08-11~13 追查線上預測漏日時確認：根因不是排程沒跑，而是",
        "**yfinance 的當日 K 線在排程執行時尚未到位**，於是當次記到較舊的基準日。",
        "",
        "對策已經落地在 `ml/record_predictions.py`：每次回補最近 N 個交易日",
        "（N 預設 = 標籤天數），跳過已記錄的日子。這使排程**冪等且自癒**——",
        "當日抓不到就隔日補回，已補回 08-06／08-10／08-12 三日。",
        "",
        "回補視窗刻意 ≤ 標籤天數：回補日的 N 日後結果**尚未實現**，",
        "所以補記的預測仍然是誠實的樣本外預測，不因回補而取得任何未來資訊。",
        "",
        "### 最晚到位的是市場寬度",
        "",
        "個股特徵只要該檔 K 線到位就能算；市場寬度要等**全池**至少 30 檔到位。",
        "決策時點因此由寬度決定，而不是由個股 K 線決定。",
        "`ml/record_predictions.py` 的「先抓全池再推論」就是為了這件事。",
        "",
        "> 由 `python -m stockta.features.availability` 自動產生。",
        "",
    ]

    out = BACKEND_ROOT.parent / "docs" / "feature_availability.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"報告已寫入 {out}")


if __name__ == "__main__":
    _write_report()
