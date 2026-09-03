"""PIT 成分股：重放事件表得到任一日期的成分股集合，且事件表必須自洽。

倖存者偏誤的修法本身也可能出錯——重放順序弄反、事件漏了一筆，
都會安靜地給出一份「看起來很合理」的錯名單。這裡把自洽性檢查釘死。
"""

from datetime import date

import pytest

from stockta.data.universe import (
    INDEX_SIZE,
    UniverseError,
    all_tickers_ever,
    coverage_start,
    last_review_effective,
    members_asof,
    membership_frame,
    pool_staleness,
    third_friday,
    validate,
)

pytestmark = pytest.mark.leakage

_SOURCE = "https://example.invalid/review/2020Q1"


def _write_events(tmp_path, rows: list[tuple[str, str, str]], review: str = "臨時調整"):
    """rows = [(生效日, 代號, add|drop)]。

    review 預設「臨時調整」：合成資料用的是隨手挑的日期，不該冒充定期審核——
    定期審核的生效日受「當月第三個星期五」規則約束，validate() 會檢查。
    要驗那條規則的測試自己傳 review="定期審核"。
    """
    path = tmp_path / "events.csv"
    lines = ["effective_date,ticker,action,review,source_url,name,note"]
    for effective_date, ticker, action in rows:
        lines.append(f"{effective_date},{ticker},{action},{review},{_SOURCE},,")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _initial_fifty(effective_date="2005-01-01"):
    return [(effective_date, f"{1000 + i}.TW", "add") for i in range(INDEX_SIZE)]


def test_members_asof_replays_events_in_order(tmp_path):
    path = _write_events(
        tmp_path,
        [
            *_initial_fifty(),
            ("2010-06-18", "1000.TW", "drop"),
            ("2010-06-18", "9999.TW", "add"),
        ],
    )

    before = members_asof(date(2010, 6, 17), path)
    after = members_asof(date(2010, 6, 18), path)

    assert "1000.TW" in before and "9999.TW" not in before
    assert "9999.TW" in after and "1000.TW" not in after
    assert len(before) == len(after) == INDEX_SIZE


def test_membership_is_empty_before_the_table_starts(tmp_path):
    path = _write_events(tmp_path, _initial_fifty("2005-01-01"))

    assert members_asof(date(2004, 12, 31), path) == set()


def test_same_day_add_and_drop_apply_together(tmp_path):
    """同日的納入與剔除必須一起套用，中途不得出現 49 或 51 檔。"""
    path = _write_events(
        tmp_path,
        [*_initial_fifty(), ("2010-06-18", "1000.TW", "drop"), ("2010-06-18", "9999.TW", "add")],
    )

    assert validate(path) == []


def test_validate_catches_a_missing_add(tmp_path):
    """剔除一檔卻沒有對應的納入 → 規模掉到 49，必須被抓到。"""
    path = _write_events(tmp_path, [*_initial_fifty(), ("2010-06-18", "1000.TW", "drop")])

    problems = validate(path)

    assert len(problems) == 1
    assert "49 檔" in problems[0]


def test_validate_catches_an_extra_add(tmp_path):
    path = _write_events(tmp_path, [*_initial_fifty(), ("2010-06-18", "9999.TW", "add")])

    problems = validate(path)

    assert len(problems) == 1
    assert "51 檔" in problems[0]


def test_dropping_a_non_member_is_an_error(tmp_path):
    """順序矛盾（先剔除後納入）必須炸，不能安靜地給出錯名單。"""
    path = _write_events(tmp_path, [*_initial_fifty(), ("2010-06-18", "8888.TW", "drop")])

    with pytest.raises(UniverseError, match="順序矛盾"):
        members_asof(date(2011, 1, 1), path)


def test_adding_an_existing_member_is_an_error(tmp_path):
    path = _write_events(tmp_path, [*_initial_fifty(), ("2010-06-18", "1000.TW", "add")])

    with pytest.raises(UniverseError, match="重複"):
        members_asof(date(2011, 1, 1), path)


def test_source_url_is_mandatory(tmp_path):
    """沒有出處的列不准進表——這份資料的價值完全建立在可查證上。"""
    path = tmp_path / "events.csv"
    path.write_text(
        "effective_date,ticker,action,review,source_url,note\n"
        "2010-06-18,1000.TW,add,定期審核,,\n",
        encoding="utf-8",
    )

    with pytest.raises(UniverseError, match="source_url"):
        members_asof(date(2011, 1, 1), path)


def test_all_tickers_ever_includes_delisted_names(tmp_path):
    """曾入榜但已被剔除的代號也要在清單裡——只抓現存成分股就是倖存者偏誤。"""
    path = _write_events(
        tmp_path,
        [*_initial_fifty(), ("2010-06-18", "1000.TW", "drop"), ("2010-06-18", "9999.TW", "add")],
    )

    ever = all_tickers_ever(path)

    assert "1000.TW" in ever  # 已被剔除，仍在清單內
    assert len(ever) == INDEX_SIZE + 1


def test_coverage_start_is_when_the_index_first_fills(tmp_path):
    path = _write_events(tmp_path, _initial_fifty("2005-01-01"))

    assert coverage_start(path) == date(2005, 1, 1)


def test_membership_frame_shape_and_values(tmp_path):
    path = _write_events(
        tmp_path,
        [*_initial_fifty(), ("2010-06-18", "1000.TW", "drop"), ("2010-06-18", "9999.TW", "add")],
    )
    dates = ["2010-06-17", "2010-06-18"]

    frame = membership_frame(dates, path)

    assert frame.shape == (2, INDEX_SIZE + 1)
    assert frame.loc["2010-06-17", "1000.TW"]
    assert not frame.loc["2010-06-18", "1000.TW"]
    assert frame.loc["2010-06-18", "9999.TW"]
    # 每一天都必須恰好 50 檔
    assert (frame.sum(axis=1) == INDEX_SIZE).all()


def test_missing_events_file_gives_an_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="人工策展"):
        members_asof(date(2020, 1, 1), tmp_path / "nope.csv")


# === 每季調整的排程：過期的池子必須藏不住 ===================================


@pytest.mark.parametrize(
    "year,month,expected",
    [
        (2024, 12, date(2024, 12, 20)),  # FTSE 公告：after close of Friday 20 Dec 2024
        (2026, 3, date(2026, 3, 20)),  # FTSE 公告：after close of Friday 20 Mar 2026
        (2025, 12, date(2025, 12, 19)),  # 中央社：2025-12-19 生效
        (2026, 6, date(2026, 6, 19)),
    ],
)
def test_third_friday_matches_published_review_dates(year, month, expected):
    """對照三份實際公告驗證排程算式，不是憑印象寫的。"""
    assert third_friday(year, month) == expected


def test_last_review_effective_walks_back_across_the_year_boundary():
    # 2026-01 還沒到 3 月審核，最近一次是 2025-12
    assert last_review_effective(date(2026, 1, 15)) == date(2025, 12, 19)
    # 生效日當天算已生效
    assert last_review_effective(date(2026, 3, 20)) == date(2026, 3, 20)
    # 生效日前一天則回到上一季
    assert last_review_effective(date(2026, 3, 19)) == date(2025, 12, 19)


def test_pool_staleness_counts_missed_reviews():
    """成分股每季換，所以『多久沒核對』本身就是要監看的量。"""
    missed, latest = pool_staleness(date(2025, 7, 1), today=date(2026, 8, 16))

    # 2025-09、2025-12、2026-03、2026-06 共四次
    assert missed == 4
    assert latest == date(2026, 6, 19)


def test_pool_staleness_is_zero_right_after_a_review():
    missed, _ = pool_staleness(date(2026, 6, 19), today=date(2026, 6, 20))

    assert missed == 0


def test_validate_flags_a_periodic_review_on_the_wrong_date(tmp_path):
    """定期審核的生效日必是當月第三個星期五——抄錯一天要被抓出來。

    這條不是假想：財經報導把 2026-06 的生效日寫成 6/18，但第三個星期五是 6/19。
    """
    path = _write_events(
        tmp_path,
        [*_initial_fifty(), ("2026-06-18", "1000.TW", "drop")],
        review="定期審核",
    )

    problems = validate(path)

    assert any("第三個星期五" in p and "2026-06-19" in p for p in problems)
