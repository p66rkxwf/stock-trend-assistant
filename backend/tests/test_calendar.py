from datetime import datetime

from stockta.data.calendar import last_completed_trading_day, warmup_lookback_days
from stockta.config import TAIPEI_TZ


def test_before_close_uses_previous_day():
    now = datetime(2024, 6, 12, 10, 0, tzinfo=TAIPEI_TZ)  # 週三，收盤前
    assert last_completed_trading_day(now).isoformat() == "2024-06-11"


def test_after_close_uses_same_day():
    now = datetime(2024, 6, 12, 14, 0, tzinfo=TAIPEI_TZ)  # 週三，收盤後
    assert last_completed_trading_day(now).isoformat() == "2024-06-12"


def test_monday_before_close_rolls_back_to_friday():
    now = datetime(2024, 6, 10, 9, 0, tzinfo=TAIPEI_TZ)  # 週一，收盤前
    assert last_completed_trading_day(now).isoformat() == "2024-06-07"


def test_warmup_lookback_days_sums_components():
    assert warmup_lookback_days(60, 60, buffer_days=10) == 130
