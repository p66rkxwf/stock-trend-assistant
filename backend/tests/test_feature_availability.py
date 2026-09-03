"""特徵可用時點：每一欄都要標註，且沒有任何一欄可以用到未來。

散文式的「本特徵只用當日資料」不會在有人加了前視特徵時亮紅燈；這裡把它變成
算術上可檢查的不變式。與 test_no_lookahead.py 互補：那邊竄改未來資料驗證
**實作**沒有前視，這邊驗證**宣告**完整且自洽。
"""

import pytest

from stockta.config import LABEL_HORIZON_DAYS
from stockta.features.availability import LABEL_AVAILABILITY, gates, latest_offset
from stockta.features.market import ALL_CONTEXT_COLUMNS, CONTEXT_AVAILABILITY
from stockta.features.pipeline import (
    FEATURE_AVAILABILITY,
    FEATURE_COLUMNS,
    RELATIVE_FEATURE_COLUMNS,
    STOCK_FEATURE_COLUMNS,
)

pytestmark = pytest.mark.leakage


def test_every_declared_column_has_an_availability():
    """新增特徵而忘了標註可用時點 → 紅燈。"""
    declared = {*STOCK_FEATURE_COLUMNS, *ALL_CONTEXT_COLUMNS, *RELATIVE_FEATURE_COLUMNS}

    assert set(FEATURE_AVAILABILITY) == declared


def test_no_feature_uses_future_information():
    """核心不變式：任何特徵的位移都不得為正。

    標了正數就是自己承認用了未來資訊。這條是宣告層面的把關，
    實作層面由 test_no_lookahead.py 竄改未來資料驗證。
    """
    offending = {
        col: value for col, value in FEATURE_AVAILABILITY.items() if value[0] > 0
    }

    assert offending == {}, f"這些特徵宣告使用了未來資訊：{offending}"


def test_label_is_strictly_later_than_every_feature():
    """特徵與標籤的時間隔離——寫成一條算術上成立的敘述，而不是宣稱。"""
    label_offset, _ = LABEL_AVAILABILITY

    assert label_offset == LABEL_HORIZON_DAYS
    assert label_offset > latest_offset(FEATURE_AVAILABILITY)


def test_production_feature_columns_are_all_annotated():
    """正式管線實際使用的 25 欄（FEATURE_COLUMNS）必須全部有標註。"""
    missing = [col for col in FEATURE_COLUMNS if col not in FEATURE_AVAILABILITY]

    assert missing == []


def test_breadth_gate_is_the_binding_constraint():
    """市場寬度是當日最晚到位的特徵——決策時點由它決定，不是由個股 K 線。

    釘住這件事是因為它會被忘記：所有特徵的位移都是 0，看起來一樣早，
    但寬度要等全池，實際到位時間晚得多。
    """
    stock_gate = FEATURE_AVAILABILITY["ret_1d"][1]
    breadth_gate = FEATURE_AVAILABILITY["breadth_up"][1]

    assert breadth_gate != stock_gate
    assert "全池" in breadth_gate
    # 寬度的門檻條件必須出現在 context 的宣告裡（兩處不得各說各話）
    assert breadth_gate in gates(CONTEXT_AVAILABILITY)


def test_regime_columns_are_annotated_even_though_unused():
    """regime 欄位目前未採用（MARKET_INCLUDE_REGIME=False），但仍須標註。

    否則哪天打開旗標，就會有四個沒人檢查過可用時點的特徵直接進正式管線。
    """
    for col in ("mkt_ma60", "mkt_ma200", "mkt_drawdown", "mkt_vol_pct"):
        assert col in FEATURE_AVAILABILITY
        assert FEATURE_AVAILABILITY[col][0] == 0
