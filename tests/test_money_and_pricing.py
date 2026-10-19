from decimal import Decimal as D

from llm_cost_meter import MICRO, MeterKind
from llm_cost_meter.units import usd_to_units


def test_money_is_integer_micro_credits(prices):
    assert MICRO == 1_000_000
    assert usd_to_units("0.0000005") == 1  # half up, not banker's rounding
    assert usd_to_units(D("0.0000015")) == 2
    # 10 tokens at $0.15/Mtok is 1.5 micro-credits; it must round to an int, never leak a float
    prices.rates["tiny"] = type(prices.rates["acme-chat"])(input_per_mtok=D("0.15"))
    cost = prices.cost_tokens("tiny", 10, 0)
    assert cost == 2 and type(cost) is int
    assert type(prices.cost_units("doc-parser", MeterKind.PAGE, 3)) is int


def test_cached_tokens_use_cached_rate(prices):
    # 400 fresh at $3 + 600 cached at $0.30 = 1200 + 180
    assert prices.cost_tokens("acme-chat", 1000, 0, cached_tokens=600) == 1380
    assert prices.cost_tokens("acme-chat", 1000, 0) == 3000


def test_missing_cached_rate_falls_back_with_warning(prices, caplog):
    assert prices.cost_tokens("no-cache-model", 1000, 0, cached_tokens=600) == 3000
    assert "no cached rate" in caplog.text
