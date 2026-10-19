from decimal import Decimal as D

import pytest

from llm_cost_meter import MeterKind, PriceTable, Rates, turn

REALTIME = Rates(input_per_mtok=D(4), cached_input_per_mtok=D("0.4"), output_per_mtok=D(16),
                 audio_input_per_mtok=D(32), cached_audio_input_per_mtok=D("0.4"), audio_output_per_mtok=D(64))


@pytest.fixture
def prices():
    return PriceTable({"rt": REALTIME, "vlm": Rates(per_vision_unit=D("0.005")), "rr": Rates(per_rerank=D("0.002"))})


def test_realtime_tokens_are_priced_per_modality(prices):
    cost = prices.cost_realtime("rt", text_in=1000, audio_in=1000, text_out=1000, audio_out=1000)
    assert cost == 1000 * 4 + 1000 * 32 + 1000 * 16 + 1000 * 64


def test_realtime_is_not_billed_entirely_at_the_audio_rate(prices):
    """Regression: every realtime token was once billed at the audio rate, about 5x too much."""
    text_heavy = dict(text_in=9000, audio_in=1000, text_out=4000, audio_out=1000)
    all_audio = (9000 + 1000) * 32 + (4000 + 1000) * 64
    right = prices.cost_realtime("rt", **text_heavy)
    assert right == 9000 * 4 + 1000 * 32 + 4000 * 16 + 1000 * 64
    assert all_audio / right > 3  # the old behaviour overcharged by a large factor
    assert right < all_audio


def test_realtime_cached_tier_is_separate_per_modality(prices):
    # 1000 cached audio input tokens cost 0.4 each, not 32 (the "80x on cached input" case)
    assert prices.cost_realtime("rt", audio_in=1000, cached_audio_in=1000) == 1000 * D("0.4")
    assert prices.cost_realtime("rt", text_in=1000, cached_text_in=400) == 600 * 4 + 400 * D("0.4")


def test_vision_units_are_a_separate_meter(prices, make_meter):
    meter = make_meter()
    meter.prices = prices
    with turn(tenant="t", payer="acme", entry_service="chat"):
        r = meter.charge_units(MeterKind.VISION_UNIT, 12, work_key="img-1", service="doc_parse", model="vlm")
    assert r.micro_credits == 60_000 and meter.ledger.rows()[0]["meter"] == "vision_unit"


def test_meter_set_matches_the_five_documented_meters():
    assert {m.name for m in MeterKind} == {"TOKEN", "PAGE", "AUDIO_MINUTE", "EMBED_TOKEN", "VISION_UNIT"}


def test_rerank_is_priced_per_search_under_the_embed_meter(prices, make_meter):
    meter = make_meter()
    meter.prices = prices
    with turn(tenant="t", payer="acme", entry_service="chat"):
        first = meter.charge_rerank(3, work_key="s1", service="rerank", model="rr")
        retry = meter.charge_rerank(3, work_key="s1", service="rerank", model="rr")
    assert first.micro_credits == 6_000 and retry.new is False
    assert [r["meter"] for r in meter.ledger.rows()] == ["embed_token"]
