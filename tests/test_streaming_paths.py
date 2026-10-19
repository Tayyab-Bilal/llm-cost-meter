import fakeredis
import pytest

from llm_cost_meter import (
    CreditsExhausted,
    InsufficientCredits,
    MeterKind,
    ainvoke_metered,
    metered_chat_model,
    turn,
)
from llm_cost_meter.balance import InMemoryBalanceStore, RedisBalanceStore
from llm_cost_meter.fakes import fake_llm_factory

WORDS_130 = " ".join(["word"] * 130)


def _llm(meter, *replies, **kw):
    return metered_chat_model("chat", "acme-chat", llm_factory=fake_llm_factory(*replies, **kw), meter=meter)


# ---- every call streams internally ------------------------------------------------------------------------

async def test_plain_ainvoke_streams_internally_and_is_metered_in_batches(make_meter):
    meter = make_meter()
    with turn(tenant="t", payer="acme", entry_service="chat"):
        msg = await _llm(meter, WORDS_130).ainvoke([("user", "hello")])  # no ainvoke_metered
    assert len(msg.content.split()) == 130
    assert [r["units"] for r in meter.ledger.rows() if r["kind"] == "debit"] == ["1", "64", "64"]  # streamed


async def test_plain_ainvoke_is_cut_off_when_credit_runs_out(make_meter):
    meter = make_meter(balance=3 + 960 + 500)
    with turn(tenant="t", payer="acme", entry_service="chat"), pytest.raises(CreditsExhausted) as err:
        await _llm(meter, WORDS_130).ainvoke([("user", "hello")])
    assert 0 < len(err.value.partial_text.split()) < 130  # the text delivered so far travels with the signal
    assert meter.balances.get("acme") == 500 - 63 * 15  # delivered tokens were settled, as on the streaming path


async def test_plain_ainvoke_refused_before_any_token_when_empty(make_meter):
    meter = make_meter(balance=0)
    llm = _llm(meter, WORDS_130)
    with turn(tenant="t", payer="acme", entry_service="chat"), pytest.raises(InsufficientCredits):
        await llm.ainvoke([("user", "hello")])
    assert meter.ledger.rows() == []


# ---- fallback chains ---------------------------------------------------------------------------------------

async def test_refusal_is_final_and_never_served_by_a_fallback_model(make_meter):
    """Regression: a refusal once fell through to the fallback chain and was served anyway."""
    meter = make_meter(balance=2_000)  # too little for the primary's input, enough for the cheaper fallback's
    primary = _llm(meter, "never sent")
    fallback = metered_chat_model("chat", "cheap-model", llm_factory=fake_llm_factory("served by the fallback"),
                               meter=meter)
    chain = primary.with_fallbacks([fallback])
    with turn(tenant="t", payer="acme", entry_service="chat"), pytest.raises(InsufficientCredits):
        await chain.ainvoke([("user", "x" * 4000)])
    assert meter.ledger.rows() == []  # the fallback was never even started
    assert await _fallback_untouched(fallback)


async def _fallback_untouched(fallback) -> bool:
    return next(fallback.messages, None) is not None  # its scripted reply is still unread


async def test_cutoff_mid_stream_is_not_continued_by_a_fallback_model(make_meter):
    meter = make_meter(balance=3 + 960 + 500)
    fallback = _llm(meter, "fallback answer")
    chain = _llm(meter, WORDS_130).with_fallbacks([fallback])
    with turn(tenant="t", payer="acme", entry_service="chat"), pytest.raises(InsufficientCredits):
        await chain.ainvoke([("user", "hello")])  # same payer, no credit left: the fallback is refused too
    assert len([r for r in meter.ledger.rows() if r["kind"] == "debit"]) == 2  # input + one output batch only
    assert await _fallback_untouched(fallback)


async def test_provider_failure_is_billed_once_and_the_fallback_is_billed_for_its_own_call(make_meter):
    meter = make_meter()
    chain = _llm(meter, "a b c d", fail_after=0).with_fallbacks([_llm(meter, "fallback answer here")])
    with turn(tenant="t", payer="acme", entry_service="chat", request_id="req"):
        msg = await chain.ainvoke([("user", "hi")])
    assert msg.content == "fallback answer here"
    spent = 1_000_000 - meter.balances.get("acme")
    assert spent == meter.prices.cost_tokens("acme-chat", 1, 3)  # failed primary: estimate credited back


@pytest.mark.parametrize("store", [InMemoryBalanceStore, lambda: RedisBalanceStore(fakeredis.FakeRedis())])
def test_refund_cannot_mint_credit_when_the_deduct_path_degraded(make_meter, store):
    """Regression: a refund once minted credit because the deduct path had degraded to the ledger only."""
    meter = make_meter(balances=store())
    meter.balances = store()  # the live balance is gone (cache lost, degraded path): no key for this payer
    with turn(tenant="t", payer="acme", entry_service="chat"):
        meter.trueup("k", "chat", -10, -5_000)  # a refund / negative true-up
    assert meter.balances.get("acme") is None  # no balance was invented for the refund
    assert meter.ledger.rows()[0]["micro_credits"] == -5_000  # the row is still recorded
    assert meter.rebuild_balance("acme") == 1_000_000 + 5_000  # the ledger, not the refund, restores the cache


# ---- identity -----------------------------------------------------------------------------------------------

async def test_singleton_llm_bills_each_turns_own_payer_never_the_first_caller(make_meter):
    """Regression: a module-level singleton once cached the first caller's identity (tenant A paid for B)."""
    meter = make_meter()
    meter.topup("other", 1_000_000, key="seed-other")
    shared = _llm(meter, "one two three", "four five six")  # built once, outside any turn, reused across turns
    for payer in ("acme", "other"):
        with turn(tenant=f"t-{payer}", payer=payer, entry_service="chat", request_id=f"req-{payer}"):
            await ainvoke_metered(shared, [("user", "hi")])
    by_payer = {p: [r for r in meter.ledger.rows(payer=p)] for p in ("acme", "other")}
    assert all(len(rows) > 0 for rows in by_payer.values())
    assert {r["tenant"] for r in by_payer["other"]} == {"t-other"}
    assert {r["request_id"] for r in by_payer["acme"]} == {"req-acme"}


# ---- thinking tokens --------------------------------------------------------------------------------------

async def test_reasoning_tokens_are_billed_once_as_output_tokens(make_meter):
    meter = make_meter()
    with turn(tenant="t", payer="acme", entry_service="chat"):
        result = await ainvoke_metered(_llm(meter, "a b c", reasoning_tokens=100), [("user", "hi")])
    assert result.usage["output_token_details"]["reasoning"] == 100
    assert 1_000_000 - meter.balances.get("acme") == meter.prices.cost_tokens("acme-chat", 1, 103)  # 3 + 100, once


def test_meter_kinds_used_by_tests_exist():
    assert MeterKind.TOKEN
