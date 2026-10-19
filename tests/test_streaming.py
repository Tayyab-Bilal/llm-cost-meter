import asyncio
import contextvars

import pytest

from llm_cost_meter import (
    InsufficientCredits,
    MeterMode,
    UnregisteredBillingLabel,
    ainvoke_metered,
    metered_chat_model,
    turn,
)
from llm_cost_meter.context import current_turn
from llm_cost_meter.fakes import fake_llm_factory

WORDS_130 = " ".join(["word"] * 130)


def _llm(meter, reply, service="chat", model="acme-chat", **kw):
    return metered_chat_model(service, model, llm_factory=fake_llm_factory(reply, **kw), meter=meter)


async def _run(meter, llm, prompt="hello", request_id="req-1"):
    with turn(tenant="t", payer="acme", entry_service="chat", request_id=request_id):
        return await ainvoke_metered(llm, [("user", prompt)])


async def test_stream_deducts_in_batches(make_meter):
    meter = make_meter()
    result = await _run(meter, _llm(meter, WORDS_130))
    assert not result.truncated and len(result.text.split()) == 130
    debits = [r for r in meter.ledger.rows() if r["kind"] == "debit"]
    # 1 input estimate + output batches at tokens 64 and 128; the last 2 tokens wait for the true-up
    assert [r["units"] for r in debits] == ["1", "64", "64"]
    assert meter.balances.get("acme") == 1_000_000 - (3 + 130 * 15)


async def test_stream_cancelled_when_credits_run_out_returns_partial(make_meter):
    meter = make_meter(balance=3 + 960 + 500)  # input + first 64-token batch (960) + not enough for the second
    result = await _run(meter, _llm(meter, WORDS_130))
    assert result.truncated is True
    assert 0 < len(result.text.split()) < 130
    # 127 words were delivered: 64 paid in the batch, the other 63 settled on cancel (overdraft < one batch)
    assert meter.balances.get("acme") == 500 - 63 * 15
    assert [r["units"] for r in meter.ledger.rows() if r["kind"] == "trueup"] == ["63"]


async def test_trueup_signed_positive_and_negative(make_meter):
    meter = make_meter()
    await _run(meter, _llm(meter, WORDS_130), request_id="up")
    [up] = [r for r in meter.ledger.rows("up") if r["kind"] == "trueup"]
    assert up["micro_credits"] == (3 + 130 * 15) - (3 + 2 * 960)

    # 1000 estimated input tokens, but the provider says all of them were cached: we over-deducted
    before = meter.balances.get("acme")
    await _run(meter, _llm(meter, "a b c", cached_tokens=1000), prompt="x" * 4000, request_id="down")
    [down] = [r for r in meter.ledger.rows("down") if r["kind"] == "trueup"]
    assert down["micro_credits"] == 300 + 45 - 3000 == -2655
    assert meter.balances.get("acme") == before - 345


async def test_preflight_refuses_when_empty(make_meter):
    meter = make_meter(balance=0)
    with pytest.raises(InsufficientCredits):
        await _run(meter, _llm(meter, "never streamed"), prompt="x" * 400)
    assert meter.ledger.rows() == []


async def test_subagent_cost_attributed_to_entry_point(make_meter):
    meter = make_meter()
    planner = _llm(meter, "step one then step two", service="planner")
    chat = _llm(meter, "final answer here")
    with turn(tenant="t", payer="acme", entry_service="chat", request_id="req-7"):
        await ainvoke_metered(planner, [("user", "plan it")])  # sub-agent: no code knows about the turn
        await ainvoke_metered(chat, [("user", "answer it")])
    rows = meter.ledger.rows()
    assert len(rows) > 2
    assert {r["request_id"] for r in rows} == {"req-7"}
    assert {r["service"] for r in rows} == {"chat"}  # the planner's cost followed the entry point
    assert {r["origin_service"] for r in rows} == {"chat", "planner"}  # ...but we still know who made the call


def test_unknown_service_label_rejected(make_meter):
    with pytest.raises(UnregisteredBillingLabel):
        factory = fake_llm_factory("x")
        metered_chat_model(  # billing-guard: ok
            "shiny_new_feature", "acme-chat", llm_factory=factory, meter=make_meter()
        )


async def test_provider_error_before_any_token_credits_back_the_estimate(make_meter):
    meter = make_meter()
    with pytest.raises(RuntimeError):
        await _run(meter, _llm(meter, WORDS_130, fail_after=0), prompt="x" * 4000)
    assert meter.balances.get("acme") == 1_000_000  # the 3000 pre-flight estimate came back


async def test_provider_error_mid_stream_bills_what_was_streamed(make_meter):
    meter = make_meter()
    with pytest.raises(RuntimeError):
        await _run(meter, _llm(meter, WORDS_130, fail_after=70))
    assert meter.balances.get("acme") == 1_000_000 - (3 + 70 * 15)  # 64 in a batch + 6 settled


async def test_shadow_mode_never_truncates_a_stream(make_meter):
    meter = make_meter(MeterMode.SHADOW, balance=1)
    result = await _run(meter, _llm(meter, WORDS_130))
    assert not result.truncated and len(result.text.split()) == 130
    assert meter.balances.get("acme") == 1


async def test_off_and_shadow_never_break_the_call_on_metering_errors(make_meter):
    for mode in (MeterMode.OFF, MeterMode.SHADOW):
        meter = make_meter(mode)
        llm = _llm(meter, "a b c", model="unpriced-model")  # pricing raises KeyError inside the callback
        result = await ainvoke_metered(llm, [("user", "hi")])  # and there is no turn at all
        assert result.text == "a b c"


async def test_enforce_still_raises_on_metering_errors(make_meter):
    meter = make_meter()
    with pytest.raises(KeyError):
        await _run(meter, _llm(meter, "a b c", model="unpriced-model"))


async def test_attribution_survives_gather_and_create_task(make_meter):
    meter = make_meter()
    a, b = _llm(meter, "one two", service="planner"), _llm(meter, "three four")
    with turn(tenant="t", payer="acme", entry_service="chat", request_id="req-g"):
        await asyncio.gather(ainvoke_metered(a, [("user", "x")]),
                             asyncio.create_task(ainvoke_metered(b, [("user", "y")])))
    assert {r["request_id"] for r in meter.ledger.rows()} == {"req-g"}


async def test_executor_loses_the_turn_unless_context_is_copied():
    loop = asyncio.get_running_loop()
    with turn(tenant="t", payer="p", entry_service="chat"):
        with pytest.raises(RuntimeError):
            await loop.run_in_executor(None, current_turn)  # bare executor: ContextVar is gone
        copied = contextvars.copy_context()
        assert (await loop.run_in_executor(None, copied.run, current_turn)).payer == "p"


async def test_missing_usage_is_warned_and_billed_from_the_stream(make_meter, caplog):
    meter = make_meter()
    await _run(meter, _llm(meter, WORDS_130, emit_usage=False))
    assert "no usage" in caplog.text
    assert meter.balances.get("acme") == 1_000_000 - (3 + 130 * 15)


async def test_list_of_blocks_content_is_counted(make_meter):
    meter = make_meter()
    blocks = [{"type": "text", "text": "x" * 4000}]
    with turn(tenant="t", payer="acme", entry_service="chat"):
        await ainvoke_metered(_llm(meter, "a"), [("user", blocks)])
    assert meter.ledger.rows()[0]["units"] == "1000"


def test_factory_keeps_existing_callbacks(make_meter):
    from langchain_core.callbacks import StdOutCallbackHandler
    mine = StdOutCallbackHandler()
    factory = fake_llm_factory("x")
    llm = metered_chat_model("chat", "acme-chat", llm_factory=lambda m: _with_cb(factory(m), mine), meter=make_meter())
    assert mine in llm.callbacks and len(llm.callbacks) == 2


def _with_cb(llm, cb):
    llm.callbacks = [cb]
    return llm
