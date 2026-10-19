import contextvars

from llm_cost_meter import MeterKind, MeterMode, turn
from llm_cost_meter.context import activate, current_turn, from_task_headers, to_task_headers


def _charge(meter):
    return meter.charge_units(MeterKind.PAGE, 2, work_key="w1", service="doc_parse", model="doc-parser")


def test_shadow_mode_writes_ledger_but_not_balance(make_meter):
    meter = make_meter(MeterMode.SHADOW)
    with turn(tenant="t", payer="acme", entry_service="chat"):
        _charge(meter)
    [row] = meter.ledger.rows()
    assert row["shadow"] == 1 and row["micro_credits"] == 20_000
    assert meter.balances.get("acme") == 1_000_000


def test_off_mode_noop(make_meter):
    meter = make_meter(MeterMode.OFF)
    with turn(tenant="t", payer="acme", entry_service="chat"):
        assert _charge(meter).ok
        assert meter.debit("k", "chat", 10, 500) is True
    assert meter.ledger.rows() == [] and meter.balances.get("acme") == 1_000_000


def test_turn_survives_task_headers_roundtrip():
    with turn(tenant="t1", payer="p1", entry_service="chat", request_id="req-9") as ctx:
        headers = to_task_headers(ctx)
    assert all(isinstance(v, str) for v in headers.values())
    restored = from_task_headers(headers)
    assert restored == ctx

    def background_job():  # a worker process starts with an empty context
        with activate(restored):
            return current_turn()

    assert contextvars.Context().run(background_job) == ctx


def test_nested_turn_keeps_entry_point():
    with turn(tenant="t", payer="p", entry_service="chat", request_id="r1"):
        with turn(tenant="t", payer="p", entry_service="planner", request_id="r2") as inner:
            assert inner.entry_service == "chat" and inner.request_id == "r1"
