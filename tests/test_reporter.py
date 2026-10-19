import threading

import pytest

from llm_cost_meter import FakeBillingBackend, MeterKind, SpendReporter, turn


def _spend(meter, payer, micro, key):
    """Post a debit of exactly `micro` for `payer` through the ledger."""
    meter.ledger.append(key, payer=payer, tenant="t", request_id="r", service="chat", meter=MeterKind.TOKEN,
                        units=1, micro=micro, kind="debit")


@pytest.fixture
def setup(make_meter):
    meter = make_meter(balance=50_000_000)
    return meter, SpendReporter(meter.ledger), FakeBillingBackend()


def test_reports_whole_units_and_carries_the_remainder(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 2_500_000, "a")
    assert reporter.sweep(backend.report_spend).reported == {"acme": 2}
    assert reporter.remainder("acme") == 500_000
    _spend(meter, "acme", 700_000, "b")  # 0.5 carried + 0.7 = 1.2 credits
    assert reporter.sweep(backend.report_spend).reported == {"acme": 1}
    assert reporter.remainder("acme") == 200_000
    assert backend.reported_units("acme") == 3  # 3.2 credits spent, 3 reported, 0.2 carried


def test_sub_unit_spend_is_not_sent_but_not_lost(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 400_000, "a")
    assert reporter.sweep(backend.report_spend).reported == {}
    assert backend.send_calls == 0 and reporter.remainder("acme") == 400_000
    _spend(meter, "acme", 700_000, "b")
    assert reporter.sweep(backend.report_spend).reported == {"acme": 1}


def test_trueup_credit_back_lowers_the_reported_spend(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 3_000_000, "a")
    meter.ledger.append("t1", payer="acme", tenant="t", request_id="r", service="chat", meter=MeterKind.TOKEN,
                        units=1, micro=-1_000_000, kind="trueup")
    reporter.sweep(backend.report_spend)
    assert backend.reported_units("acme") == 2


def test_topups_are_not_reported_as_spend(setup):
    meter, reporter, backend = setup  # the fixture already topped up 50 credits
    assert reporter.sweep(backend.report_spend).reported == {}
    assert backend.send_calls == 0


def test_positions_only_move_forward_and_a_clean_sweep_is_a_noop(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 1_000_000, "a")
    reporter.sweep(backend.report_spend)
    first = reporter.position("acme")
    assert first > 0
    reporter.sweep(backend.report_spend)  # nothing new: no send, no movement
    assert backend.send_calls == 1 and reporter.position("acme") == first
    with meter.ledger.transaction() as db:  # replaying an older position must not move it back
        reporter._advance(db, "acme", first - 1, 0)  # noqa: SLF001
    assert reporter.position("acme") == first


def test_attempt_is_recorded_before_the_send(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 1_000_000, "a")
    seen = {}

    def send(report):
        with meter.ledger.transaction() as db:
            seen["row"] = db.execute("SELECT status, units FROM report_attempts WHERE op_id=?",
                                     (report.op_id,)).fetchone()
        backend.report_spend(report)

    reporter.sweep(send)
    assert tuple(seen["row"]) == ("pending", 1)


def test_failing_sender_keeps_the_position_and_retry_reports_once(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 2_000_000, "a")
    backend.available = False
    result = reporter.sweep(backend.report_spend)
    assert result.failed == ["acme"] and reporter.position("acme") == 0 and backend.reported_units("acme") == 0
    _spend(meter, "acme", 1_000_000, "b")  # new spend arrives while the old attempt is still open
    backend.available = True
    assert reporter.sweep(backend.report_spend).reported == {"acme": 2}  # the open attempt is resent unchanged
    assert reporter.sweep(backend.report_spend).reported == {"acme": 1}  # then the newer spend goes out
    assert backend.reported_units("acme") == 3 and len(backend.ops) == 2


def test_ambiguous_failure_is_deduplicated_by_the_operation_id(setup):
    """The backend applied the report but the reply was lost: the retry reuses the op id, so no double count."""
    meter, reporter, backend = setup
    _spend(meter, "acme", 2_000_000, "a")

    def lost_reply(report):
        backend.report_spend(report)
        raise TimeoutError("reply lost")

    assert reporter.sweep(lost_reply).failed == ["acme"]
    reporter.sweep(backend.report_spend)  # resend, same op id
    assert backend.reported_units("acme") == 2 and backend.balances["acme"] == -2


def test_two_concurrent_sweeps_never_report_the_same_spend_twice(setup):
    meter, reporter, backend = setup
    _spend(meter, "acme", 4_000_000, "a")
    inside, release = threading.Event(), threading.Event()

    def slow_send(report):
        inside.set()
        assert release.wait(5)
        backend.report_spend(report)

    results = {}
    first = threading.Thread(target=lambda: results.update(first=reporter.sweep(slow_send)))
    first.start()
    assert inside.wait(5)  # sweep one holds the payer's claim and is mid-send
    results["second"] = reporter.sweep(backend.report_spend)  # overlapping sweep
    release.set()
    first.join()
    assert results["second"].skipped == ["acme"] and results["second"].reported == {}
    assert results["first"].reported == {"acme": 4}
    assert backend.reported_units("acme") == 4 and backend.send_calls == 1
    assert reporter.sweep(backend.report_spend).reported == {}  # and nothing is left to report


def test_each_payer_is_reported_separately(setup):
    meter, reporter, backend = setup
    meter.topup("other", 10_000_000, key="seed-other")
    with turn(tenant="t", payer="other", entry_service="chat"):
        meter.charge_units(MeterKind.PAGE, 100, work_key="w", service="doc_parse", model="doc-parser")  # 1 credit
    _spend(meter, "acme", 2_000_000, "a")
    assert reporter.sweep(backend.report_spend).reported == {"acme": 2, "other": 1}
