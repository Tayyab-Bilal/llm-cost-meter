import threading
from concurrent.futures import ThreadPoolExecutor

import fakeredis
import pytest

from llm_cost_meter import Ledger, Meter, MeterKind, MeterMode, UnregisteredBillingLabel, turn
from llm_cost_meter.balance import InMemoryBalanceStore, RedisBalanceStore


def _row(ledger, key, micro=5):
    return ledger.append(key, payer="p", tenant="t", request_id="r", service="chat",
                         meter=MeterKind.TOKEN, units=1, micro=micro, kind="debit")


def test_ledger_idempotent_on_key():
    ledger = Ledger()
    assert _row(ledger, "k1") is True
    assert _row(ledger, "k1", micro=999) is False
    assert len(ledger.rows()) == 1 and ledger.balance_delta("p") == -5


def test_retried_ingestion_bills_once(make_meter):
    meter = make_meter()
    with turn(tenant="t", payer="acme", entry_service="chat"):
        first = meter.charge_units(MeterKind.PAGE, 3, work_key="doc-42", service="doc_parse", model="doc-parser")
        retry = meter.charge_units(MeterKind.PAGE, 3, work_key="doc-42", service="doc_parse", model="doc-parser")
    assert first.new and not retry.new
    assert len(meter.ledger.rows()) == 1
    assert meter.balances.get("acme") == 1_000_000 - 30_000


def test_charge_units_never_raises(make_meter):
    meter = make_meter()
    with turn(tenant="t", payer="acme", entry_service="chat"):
        result = meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="unpriced")
    assert not result.ok and result.error


def test_redis_lua_refuses_missing_key():
    r = fakeredis.FakeRedis()
    store = RedisBalanceStore(r)
    assert store.decrement_if_sufficient("ghost", 10) == (False, 0)
    assert not r.exists("credits:{ghost}")  # a missing balance is never invented


def test_redis_lua_refuses_overdraw_and_keeps_balance():
    store = RedisBalanceStore(fakeredis.FakeRedis())
    store.set("p", 100)
    assert store.decrement_if_sufficient("p", 101) == (False, 100)
    assert store.decrement_if_sufficient("p", 100) == (True, 0)
    assert store.get("p") == 0


def test_redis_lua_no_overdraw_under_concurrency(prices):
    store = RedisBalanceStore(fakeredis.FakeRedis())
    meter = Meter(Ledger(), store, prices, MeterMode.ENFORCE)
    meter.topup("acme", 20_000, key="seed")  # enough for exactly 20 of the 50 attempts at 1000 each
    seen_low = []

    def attempt(i: int) -> bool:
        with turn(tenant="t", payer="acme", entry_service="chat", request_id=f"r{i}"):
            ok = meter.debit(f"k{i}", "chat", 1, 1000)
        seen_low.append(store.get("acme"))
        return ok

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(attempt, range(50)))
    assert sum(results) == 20
    assert min(seen_low) >= 0 and store.get("acme") == 0
    assert meter.ledger.balance_delta("acme") == store.get("acme") == 0  # ledger total == balance


def test_soak_ledger_matches_balance(make_meter):
    meter = make_meter(balance=0)
    for p in range(8):
        meter.topup(f"payer{p}", 10_000_000, key=f"seed{p}")

    def run(p: int) -> None:
        with turn(tenant="t", payer=f"payer{p}", entry_service="chat", request_id=f"r{p}"):
            for i in range(60):
                work = f"p{p}-w{i % 55}"  # the last five repeat an earlier key: retries must not double bill
                meter.charge_rerank(i % 7 + 1, work_key=work, service="rerank",
                                   model="reranker")

    threads = [threading.Thread(target=run, args=(p,)) for p in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    gaps = [abs(meter.ledger.balance_delta(f"payer{p}") - meter.balances.get(f"payer{p}"))
            for p in range(8)]
    assert max(gaps) == 0
    assert len(meter.ledger.rows()) == 8 * 55


# ---- review fixes: missing balances, drift, retries, shadow, atomicity -------------------------------------

def test_adjust_refuses_missing_balance_in_both_stores():
    r = fakeredis.FakeRedis()
    for store in (RedisBalanceStore(r), InMemoryBalanceStore()):
        assert store.adjust("ghost", -500) is False  # a negative true-up must not mint a balance
        assert store.get("ghost") is None
    assert not r.exists("credits:{ghost}")


def test_lua_decrement_rejects_non_positive_amounts():
    store = RedisBalanceStore(fakeredis.FakeRedis())
    store.set("p", 100)
    assert store.decrement_if_sufficient("p", -50) == (False, 100)  # would otherwise add credit
    assert store.get("p") == 100


def test_missing_balance_keeps_ledger_row_and_rebuild_restores_it(make_meter):
    meter = make_meter()
    meter.balances = InMemoryBalanceStore()  # cache lost
    with turn(tenant="t", payer="acme", entry_service="chat"):
        r = meter.charge_rerank(5, work_key="w", service="rerank", model="reranker")
    assert r.ok and len(meter.ledger.rows()) == 1 and meter.balances.get("acme") is None
    assert meter.rebuild_balance("acme") == 1_000_000 - 10_000


def test_topup_goes_through_ledger_and_never_wipes_a_debt(make_meter):
    meter = make_meter(balance=1_000)
    with turn(tenant="t", payer="acme", entry_service="chat"):
        meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")  # -10000
    assert meter.balances.get("acme") == -9_000
    meter.topup("acme", 50_000, key="t2")
    meter.topup("acme", 50_000, key="t2")  # retried webhook: once
    assert meter.balances.get("acme") == 41_000 == meter.ledger.balance_delta("acme")


def test_balance_drift_is_detected(make_meter):
    from llm_cost_meter.reconcile import balance_drift
    meter = make_meter()
    assert balance_drift(meter.ledger, meter.balances, ["acme"]) == {}
    meter.balances.set("acme", 5)  # someone wrote the cache behind the ledger's back
    assert balance_drift(meter.ledger, meter.balances, ["acme"]) == {"acme": 5 - 1_000_000}


def test_debit_compensates_when_ledger_insert_fails(make_meter, monkeypatch):
    meter = make_meter()

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(meter.ledger, "append", boom)
    with turn(tenant="t", payer="acme", entry_service="chat"), pytest.raises(RuntimeError):
        meter.debit("k", "chat", 1, 700)
    assert meter.balances.get("acme") == 1_000_000  # the user was not charged for an unrecorded debit


def test_debit_duplicate_key_is_compensated(make_meter):
    meter = make_meter()
    with turn(tenant="t", payer="acme", entry_service="chat"):
        assert meter.debit("k", "chat", 1, 700) and meter.debit("k", "chat", 1, 700)
    assert meter.balances.get("acme") == 1_000_000 - 700 and len(meter.ledger.rows()) == 1


def test_failed_balance_move_is_finished_by_a_retry(make_meter, monkeypatch):
    meter = make_meter()
    real = meter.balances.adjust
    monkeypatch.setattr(meter.balances, "adjust", lambda *a: (_ for _ in ()).throw(RuntimeError("redis down")))
    with turn(tenant="t", payer="acme", entry_service="chat"):
        first = meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
        assert not first.ok and meter.balances.get("acme") == 1_000_000
        monkeypatch.setattr(meter.balances, "adjust", real)
        retry = meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
        meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
    assert retry.ok and meter.balances.get("acme") == 1_000_000 - 10_000  # moved once, not zero, not twice


def test_shadow_row_does_not_block_later_enforce_billing(make_meter):
    meter = make_meter(MeterMode.SHADOW)
    with turn(tenant="t", payer="acme", entry_service="chat"):
        meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
        meter.mode = MeterMode.ENFORCE
        meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
    assert meter.balances.get("acme") == 1_000_000 - 10_000
    assert sorted(r["shadow"] for r in meter.ledger.rows() if r["kind"] == "debit") == [0, 1]


def test_float_units_are_rejected(prices):
    with pytest.raises(TypeError):
        prices.cost_units("doc-parser", MeterKind.PAGE, 0.1 + 0.2)


def test_nested_turn_for_another_payer_raises():
    with turn(tenant="t", payer="a", entry_service="chat"), pytest.raises(ValueError):
        with turn(tenant="t", payer="b", entry_service="chat"):
            pass


class _NaiveStore(InMemoryBalanceStore):
    """Check-then-write with a barrier between the two steps: the interleaving a non-atomic store allows."""

    def __init__(self, barrier):
        super().__init__()
        self.barrier = barrier

    def decrement_if_sufficient(self, payer, micro):
        cur = self.get(payer)
        self.barrier.wait()
        if cur is None or cur < micro:
            return False, cur or 0
        self.set(payer, cur - micro)
        return True, cur - micro


def _race(store) -> list[bool]:
    store.set("p", 1000)
    with ThreadPoolExecutor(2) as pool:
        return list(pool.map(lambda _: store.decrement_if_sufficient("p", 1000)[0], range(2)))


def test_overdraw_detector_catches_a_non_atomic_store_and_lua_passes():
    naive = _NaiveStore(threading.Barrier(2))
    assert _race(naive) == [True, True] and naive.get("p") == 0  # 2000 spent from 1000, and the cache says 0
    assert sum(_race(_NaiveStore(threading.Barrier(2)))) == 2  # both pass the check: a double spend
    for _ in range(20):
        assert sum(_race(RedisBalanceStore(fakeredis.FakeRedis()))) == 1  # Lua: exactly one wins


# ---- second review ------------------------------------------------------------------------------------------

def _down(meter, monkeypatch):
    """Make every balance move fail, like Redis being down."""
    monkeypatch.setattr(meter.balances, "adjust", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))


def test_rebuild_then_retry_charges_once(make_meter, monkeypatch):
    meter = make_meter()
    real = meter.balances.adjust
    _down(meter, monkeypatch)
    with turn(tenant="t", payer="acme", entry_service="chat"):
        meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
        monkeypatch.setattr(meter.balances, "adjust", real)
        assert meter.rebuild_balance("acme") == 1_000_000 - 10_000
        meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")  # the retry
    assert meter.balances.get("acme") == 1_000_000 - 10_000 == meter.ledger.balance_delta("acme")


def test_interleaved_first_topups_match_ledger(prices):
    store = InMemoryBalanceStore()
    meter = Meter(Ledger(), store, prices, MeterMode.ENFORCE)
    real = store.adjust
    calls = []

    def adjust(payer, delta, key=None):
        if not calls:  # webhook A has claimed its row; webhook B runs entirely inside A's adjust
            calls.append(1)
            meter.topup("p", 200, key="B")
        return real(payer, delta, key)

    store.adjust = adjust
    meter.topup("p", 100, key="A")
    assert store.get("p") == meter.ledger.balance_delta("p") == 300  # not 400, not 100


def test_idempotency_keys_are_per_payer(make_meter):
    meter = make_meter()
    meter.topup("other", 1_000_000, key="seed2", tenant="t2")
    for payer in ("acme", "other"):
        with turn(tenant=payer, payer=payer, entry_service="chat"):
            meter.charge_units(MeterKind.PAGE, 1, work_key="job-1", service="doc_parse", model="doc-parser")
    assert meter.balances.get("acme") == meter.balances.get("other") == 1_000_000 - 10_000


def test_one_payers_pending_row_cannot_be_claimed_by_another(make_meter):
    meter = make_meter()
    meter.ledger.append("job-1", payer="acme", tenant="t", request_id="r", service="chat",
                        meter=MeterKind.TOKEN, units=1, micro=5, kind="debit")
    assert meter.ledger.claim("other", "job-1") is False and meter.ledger.claim("acme", "job-1") is True


def test_shadow_rows_never_move_rebuild_or_drift(make_meter):
    from llm_cost_meter.reconcile import balance_drift
    meter = make_meter()
    meter.mode = MeterMode.SHADOW
    with turn(tenant="t", payer="acme", entry_service="chat"):
        meter.charge_units(MeterKind.PAGE, 5, work_key="w", service="doc_parse", model="doc-parser")
    assert balance_drift(meter.ledger, meter.balances, ["acme"]) == {}
    assert meter.rebuild_balance("acme") == 1_000_000 == meter.balances.get("acme")


def test_zero_value_rows_are_born_applied(make_meter):
    meter = make_meter()
    with turn(tenant="t", payer="acme", entry_service="health_probe"):
        meter.debit("k", "health_probe", 5, 0)
    assert meter.ledger.rows()[0]["applied"] == 1


@pytest.mark.parametrize("store_factory", [InMemoryBalanceStore, lambda: RedisBalanceStore(fakeredis.FakeRedis())])
def test_decrement_rejects_non_positive_in_every_store(store_factory):
    store = store_factory()
    store.set("p", 100)
    assert store.decrement_if_sufficient("p", 0)[0] is False
    assert store.decrement_if_sufficient("p", -5)[0] is False
    assert store.get("p") == 100


def test_debit_rejects_negative(make_meter):
    with turn(tenant="t", payer="acme", entry_service="chat"), pytest.raises(ValueError):
        make_meter().debit("k", "chat", 1, -5)


def test_ambiguous_redis_failure_applies_once(prices):
    store = RedisBalanceStore(fakeredis.FakeRedis())
    meter = Meter(Ledger(), store, prices, MeterMode.ENFORCE)
    meter.topup("acme", 1_000_000, key="seed")
    real, fail = store._adjust, [True]

    def commit_then_raise(*, keys, args):
        out = real(keys=keys, args=args)
        if fail.pop() if fail else False:
            raise ConnectionError("timeout after commit")
        return out

    store._adjust = commit_then_raise
    with turn(tenant="t", payer="acme", entry_service="chat"):
        first = meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser")
        assert not first.ok
        assert meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="doc_parse", model="doc-parser").ok
    assert store.get("acme") == 1_000_000 - 10_000


def test_unknown_label_in_charge_units_raises_when_enforcing_and_logs_otherwise(make_meter, caplog):
    def charge(meter):
        return meter.charge_units(MeterKind.PAGE, 1, work_key="w", service="nope", model="m")  # billing-guard: ok

    with turn(tenant="t", payer="acme", entry_service="chat"):
        with pytest.raises(UnregisteredBillingLabel):
            charge(make_meter())
        assert not charge(make_meter(MeterMode.OFF)).ok and not charge(make_meter(MeterMode.SHADOW)).ok
    assert "nope" in caplog.text
