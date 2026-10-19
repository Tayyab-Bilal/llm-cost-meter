import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_cost_meter import MeterKind, PriceHistory, Rates, turn
from llm_cost_meter.api import create_credits_router


@pytest.fixture
def client(make_meter):
    meter = make_meter(balance=1_000_000)
    meter.topup("other", 5_000_000, key="seed-other")
    with turn(tenant="t", payer="acme", entry_service="chat", request_id="req-1"):
        meter.charge_units(MeterKind.PAGE, 4, work_key="w1", service="doc_parse", model="doc-parser")
    history = PriceHistory()
    from decimal import Decimal as D
    history.set_rates("doc-parser", Rates(per_page=D("0.01")), changed_by="ops", reason="initial")
    history.set_rates("doc-parser", Rates(per_page=D("0.02")), changed_by="ops", reason="vendor raised price")
    app = FastAPI()
    app.include_router(create_credits_router(meter, history))
    return TestClient(app), meter


def test_balance_shows_cache_ledger_and_drift(client):
    http, _ = client
    body = http.get("/credits/balance", headers={"x-payer": "acme"}).json()
    assert body == {"payer": "acme", "balance_micro": 960_000, "ledger_micro": 960_000, "drift_micro": 0}


def test_usage_is_scoped_to_the_callers_own_payer(client):
    http, _ = client
    mine = http.get("/credits/usage", headers={"x-payer": "acme"}).json()
    assert [r["meter"] for r in mine["rows"]] == ["page"] and mine["total_micro"] == 40_000
    other = http.get("/credits/usage", headers={"x-payer": "other"}).json()
    assert other["rows"] == []  # acme's usage is invisible to another payer


def test_usage_can_be_filtered_by_request(client):
    http, _ = client
    assert http.get("/credits/usage?request_id=nope", headers={"x-payer": "acme"}).json()["rows"] == []
    assert len(http.get("/credits/usage?request_id=req-1", headers={"x-payer": "acme"}).json()["rows"]) == 1


def test_identity_is_required(client):
    http, _ = client
    assert http.get("/credits/balance").status_code == 401


def test_reconciliation_names_the_cause_of_a_gap(client):
    http, _ = client
    export = [{"service": "doc_parse", "model": "doc-parser", "meter": "page", "cost_micro": 90_000}]
    found = http.post("/credits/reconciliation", json={"export": export}).json()["mismatches"]
    assert found == [{"service": "doc_parse", "ledger_micro": 40_000, "provider_micro": 90_000, "cause": "unknown"}]


def test_pricing_history_is_append_only_and_ordered(client):
    http, _ = client
    changes = http.get("/credits/pricing-history?model=doc-parser").json()["changes"]
    assert [c["rates"]["per_page"] for c in changes] == ["0.01", "0.02"]
    assert [c["seq"] for c in changes] == [1, 2] and changes[1]["reason"] == "vendor raised price"
