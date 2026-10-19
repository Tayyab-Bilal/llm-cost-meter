import json
from dataclasses import replace
from decimal import Decimal as D
from pathlib import Path

from langchain_core.embeddings import Embeddings

from llm_cost_meter import PriceTable, Rates
from llm_cost_meter.guards.ast_guard import check_file
from llm_cost_meter.guards.rate_drift import check_rate_drift, snapshot

FIX = Path(__file__).parent / "fixtures"
PINNED = json.loads((FIX / "pinned_rates.json").read_text())
NO_CACHE_OK = frozenset({"no-cache-model"})


def test_wrapper_without_a_real_base_class_is_flagged():
    problems = check_file(FIX / "bad_wrapper.py")
    assert len(problems) == 1 and "MeteredEmbeddings" in problems[0] and "embed_documents" in problems[0]


def test_wrapper_with_the_real_base_class_passes():
    assert check_file(FIX / "good_wrapper.py") == []


def test_the_real_base_class_makes_isinstance_checks_pass():
    """Why the guard exists: an SDK's isinstance check rejected a look-alike wrapper and stopped ingestion."""
    from fixtures.bad_wrapper import MeteredEmbeddings as Lookalike
    from fixtures.good_wrapper import MeteredEmbeddings as Real
    assert isinstance(Real(None), Embeddings) and not isinstance(Lookalike(None), Embeddings)


def test_current_rates_match_the_pinned_snapshot(prices):
    assert check_rate_drift(prices, PINNED, no_cache_ok=NO_CACHE_OK) == []
    assert snapshot(prices) == PINNED


def test_a_changed_rate_is_reported_as_drift(prices):
    rates = dict(prices.rates)
    rates["acme-chat"] = replace(rates["acme-chat"], output_per_mtok=D(30))
    problems = check_rate_drift(PriceTable(rates), PINNED, no_cache_ok=NO_CACHE_OK)
    assert problems == ["acme-chat.output_per_mtok: pinned 15, live 30"]


def test_an_unpinned_model_and_a_removed_model_are_reported(prices):
    rates = dict(prices.rates)
    del rates["reranker"]
    rates["brand-new"] = Rates(input_per_mtok=D(1), output_per_mtok=D(2), cached_input_per_mtok=D("0.1"))
    problems = check_rate_drift(PriceTable(rates), PINNED, no_cache_ok=NO_CACHE_OK)
    assert any("reranker: pinned but missing" in p for p in problems)
    assert any("brand-new: priced but not pinned" in p for p in problems)


def test_a_model_with_no_cached_or_output_rate_is_reported():
    table = PriceTable({"half": Rates(input_per_mtok=D(1))})
    problems = check_rate_drift(table, {"half": {"input_per_mtok": "1"}})
    assert any("no output rate" in p for p in problems) and any("no cached input rate" in p for p in problems)
