from decimal import Decimal as D

import pytest

from llm_cost_meter import Ledger, Meter, MeterMode, PriceTable, Rates
from llm_cost_meter.balance import InMemoryBalanceStore


@pytest.fixture
def prices() -> PriceTable:
    return PriceTable({
        "acme-chat": Rates(input_per_mtok=D(3), output_per_mtok=D(15), cached_input_per_mtok=D("0.3")),
        "cheap-model": Rates(input_per_mtok=D(1), output_per_mtok=D(5), cached_input_per_mtok=D("0.1")),
        "no-cache-model": Rates(input_per_mtok=D(3), output_per_mtok=D(15)),
        "doc-parser": Rates(per_page=D("0.01")),
        "reranker": Rates(per_rerank=D("0.002")),
    })


@pytest.fixture
def make_meter(prices):
    def make(mode=MeterMode.ENFORCE, balance=1_000_000, balances=None):
        balances = balances or InMemoryBalanceStore()
        meter = Meter(Ledger(), balances, prices, mode)
        meter.topup("acme", balance, key="seed")  # through the ledger, so cache and ledger agree
        return meter
    return make
