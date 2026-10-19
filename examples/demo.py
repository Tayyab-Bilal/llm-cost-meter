"""End-to-end walkthrough with fakes: attribution, partial answers, idempotent charges, the guard."""
import asyncio
import logging
import os
from decimal import Decimal as D
from pathlib import Path

from llm_cost_meter import (
    FakeBillingBackend,
    Ledger,
    Meter,
    MeterKind,
    MeterMode,
    PayerResolver,
    PriceTable,
    Rates,
    SpendReporter,
    ainvoke_metered,
    metered_chat_model,
    turn,
)
from llm_cost_meter.balance import InMemoryBalanceStore
from llm_cost_meter.fakes import fake_llm_factory
from llm_cost_meter.guards import ast_guard

# LangChain logs a warning whenever a handler raises; our cancel signal is expected, so keep the output clean
logging.getLogger("langchain_core.callbacks.manager").setLevel(logging.ERROR)

PRICES = PriceTable({
    "acme-chat": Rates(input_per_mtok=D(3), output_per_mtok=D(15), cached_input_per_mtok=D("0.3")),
    "acme-planner": Rates(input_per_mtok=D(1), output_per_mtok=D(5), cached_input_per_mtok=D("0.1")),
    "doc-parser": Rates(per_page=D("0.01")),
    "reranker": Rates(per_rerank=D("0.002")),
})


def show(meter: Meter, title: str) -> None:
    print(f"\n--- {title}")
    print(f"{'request':<10}{'payer':<9}{'service':<11}{'meter':<13}{'kind':<7}{'units':>7}{'micro':>9}")
    for r in meter.ledger.rows():
        print(f"{r['request_id']:<10}{r['payer']:<9}{r['service']:<11}{r['meter']:<13}{r['kind']:<7}"
              f"{r['units']:>7}{r['micro_credits']:>9}")
    for payer in ("acme-a", "acme-b"):
        print(f"balance {payer}: {meter.balances.get(payer):>9} micro-credits")


async def main() -> None:
    meter = Meter(Ledger(), InMemoryBalanceStore(), PRICES, MeterMode.ENFORCE)
    meter.topup("acme-a", 1_000_000, key="buy-a1")  # tenant A: plenty
    meter.topup("acme-b", 3_000, key="buy-b1")  # tenant B: about 190 output tokens

    answer = " ".join(f"w{i}" for i in range(150))
    plan = fake_llm_factory("1 find the notes 2 summarise 3 reply")
    chat = fake_llm_factory(answer)

    print("1) Tenant A: one chat turn. The planner sub-agent and the rerank are billed to the same request.")
    with turn(tenant="acme", payer="acme-a", entry_service="chat", request_id="req-A1"):
        planner = metered_chat_model("planner", "acme-planner", llm_factory=plan, meter=meter)
        await ainvoke_metered(planner, [("user", "plan a project status update")])  # no turn plumbing here
        result = await ainvoke_metered(metered_chat_model("chat", "acme-chat", llm_factory=chat, meter=meter),
                                       [("user", "write the status update")])
        meter.charge_rerank(20, work_key="req-A1-rerank", service="rerank",
                           model="reranker")
        print(f"   answer: {len(result.text.split())} words, truncated={result.truncated}")

    print("2) Tenant B runs dry mid-answer: partial text, no error (delivered tokens settled).")
    with turn(tenant="acme", payer="acme-b", entry_service="chat", request_id="req-B1"):
        llm = metered_chat_model("chat", "acme-chat", llm_factory=fake_llm_factory(" ".join([answer] * 3)), meter=meter)
        result = await ainvoke_metered(llm, [("user", "write a very long report")])
        print(f"   answer: {len(result.text.split())} of {len(answer.split()) * 3} words, "
              f"truncated={result.truncated}")

    print("3) A document-parse job is retried after a worker crash: billed once.")
    with turn(tenant="acme", payer="acme-a", entry_service="doc_parse", request_id="req-A2"):
        for attempt in (1, 2):
            r = meter.charge_units(MeterKind.PAGE, 12, work_key="doc-7f3a", service="doc_parse",
                                   model="doc-parser")
            print(f"   attempt {attempt}: new_row={r.new}")

    show(meter, "ledger and balances")

    print("\n4) The backend owns the balance. Resolve the payer, then report whole units of spend to it.")
    backend = FakeBillingBackend()
    backend.register_org("acme", "acme-a", credits=1_000)
    resolver = PayerResolver(meter.ledger, backend.payer_for)
    alice, bob = resolver.resolve("acme", "alice"), resolver.resolve("acme", "bob")
    print(f"   alice -> {alice}, bob -> {bob} (one shared payer)")
    reporter = SpendReporter(meter.ledger, unit_micro=1_000)  # a unit of 1,000 micro-credits keeps the demo small
    first = reporter.sweep(backend.report_spend)
    second = reporter.sweep(backend.report_spend)
    print(f"   sweep 1 reported {first.reported}, carried {reporter.remainder('acme-a')} micro; "
          f"sweep 2 reported {second.reported}")

    print("\n5) Build-time guard over examples/bad_service.py")
    code = ast_guard.main([os.path.relpath(Path(__file__).with_name("bad_service.py"))])
    print(f"guard exit code: {code}")


asyncio.run(main())
