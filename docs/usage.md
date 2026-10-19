# Usage guide

How to put `llm-cost-meter` in front of your own AI calls. Every `python` block here runs as written
(`tests/test_docs.py` executes them), and later blocks reuse names from earlier ones on the same page.

Money is always an integer count of **micro-credits** (1 credit = 1,000,000 micro-credits). Rates are written
as `Decimal`; rounding happens once, half up, at the end of a calculation.

## 1. Set up a meter

A `Meter` needs a `Ledger` (the source of truth), a balance store (the live cache) and a price table.
`MeterMode.ENFORCE` deducts and refuses; `SHADOW` writes ledger rows only; `OFF` does nothing.

```python
import asyncio
from decimal import Decimal as D

from llm_cost_meter import (
    Ledger, Meter, MeterKind, MeterMode, PriceTable, Rates,
    ainvoke_metered, metered_chat_model, turn,
)
from llm_cost_meter.balance import InMemoryBalanceStore
from llm_cost_meter.fakes import fake_llm_factory

prices = PriceTable({
    "acme-chat": Rates(input_per_mtok=D(3), output_per_mtok=D(15), cached_input_per_mtok=D("0.3")),
    "doc-parser": Rates(per_page=D("0.01")),
    "reranker": Rates(per_rerank=D("0.002")),
})
meter = Meter(Ledger(), InMemoryBalanceStore(), prices, MeterMode.ENFORCE)
meter.topup("acme-a", 1_000_000, key="purchase-1")  # a purchase is a ledger row; never SET a balance
print(meter.balances.get("acme-a"))
```

Use `meter.topup`, not `balances.set`, for purchases. In the real design the billing backend owns the purchased
balance (section 7); `topup` is the stand-in for it.

## 2. Say who is paying: the turn

Open a `turn` once, where the user action starts (a request middleware, a task wrapper). Everything below it
(sub-agents, tools, fallbacks) is billed to that payer without passing anything around. The identity is read
from the turn at call time, never cached on an object, so one shared LLM object can serve many tenants.

```python
from llm_cost_meter import MissingIdentity, current_turn

with turn(tenant="acme", payer="acme-a", entry_service="chat", request_id="req-1") as t:
    assert current_turn() is t

try:
    with turn(tenant="system", payer="acme-a", entry_service="chat"):
        pass
except MissingIdentity as exc:
    print("refused:", exc)  # a missing or catch-all tenant fails closed
```

## 3. Call a model

`metered_chat_model` builds the model through your factory and attaches the credit enforcer. Here the factory
returns a scripted fake; in your code it builds a real chat model (see "Real providers" below).

`ainvoke_metered` returns the partial answer with `truncated=True` when credit runs out. A plain `ainvoke`
also streams internally, so it is enforced and cut off too, but it raises `CreditsExhausted` instead (the text
so far is `exc.partial_text`). If the payer cannot even cover the input, `InsufficientCredits` is raised before
any token is generated.

```python
from llm_cost_meter import CreditsExhausted

async def chat() -> None:
    reply = " ".join(f"w{i}" for i in range(100))
    with turn(tenant="acme", payer="acme-a", entry_service="chat", request_id="req-2"):
        llm = metered_chat_model("chat", "acme-chat", llm_factory=fake_llm_factory(reply, reply), meter=meter)
        result = await ainvoke_metered(llm, [("user", "write a status update")])
        print(len(result.text.split()), "words, truncated =", result.truncated)
        message = await llm.ainvoke([("user", "again")])  # plain ainvoke is metered the same way
        print(len(message.content.split()), "words")

asyncio.run(chat())
print(meter.balances.get("acme-a"), "micro-credits left")
```

Service labels (`"chat"`, `"planner"`, ...) must be registered in `SERVICE_BUCKETS`; an unknown label raises. A
sub-agent label such as `"planner"` is billed under the turn's entry service.

## 4. Charge non-token work

Pages, minutes, embeddings and vision units each have their own meter and are never converted into each other.
`charge_units` runs after the work, is keyed on the *work* (a retry bills once) and never raises. Rerank has no
meter of its own: it is priced per search with `charge_rerank`.

```python
with turn(tenant="acme", payer="acme-a", entry_service="doc_parse", request_id="req-3"):
    first = meter.charge_units(MeterKind.PAGE, 12, work_key="doc-7f3a", service="doc_parse", model="doc-parser")
    retry = meter.charge_units(MeterKind.PAGE, 12, work_key="doc-7f3a", service="doc_parse", model="doc-parser")
    rerank = meter.charge_rerank(3, work_key="search-9", service="rerank", model="reranker")
print(first.micro_credits, first.new, retry.new, rerank.micro_credits)
```

Embedding counts come from the caller (this repo does not ship a tokenizer, so counts are not exact). Audio minutes also come from the caller, not the real audio duration.

## 5. Price realtime voice per modality

Text tokens and audio tokens have very different prices, and each has a cached tier. Price them separately:
billing every token at the audio rate overcharges several times over.

```python
voice = PriceTable({"acme-realtime": Rates(
    input_per_mtok=D(4), cached_input_per_mtok=D("0.4"), output_per_mtok=D(16),
    audio_input_per_mtok=D(32), cached_audio_input_per_mtok=D("0.4"), audio_output_per_mtok=D(64))})
cost = voice.cost_realtime("acme-realtime", text_in=9000, audio_in=1000, text_out=4000, audio_out=1000)
print(cost, "micro-credits")
```

## 6. Fallback chains

Use LangChain's `with_fallbacks` as usual. A provider failure moves on to the fallback and each call is billed
for what it did. A refusal (`InsufficientCredits`) is final: it does not derive from `Exception`, so no fallback
model ever serves it.

```python
async def with_fallback() -> None:
    chain = metered_chat_model("chat", "acme-chat", llm_factory=fake_llm_factory("x y z", fail_after=0),
                            meter=meter).with_fallbacks(
        [metered_chat_model("chat", "acme-chat", llm_factory=fake_llm_factory("fallback answer"), meter=meter)])
    with turn(tenant="acme", payer="acme-a", entry_service="chat", request_id="req-4"):
        print((await chain.ainvoke([("user", "hi")])).content)

asyncio.run(with_fallback())
```

## 7. The backend owns the balance: payer resolution and spend reports

The AI service owns the meter; the billing backend owns the balance. Two small pieces connect them.
`PayerResolver` answers "who pays for this user?" through a TTL cache, then a durable map, then the backend,
and fails closed if none can answer. All users of one tenant resolve to one shared organisation payer.
`SpendReporter` sends whole credits per payer to the backend and carries the remainder.

```python
from llm_cost_meter import FakeBillingBackend, PayerResolver, PayerUnavailable, SpendReporter

backend = FakeBillingBackend()
backend.register_org("acme", "acme-a", credits=1_000)
resolver = PayerResolver(meter.ledger, backend.payer_for, ttl=300)
print(resolver.resolve("acme", "alice"), resolver.resolve("acme", "bob"))  # one shared payer

backend.available = False
try:
    resolver.resolve("acme", "carol")  # not cached, authority down
except PayerUnavailable:
    print("refused: no payer, so no work")
backend.available = True

reporter = SpendReporter(meter.ledger, unit_micro=1_000)  # 1 unit = 1,000 micro-credits in this example
print(reporter.sweep(backend.report_spend).reported)  # whole units sent
print(reporter.sweep(backend.report_spend).reported)  # nothing new, nothing sent twice
```

A failed send leaves the position where it was; the next sweep resends the same operation id, which the
backend applies once. Run `sweep` on a schedule (every couple of minutes).

## 8. Rollout modes

Measure first, then meter, then enforce. In `SHADOW` the ledger gets rows marked as shadow and no balance
moves. `reconcile(..., include_shadow=True)` audits a shadow rollout before you turn enforcement on.

```python
shadow = Meter(Ledger(), InMemoryBalanceStore(), prices, MeterMode.SHADOW)
with turn(tenant="acme", payer="acme-a", entry_service="doc_parse"):
    shadow.charge_units(MeterKind.PAGE, 5, work_key="w1", service="doc_parse", model="doc-parser")
print(len(shadow.ledger.rows()), "shadow row, balance untouched:", shadow.balances.get("acme-a"))
```

## 9. Reconcile against the provider's own numbers

`reconcile` compares billed totals with a provider cost export and names the cause of each gap.
`balance_drift` compares the live cache with the ledger.

```python
from llm_cost_meter.reconcile import balance_drift, reconcile

export = [{"service": "doc_parse", "model": "doc-parser", "meter": "page", "cost_micro": 999_999}]
for gap in reconcile(meter.ledger, prices, export):
    print(gap.service, gap.cause.value, gap.ledger_micro, "vs", gap.provider_micro)
print(balance_drift(meter.ledger, meter.balances, ["acme-a"]))
```

## 10. Build-time guards

Run these in CI. The AST guard fails on a provider client built outside the factory, an unregistered service
label, and a wrapper that imitates a third-party class without subclassing it. The rate-drift check fails when
the price table differs from a reviewed, pinned snapshot.

```bash
python -m llm_cost_meter.guards.ast_guard src/
```

```python
from llm_cost_meter.guards.rate_drift import check_rate_drift, snapshot

pinned = snapshot(prices)  # in CI, load this from a reviewed JSON file instead
changed = PriceTable({**prices.rates, "acme-chat": Rates(input_per_mtok=D(4), output_per_mtok=D(15),
                                                         cached_input_per_mtok=D("0.3"))})
print(check_rate_drift(changed, pinned, no_cache_ok=frozenset({"doc-parser", "reranker"})))
```

## 11. Expose a credits API

A small FastAPI router gives balance, usage, reconciliation and pricing history. The payer comes from a
dependency (your auth layer), never from the URL, so a caller can only read its own data.

```python
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llm_cost_meter import PriceHistory
from llm_cost_meter.api import create_credits_router

history = PriceHistory()
history.set_rates("doc-parser", Rates(per_page=D("0.01")), changed_by="ops", reason="initial price")
app = FastAPI()
app.include_router(create_credits_router(meter, history))
http = TestClient(app)
print(http.get("/credits/balance", headers={"x-payer": "acme-a"}).json())
```

## Real providers

Production code builds real chat models in one factory module (the only place the AST guard allows it) and
passes that function as `llm_factory`. Nothing in this repo imports a provider SDK, and no test touches the
network.
