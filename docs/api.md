# API reference

Everything below is importable from `llm_cost_meter` unless a module is named. Money is an `int` of
micro-credits (`MICRO` = 1,000,000 per credit). See [usage.md](usage.md) for worked examples.

## Meters and units (`units`)

| Name | Description |
|---|---|
| `MeterKind` | `TOKEN`, `PAGE`, `AUDIO_MINUTE`, `EMBED_TOKEN`, `VISION_UNIT`. Separate meters, never converted into each other. Rerank has no meter: it is priced per search and filed under `EMBED_TOKEN`. |
| `MICRO` | `1_000_000`. |
| `to_units(amount: Decimal) -> int` | Round once, half up. |
| `usd_to_units(usd) -> int` | `Decimal`, `str` or `int` USD to micro-credits. |

## Pricing (`pricing`, `pricing_history`)

| Name | Description |
|---|---|
| `Rates(...)` | Frozen dataclass of `Decimal | None` rates: `input_per_mtok`, `output_per_mtok`, `cached_input_per_mtok`, `per_page`, `per_audio_minute`, `per_embed_mtok`, `per_vision_unit`, `per_rerank` (per search), and the realtime audio rates `audio_input_per_mtok`, `audio_output_per_mtok`, `cached_audio_input_per_mtok`. |
| `PriceTable(rates: dict[str, Rates])` | Model name to rates. |
| `PriceTable.cost_tokens(model, input_tokens, output_tokens, cached_tokens=0) -> int` | `input_tokens` includes the cached ones. Reasoning tokens are part of `output_tokens` and billed at the output rate. |
| `PriceTable.cost_units(model, kind, units) -> int` | Non-token work. `units` is `int`, `Decimal` or `str`, never `float`. Embeddings are per million tokens. |
| `PriceTable.cost_rerank(model, searches) -> int` | Rerank, per search. |
| `PriceTable.cost_realtime(model, *, text_in, audio_in, text_out, audio_out, cached_text_in, cached_audio_in) -> int` | Realtime voice per modality and cache tier. `text_in` and `audio_in` include their cached tokens. |
| `PriceHistory` | Append-only price log. `set_rates(model, rates, *, changed_by, reason) -> PriceChange`, `history(model=None) -> list[PriceChange]`, `table() -> PriceTable` (latest per model). |

## Identity (`context`, `resolver`)

| Name | Description |
|---|---|
| `turn(*, tenant, payer, entry_service, request_id=None)` | Context manager. Opens the turn that every call below it is billed to. A nested turn for the same payer reuses the outer one; a different payer raises `ValueError`. A missing tenant, a `"system"` tenant or an empty payer raises `MissingIdentity`. |
| `current_turn() -> TurnContext` | The active turn; raises `RuntimeError` outside one. |
| `TurnContext` | Frozen dataclass: `request_id`, `tenant`, `payer`, `entry_service`. |
| `to_task_headers(ctx)` / `from_task_headers(headers)` (`context`) | Carry a turn into a background job. |
| `PayerResolver(ledger, authority, *, ttl=300.0, clock=time.monotonic)` | `resolve(tenant, user_id) -> str`. Cache, then durable map, then `authority(tenant, user_id)`. Raises `PayerUnavailable` when nothing can answer and `MissingIdentity` for a missing tenant or user. |

## Ledger, balance and meter (`ledger`, `balance`, `meter`)

| Name | Description |
|---|---|
| `Ledger(path=":memory:")` | Append-only SQLite ledger. `append(...)`, `rows(request_id=None, *, topups=False, payer=None)`, `balance_delta(payer)`, `totals_by_service(include_shadow=False)`, `transaction()`. Idempotent per `(payer, key, shadow)`. |
| `InMemoryBalanceStore`, `RedisBalanceStore(client)` (`balance`) | Live balance cache. `get`, `set` (seeding and rebuild only), `decrement_if_sufficient(payer, micro) -> (ok, remaining)`, `adjust(payer, delta, key=None) -> bool`. `adjust` and decrement refuse a missing key. |
| `Meter(ledger, balances, prices, mode=MeterMode.ENFORCE)` | The meter. |
| `Meter.topup(payer, micro, *, key, tenant="system") -> bool` | Record a purchase, then move the cache. |
| `Meter.debit(key, service, units, micro) -> bool` | Charge while streaming; `False` means the payer cannot cover it. |
| `Meter.trueup(key, service, units, delta_micro)` | Signed correction; a negative delta credits back and never creates a balance. |
| `Meter.charge_units(kind, units, *, work_key, service, model=None) -> ChargeResult` | Bill finished non-token work. Keyed on the work. Never raises, except for an unregistered service label while enforcing. |
| `Meter.charge_rerank(searches, *, work_key, service, model=None) -> ChargeResult` | Per-search rerank charge. |
| `Meter.rebuild_balance(payer) -> int` | Reset the cache from the ledger. Run with the payer quiet. |
| `MeterMode` | `OFF`, `SHADOW`, `ENFORCE`. |
| `ChargeResult` | `ok`, `new` (False on a retried key), `micro_credits`, `error`. |

## Calling models (`factory`, `streaming`)

| Name | Description |
|---|---|
| `metered_chat_model(service, model, *, llm_factory, meter, batch_tokens=64) -> BaseChatModel` | Builds the model with your factory and attaches the credit enforcer. Calls stream internally, so `ainvoke`, `astream` and fallback chains are all enforced. Raises `ValueError` for a per-unit service label. |
| `ainvoke_metered(llm, messages) -> MeteredResult` | Stream and return `MeteredResult(text, truncated, usage)`. Out of credit mid-answer gives the partial text with `truncated=True`. |
| `InsufficientCredits` | Raised before any token. Derives from `BaseException` so fallbacks never serve it. |
| `CreditsExhausted` | Mid-stream cut-off signal; `partial_text` holds what was delivered. A plain `ainvoke` raises it. |
| `CreditEnforcer(meter, service, model, batch_tokens=64)` (`streaming`) | The LangChain callback handler: pre-flight estimate, batched deduction, signed true-up. |

## Backend and reporting (`backend`, `reporter`)

| Name | Description |
|---|---|
| `FakeBillingBackend` | Test double for the billing backend. `register_org(tenant, payer, credits=0)`, `payer_for(tenant, user)`, `report_spend(report)`, `reported_units(payer)`, `available` flag. Raises `BackendUnavailable` when down. |
| `SpendReport(op_id, payer, units)` | One report; the backend applies each `op_id` once. |
| `SpendReporter(ledger, *, unit_micro=MICRO)` | `sweep(send) -> SweepResult`, `position(payer)`, `remainder(payer)`. Whole units only, remainder carried, positions forward only, attempt recorded before send, one claim per payer. |
| `SweepResult` | `reported` (payer to units), `skipped` (claimed elsewhere), `failed`. |

## Service labels (`buckets`)

| Name | Description |
|---|---|
| `SERVICE_BUCKETS`, `bucket_for(label) -> Bucket` | Label to billing bucket. Unknown labels raise `UnregisteredBillingLabel`. |
| `Bucket` | `CHARGED`, `CHARGED_TO_ENTRY_POINT`, `RECORDED_ZERO`, `SEPARATELY_METERED`. |

## Reconciliation (`reconcile`)

| Name | Description |
|---|---|
| `reconcile(ledger, prices, export, *, include_shadow=False) -> list[Mismatch]` | Compare billed totals with a provider export. Causes: `UNMETERED_SERVICE`, `NO_PROVIDER_SPEND`, `MISSING_CACHED_RATE`, `PRICE_MISMATCH`, `UNKNOWN`. |
| `balance_drift(ledger, balances, payers) -> dict[str, int]` | Cache minus ledger, only for payers that differ. |

## Build guards (`guards`)

| Name | Description |
|---|---|
| `python -m llm_cost_meter.guards.ast_guard <path>...` | Exit 1 on a provider client outside the factory, an unregistered or non-literal service label, or a look-alike wrapper class with no real base class. `# billing-guard: ok` suppresses one reviewed line. |
| `check_file(path) -> list[str]` (`guards.ast_guard`) | The same check from Python. |
| `snapshot(prices)`, `check_rate_drift(prices, pinned, *, no_cache_ok=frozenset()) -> list[str]` (`guards.rate_drift`) | Pin the reviewed rates and fail when the live table differs, a model is unpinned, or a rate is missing. |

## Credits API (`api`)

`create_credits_router(meter, history, payer_dependency=header_payer) -> APIRouter`, mounted under `/credits`.

| Route | Description |
|---|---|
| `GET /credits/balance` | The caller's cache balance, ledger balance and drift. |
| `GET /credits/usage?request_id=` | The caller's ledger rows and their total. |
| `POST /credits/reconciliation` | Body `{"export": [...], "include_shadow": false}`; returns mismatches with causes. Operator view. |
| `GET /credits/pricing-history?model=` | Price changes in order, with who and why. Operator view. |

`header_payer` reads `X-Payer` and returns 401 without it; replace it with a dependency that reads your auth token.

```python
from llm_cost_meter import MeterKind, PriceTable, Rates

assert {m.name for m in MeterKind} == {"TOKEN", "PAGE", "AUDIO_MINUTE", "EMBED_TOKEN", "VISION_UNIT"}
from decimal import Decimal
assert PriceTable({"m": Rates(per_rerank=Decimal("0.002"))}).cost_rerank("m", 3) == 6_000
```
