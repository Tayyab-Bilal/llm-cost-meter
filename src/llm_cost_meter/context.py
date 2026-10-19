"""Turn context: who is paying and which user action started the work.

Stored in a ContextVar so sub-agents, tools and fallbacks inherit it with no call-site changes.
It follows `await`, `asyncio.gather`, `create_task` and `asyncio.to_thread`. It does NOT follow a bare
`loop.run_in_executor` or a plain thread: run those through `contextvars.copy_context().run(...)`.
"""
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass


class MissingIdentity(ValueError):
    """The turn has no real tenant or payer. Failing closed beats billing a catch-all "system" tenant."""


UNATTRIBUTED_TENANTS = {"", "system"}


@dataclass(frozen=True)
class TurnContext:
    request_id: str
    tenant: str
    payer: str
    entry_service: str


_current: ContextVar[TurnContext | None] = ContextVar("llm_cost_meter_turn", default=None)


def current_turn() -> TurnContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError("no active turn: wrap the entry point in `with turn(...)`")
    return ctx


@contextmanager
def activate(ctx: TurnContext) -> Iterator[TurnContext]:
    token = _current.set(ctx)
    try:
        yield ctx
    finally:
        _current.reset(token)


@contextmanager
def turn(*, tenant: str, payer: str, entry_service: str, request_id: str | None = None) -> Iterator[TurnContext]:
    """Start a turn. If one is already active the outer one wins: cost follows the entry point.
    A nested turn for a different payer or tenant is a bug (it would bill the wrong customer), so it raises.
    A missing or catch-all tenant raises too: 72 of 77 rows once landed on a "system" tenant that way."""
    if tenant in UNATTRIBUTED_TENANTS or not payer:
        raise MissingIdentity(f"turn() needs a real tenant and payer, got tenant={tenant!r} payer={payer!r}")
    existing = _current.get()
    if existing is not None:
        if (existing.payer, existing.tenant) != (payer, tenant):
            raise ValueError("nested turn() for a different payer/tenant")
        yield existing
        return
    with activate(TurnContext(request_id or uuid.uuid4().hex, tenant, payer, entry_service)) as ctx:
        yield ctx


def to_task_headers(ctx: TurnContext) -> dict[str, str]:
    """What a background job (e.g. a Celery task) carries so its cost lands on the same turn."""
    return {f"x-turn-{k.replace('_', '-')}": v for k, v in asdict(ctx).items()}


def from_task_headers(headers: dict[str, str]) -> TurnContext:
    return TurnContext(**{k: headers[f"x-turn-{k.replace('_', '-')}"] for k in TurnContext.__dataclass_fields__})
