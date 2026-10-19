"""Compare what we billed with what providers say we spent, and name the cause of each gap."""
from dataclasses import dataclass
from enum import StrEnum

from .balance import BalanceStore
from .buckets import Bucket, UnregisteredBillingLabel, bucket_for
from .ledger import Ledger
from .pricing import PriceTable


class Cause(StrEnum):
    UNMETERED_SERVICE = "unmetered_service"  # spend with no ledger rows at all
    NO_PROVIDER_SPEND = "no_provider_spend"  # we billed it, the provider shows nothing
    MISSING_CACHED_RATE = "missing_cached_rate"
    PRICE_MISMATCH = "price_mismatch"  # our rates do not reproduce the provider's cost
    UNKNOWN = "unknown"


@dataclass
class Mismatch:
    service: str
    ledger_micro: int
    provider_micro: int
    cause: Cause


def _free_by_design(svc: str) -> bool:
    """RECORDED_ZERO services are tracked but never billed, so provider spend there is expected."""
    try:
        return bucket_for(svc) is Bucket.RECORDED_ZERO
    except UnregisteredBillingLabel:
        return False


def reconcile(ledger: Ledger, prices: PriceTable, export: list[dict], *,
              include_shadow: bool = False) -> list[Mismatch]:
    """`export` rows: service, model, input_tokens, output_tokens, cached_tokens, cost_micro, and optionally
    `meter` (default "token"). Rows are summed per service first. Services are the ledger's billed labels
    (sub-agent spend sits under its entry point). Set `include_shadow` to audit a SHADOW rollout."""
    billed = ledger.totals_by_service(include_shadow)
    by_service: dict[str, list[dict]] = {}
    for row in export:
        by_service.setdefault(row["service"], []).append(row)
    out = []
    for svc in sorted(by_service.keys() | billed.keys()):
        rows = by_service.get(svc, [])
        provider, mine = sum(r["cost_micro"] for r in rows), billed.get(svc, 0)
        if mine == provider or _free_by_design(svc):
            continue
        token_rows = [r for r in rows if r.get("meter", "token") == "token"]  # unit-metered work has no token maths
        if not rows:
            cause = Cause.NO_PROVIDER_SPEND
        elif svc not in billed:
            cause = Cause.UNMETERED_SERVICE
        elif any(r.get("cached_tokens") and not prices.has_cached_rate(r["model"]) for r in token_rows):
            cause = Cause.MISSING_CACHED_RATE
        elif token_rows and len(token_rows) == len(rows) and sum(
                prices.cost_tokens(r["model"], r["input_tokens"], r["output_tokens"], r.get("cached_tokens", 0))
                for r in rows) != provider:
            cause = Cause.PRICE_MISMATCH
        else:
            cause = Cause.UNKNOWN
        out.append(Mismatch(svc, mine, provider, cause))
    return out


def balance_drift(ledger: Ledger, balances: BalanceStore, payers: list[str]) -> dict[str, int]:
    """Cache minus ledger for each payer; non-zero means the cache drifted (fix with Meter.rebuild_balance)."""
    return {p: (balances.get(p) or 0) - ledger.balance_delta(p) for p in payers
            if (balances.get(p) or 0) != ledger.balance_delta(p)}
