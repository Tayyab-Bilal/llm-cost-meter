"""The meter: turns work into ledger rows and balance changes, according to the rollout mode.

The ledger is the truth. Every balance move is either made *before* its row (debit, with compensation)
or *after* it (everything else, guarded by the row's `applied` flag so a retry finishes the job).
"""
import logging
import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from .balance import BalanceStore
from .buckets import Bucket, UnregisteredBillingLabel, bucket_for
from .context import current_turn
from .ledger import Ledger
from .pricing import PriceTable
from .units import MeterKind

log = logging.getLogger(__name__)


class MeterMode(StrEnum):
    """Rollout: measure first (OFF + reports), then SHADOW (ledger only), then ENFORCE."""

    OFF = "off"
    SHADOW = "shadow"
    ENFORCE = "enforce"


@dataclass
class ChargeResult:
    ok: bool
    new: bool = False  # False on a retried work_key: the row already existed
    micro_credits: int = 0
    error: str | None = None


class Meter:
    def __init__(self, ledger: Ledger, balances: BalanceStore, prices: PriceTable,
                 mode: MeterMode = MeterMode.ENFORCE):
        self.ledger, self.balances, self.prices, self.mode = ledger, balances, prices, mode

    def _apply(self, payer: str, key: str, delta: int) -> None:
        """Move the balance once for a ledger row. Safe to call again on retry."""
        if delta == 0 or not self.ledger.claim(payer, key):
            return
        try:
            moved = self.balances.adjust(payer, delta, key)  # the key makes an ambiguous failure safe to retry
        except Exception:
            self.ledger.unclaim(payer, key)  # so the next retry applies it
            raise
        self.ledger.finish(payer, key)
        if not moved:  # row is kept and counts as applied: rebuild_balance() restores the cache from the ledger
            log.error("no balance for payer %s; ledger row %s kept, run rebuild_balance()", payer, key)

    def _post(self, key: str, service: str, kind_meter: MeterKind, units, micro: int, kind: str,
              pre_applied: bool = False) -> bool:
        turn = current_turn()
        bucket = bucket_for(service)
        micro = 0 if bucket is Bucket.RECORDED_ZERO else micro
        billed = turn.entry_service if bucket is Bucket.CHARGED_TO_ENTRY_POINT else service
        new = self.ledger.append(key, payer=turn.payer, tenant=turn.tenant, request_id=turn.request_id,
                                 service=billed, origin_service=service, meter=kind_meter, units=units,
                                 micro=micro, kind=kind, shadow=self.mode is MeterMode.SHADOW,
                                 applied=pre_applied)
        if self.mode is MeterMode.ENFORCE and not pre_applied:
            self._apply(turn.payer, key, -micro)  # also finishes a retried row whose first apply failed
        return new

    def debit(self, key: str, service: str, units: int, micro: int) -> bool:
        """Charge while streaming. False means the payer cannot cover it (nothing was charged)."""
        if micro < 0:
            raise ValueError("debit must be >= 0; use trueup for corrections")
        if self.mode is not MeterMode.ENFORCE:
            if self.mode is MeterMode.SHADOW:
                self._post(key, service, MeterKind.TOKEN, units, micro, "debit")
            return True
        if micro == 0 or bucket_for(service) is Bucket.RECORDED_ZERO:
            self._post(key, service, MeterKind.TOKEN, units, micro, "debit")
            return True
        payer = current_turn().payer
        ok, _ = self.balances.decrement_if_sufficient(payer, micro)
        if not ok:
            return False
        try:
            new = self._post(key, service, MeterKind.TOKEN, units, micro, "debit", pre_applied=True)
        except Exception:
            self._refund(payer, key, micro)  # the user must not pay for a charge with no record
            raise
        if not new:  # retry of an already-billed key: undo the duplicate decrement
            self._refund(payer, key, micro)
        return True

    def _refund(self, payer: str, key: str, micro: int) -> None:
        # Simplification: one-off refund keys, so an ambiguous failure of the refund itself is not deduplicated
        self.balances.adjust(payer, micro, f"{key}:refund:{uuid.uuid4().hex}")

    def trueup(self, key: str, service: str, units: int, delta_micro: int) -> None:
        """Signed correction; a negative delta credits the payer back."""
        if self.mode is not MeterMode.OFF and delta_micro != 0:
            self._post(key, service, MeterKind.TOKEN, units, delta_micro, "trueup")

    def topup(self, payer: str, micro: int, *, key: str, tenant: str = "system") -> bool:
        """Record a purchase in the ledger, then move the cache. Never wipes a balance with a plain SET."""
        new = self.ledger.append(key, payer=payer, tenant=tenant, request_id="topup", service="billing",
                                 meter="topup", units=micro, micro=micro, kind="topup")
        if self.ledger.claim(payer, key):
            try:
                moved = self.balances.adjust(payer, micro, key)
            except Exception:
                self.ledger.unclaim(payer, key)
                raise
            self.ledger.finish(payer, key)
            if not moved:
                self.rebuild_balance(payer)  # first top-up: the ledger (including this row) is the balance
        return new

    def rebuild_balance(self, payer: str) -> int:
        """Reset the cache from the ledger, e.g. after the cache was lost or an adjust was refused.
        # Simplification: run with the payer quiesced; a debit between the ledger read and the SET is lost.
        Pending rows are counted and closed together, so a later retry cannot apply them again."""
        value = self.ledger.settle_all(payer)
        self.balances.set(payer, value)
        return value

    def charge_units(self, kind: MeterKind, units: Decimal | int, *, work_key: str, service: str,
                     model: str | None = None) -> ChargeResult:
        """Bill finished non-token work. Keyed on the *work*, so a retried job bills once. Never raises
        (except on an unregistered service label while enforcing: that is a bug, so it is loud)."""
        return self._charge(kind, units, work_key, service,
                            lambda: self.prices.cost_units(model or service, kind, units))

    def charge_rerank(self, searches: int, *, work_key: str, service: str, model: str | None = None) -> ChargeResult:
        """Rerank is priced per search and has no meter of its own; the ledger files it under EMBED_TOKEN
        (the retrieval meter), keyed `embed_token:<work_key>`."""
        return self._charge(MeterKind.EMBED_TOKEN, searches, work_key, service,
                            lambda: self.prices.cost_rerank(model or service, searches))

    def _charge(self, kind: MeterKind, units, work_key: str, service: str, cost) -> ChargeResult:
        try:
            bucket_for(service)  # an unregistered label is a bug: loud when enforcing, logged otherwise
        except UnregisteredBillingLabel as exc:
            if self.mode is MeterMode.ENFORCE:
                raise
            log.error("charge_units: %s", exc)
            return ChargeResult(ok=False, error=repr(exc))
        try:
            if self.mode is MeterMode.OFF:
                return ChargeResult(ok=True)
            micro = cost()
            new = self._post(f"{kind.value}:{work_key}", service, kind, units, micro, "debit")
            return ChargeResult(ok=True, new=new, micro_credits=micro)
        except Exception as exc:  # billing must never break the user's work
            log.exception("charge_units failed for %s", work_key)
            return ChargeResult(ok=False, error=repr(exc))
