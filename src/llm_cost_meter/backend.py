"""A fake billing backend. The backend owns the purchased balance; the AI service only owns the meter.

It plays two roles: the payer authority (who pays for this user?) and the receiver of spend reports.
Production is an HTTP service; here it is an object so tests need no network.
"""
import threading
from dataclasses import dataclass


class BackendUnavailable(Exception):
    """The backend cannot be reached (timeout, outage). Callers must fail closed."""


class UnknownTenant(Exception):
    pass


@dataclass(frozen=True)
class SpendReport:
    op_id: str  # idempotency key: the backend applies each op_id once, so a retry is safe
    payer: str
    units: int  # whole credits


class FakeBillingBackend:
    def __init__(self) -> None:
        self.available = True
        self.org_payer: dict[str, str] = {}  # tenant -> the one payer (organisation) all its users share
        self.balances: dict[str, int] = {}  # payer -> whole credits left
        self.ops: dict[str, SpendReport] = {}
        self.send_calls = 0
        self.authority_calls = 0
        self._lock = threading.Lock()

    def register_org(self, tenant: str, payer: str, credits: int = 0) -> None:
        self.org_payer[tenant] = payer
        self.balances.setdefault(payer, credits)

    def payer_for(self, tenant: str, user: str) -> str:
        """The authority. Every user of a tenant resolves to the tenant's organisation payer."""
        self.authority_calls += 1
        if not self.available:
            raise BackendUnavailable("billing backend is down")
        try:
            return self.org_payer[tenant]
        except KeyError:
            raise UnknownTenant(tenant) from None

    def report_spend(self, report: SpendReport) -> None:
        with self._lock:
            self.send_calls += 1
            if not self.available:
                raise BackendUnavailable("billing backend is down")
            if report.op_id in self.ops:  # a retry of an op already applied
                return
            self.ops[report.op_id] = report
            self.balances[report.payer] = self.balances.get(report.payer, 0) - report.units

    def reported_units(self, payer: str) -> int:
        return sum(r.units for r in self.ops.values() if r.payer == payer)
