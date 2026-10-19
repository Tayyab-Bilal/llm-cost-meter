"""Append-only price history: every change is a new row with who and why; nothing is edited or deleted."""
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from .pricing import PriceTable, Rates


@dataclass(frozen=True)
class PriceChange:
    seq: int
    model: str
    rates: Rates
    changed_by: str
    reason: str
    at: str


class PriceHistory:
    # Simplification: an in-memory list; production is a table with row-level locking and an approval queue.
    def __init__(self) -> None:
        self._rows: list[PriceChange] = []
        self._lock = threading.Lock()

    def set_rates(self, model: str, rates: Rates, *, changed_by: str, reason: str) -> PriceChange:
        if not changed_by or not reason:
            raise ValueError("a price change needs who and why")
        with self._lock:
            change = PriceChange(len(self._rows) + 1, model, rates, changed_by, reason, datetime.now(UTC).isoformat())
            self._rows.append(change)
        return change

    def history(self, model: str | None = None) -> list[PriceChange]:
        with self._lock:
            return [c for c in self._rows if model is None or c.model == model]

    def table(self) -> PriceTable:
        """The current prices: the latest change per model."""
        with self._lock:
            return PriceTable({c.model: c.rates for c in self._rows})
