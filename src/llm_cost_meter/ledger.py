"""Append-only ledger: the source of truth. The live balance is only a cache of it."""
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime

from .units import MeterKind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  idempotency_key TEXT NOT NULL,
  payer TEXT NOT NULL, tenant TEXT NOT NULL, request_id TEXT NOT NULL,
  service TEXT NOT NULL,        -- who is billed: the entry point for sub-agent calls
  origin_service TEXT NOT NULL, -- the label that actually made the call
  meter TEXT NOT NULL,
  units TEXT NOT NULL,
  micro_credits INTEGER NOT NULL,   -- signed for trueup; positive for debit and topup
  kind TEXT NOT NULL CHECK (kind IN ('debit','trueup','topup')),
  shadow INTEGER NOT NULL DEFAULT 0,
  applied INTEGER NOT NULL DEFAULT 0, -- 0 pending, 2 being applied right now, 1 applied (or nothing to apply)
  created_at TEXT NOT NULL,
  UNIQUE (payer, idempotency_key, shadow))  -- keys are per payer; a shadow row must not block the real charge
"""


class Ledger:
    # Simplification: one connection behind a lock, Postgres with NUMERIC and row-level concurrency in production
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._db.execute(_SCHEMA)

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run several statements as one locked, atomic step (the reporter and resolver keep their tables here)."""
        with self._lock:
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                self._db.rollback()
                raise

    def append(self, key: str, *, payer: str, tenant: str, request_id: str, service: str,
               meter: MeterKind | str, units, micro: int, kind: str, shadow: bool = False,
               origin_service: str | None = None, applied: bool = False) -> bool:
        """Insert-or-ignore on (key, shadow): at most one row per (payer, key). True only if this call created it."""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO ledger(idempotency_key,payer,tenant,request_id,service,origin_service,"
                "meter,units,micro_credits,kind,shadow,applied,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (key, payer, tenant, request_id, service, origin_service or service, getattr(meter, "value", meter),
                 str(units), micro, kind, int(shadow), int(applied or micro == 0), datetime.now(UTC).isoformat()),
            )
            self._db.commit()
            return cur.rowcount == 1

    def claim(self, payer: str, key: str) -> bool:
        """Atomically take the right to move the balance for this row (exactly one caller wins)."""
        return self._set_applied(payer, key, 0, 2)

    def finish(self, payer: str, key: str) -> None:
        self._set_applied(payer, key, 2, 1)

    def unclaim(self, payer: str, key: str) -> None:
        """The balance move failed; make the row pending again so a retry applies it."""
        self._set_applied(payer, key, 2, 0)

    def _set_applied(self, payer: str, key: str, old: int, new: int) -> bool:
        with self._lock:
            cur = self._db.execute(
                "UPDATE ledger SET applied=? WHERE payer=? AND idempotency_key=? AND shadow=0 AND applied=?",
                (new, payer, key, old))
            self._db.commit()
            return cur.rowcount == 1

    def settle_all(self, payer: str) -> int:
        """Close every pending row and return the balance, in one locked step. Pending rows must be counted
        *and* closed together, or a later retry would apply them a second time. Rows another caller is
        applying right now (2) are left out: that caller moves the cache itself."""
        with self._lock:
            self._db.execute("UPDATE ledger SET applied=1 WHERE payer=? AND shadow=0 AND applied=0", (payer,))
            row = self._db.execute(
                "SELECT COALESCE(SUM(CASE kind WHEN 'topup' THEN micro_credits ELSE -micro_credits END),0)"
                " FROM ledger WHERE payer=? AND shadow=0 AND applied=1", (payer,)).fetchone()
            self._db.commit()
        return int(row[0])

    def balance_delta(self, payer: str) -> int:
        """What the ledger says the payer's balance changed by (shadow rows excluded)."""
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(CASE kind WHEN 'topup' THEN micro_credits ELSE -micro_credits END),0)"
                " FROM ledger WHERE payer=? AND shadow=0", (payer,)).fetchone()
        return int(row[0])

    def rows(self, request_id: str | None = None, *, topups: bool = False,
             payer: str | None = None) -> list[sqlite3.Row]:
        """Usage rows in insert order; purchases are left out unless asked for."""
        where, args = ["1=1"], []
        if payer:
            where.append("payer=?")
            args.append(payer)
        if request_id:
            where.append("request_id=?")
            args.append(request_id)
        if not topups:
            where.append("kind != 'topup'")
        with self._lock:
            return self._db.execute(f"SELECT * FROM ledger WHERE {' AND '.join(where)} ORDER BY id", args).fetchall()

    def totals_by_service(self, include_shadow: bool = False) -> dict[str, int]:
        """Net billed micro-credits per billed service (top-ups excluded)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT service, SUM(micro_credits) FROM ledger WHERE kind != 'topup' AND (shadow=0 OR ?)"
                " GROUP BY service", (int(include_shadow),)).fetchall()
        return {r[0]: int(r[1]) for r in rows}
