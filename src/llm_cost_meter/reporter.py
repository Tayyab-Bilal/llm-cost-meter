"""Report spend to the billing backend. The backend owns the balance; this is the AI service's meter output.

Per payer, a sweep reports whole credits only and carries the sub-credit remainder to the next sweep.
It records each attempt before sending, moves the ledger position forward only after the backend accepted,
and takes a per-payer claim so two overlapping sweeps never report the same spend twice. A failed attempt is
retried with the same operation id, which the backend applies once.

Postgres would claim a payer with `SELECT ... FOR UPDATE SKIP LOCKED`; here a claim row does the same job.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

from .backend import SpendReport
from .ledger import Ledger
from .units import MICRO

_SCHEMA = [
    "CREATE TABLE IF NOT EXISTS report_state(payer TEXT PRIMARY KEY, position INTEGER NOT NULL,"
    " remainder INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS report_attempts(op_id TEXT PRIMARY KEY, payer TEXT NOT NULL,"
    " to_pos INTEGER NOT NULL, units INTEGER NOT NULL, remainder_after INTEGER NOT NULL,"
    " status TEXT NOT NULL CHECK (status IN ('pending','failed','sent')), error TEXT)",
    "CREATE TABLE IF NOT EXISTS report_claims(payer TEXT PRIMARY KEY)",
]


@dataclass
class SweepResult:
    reported: dict[str, int] = field(default_factory=dict)  # payer -> whole credits sent
    skipped: list[str] = field(default_factory=list)  # another sweep holds the claim
    failed: list[str] = field(default_factory=list)  # send raised; will retry with the same op id


class SpendReporter:
    def __init__(self, ledger: Ledger, *, unit_micro: int = MICRO) -> None:
        self.ledger, self.unit_micro = ledger, unit_micro
        with ledger.transaction() as db:
            for stmt in _SCHEMA:
                db.execute(stmt)

    def position(self, payer: str) -> int:
        with self.ledger.transaction() as db:
            row = db.execute("SELECT position FROM report_state WHERE payer=?", (payer,)).fetchone()
        return row[0] if row else 0

    def remainder(self, payer: str) -> int:
        with self.ledger.transaction() as db:
            row = db.execute("SELECT remainder FROM report_state WHERE payer=?", (payer,)).fetchone()
        return row[0] if row else 0

    def sweep(self, send: Callable[[SpendReport], None]) -> SweepResult:
        """One pass over every payer with spend. `send` raises on failure."""
        result = SweepResult()
        with self.ledger.transaction() as db:
            payers = [r[0] for r in db.execute("SELECT DISTINCT payer FROM ledger WHERE shadow=0").fetchall()]
        for payer in sorted(payers):
            if not self._claim(payer):
                result.skipped.append(payer)
                continue
            try:
                self._report(payer, send, result)
            finally:
                self._release(payer)
        return result

    def _claim(self, payer: str) -> bool:
        with self.ledger.transaction() as db:
            return db.execute("INSERT OR IGNORE INTO report_claims(payer) VALUES(?)", (payer,)).rowcount == 1

    def _release(self, payer: str) -> None:
        with self.ledger.transaction() as db:
            db.execute("DELETE FROM report_claims WHERE payer=?", (payer,))

    def _report(self, payer: str, send: Callable[[SpendReport], None], result: SweepResult) -> None:
        attempt = self._open_attempt(payer) or self._new_attempt(payer)
        if attempt is None:
            return
        op_id, units = attempt
        try:
            send(SpendReport(op_id, payer, units))
        except Exception as exc:
            with self.ledger.transaction() as db:
                db.execute("UPDATE report_attempts SET status='failed', error=? WHERE op_id=?", (repr(exc), op_id))
            result.failed.append(payer)
            return
        with self.ledger.transaction() as db:
            if db.execute("UPDATE report_attempts SET status='sent' WHERE op_id=? AND status!='sent'",
                          (op_id,)).rowcount == 1:
                to_pos, rem = db.execute("SELECT to_pos, remainder_after FROM report_attempts WHERE op_id=?",
                                         (op_id,)).fetchone()
                self._advance(db, payer, to_pos, rem)
        result.reported[payer] = units

    @staticmethod
    def _advance(db, payer: str, to_pos: int, remainder: int) -> None:
        """Forward only: a position never moves back, even if an old attempt is replayed."""
        db.execute("INSERT OR IGNORE INTO report_state(payer, position, remainder) VALUES(?,0,0)", (payer,))
        db.execute("UPDATE report_state SET position=?, remainder=? WHERE payer=? AND position<=?",
                   (to_pos, remainder, payer, to_pos))

    def _open_attempt(self, payer: str) -> tuple[str, int] | None:
        """An earlier attempt that was not confirmed: resend it unchanged, so its op id dedupes."""
        with self.ledger.transaction() as db:
            row = db.execute("SELECT op_id, units FROM report_attempts WHERE payer=? AND status!='sent'",
                             (payer,)).fetchone()
        return (row[0], row[1]) if row else None

    def _new_attempt(self, payer: str) -> tuple[str, int] | None:
        with self.ledger.transaction() as db:
            row = db.execute("SELECT position, remainder FROM report_state WHERE payer=?", (payer,)).fetchone()
            pos, carried = row if row else (0, 0)
            top = db.execute("SELECT COALESCE(MAX(id),0) FROM ledger WHERE payer=? AND shadow=0",
                             (payer,)).fetchone()[0]
            if top <= pos:
                return None
            spent = db.execute(
                "SELECT COALESCE(SUM(micro_credits),0) FROM ledger WHERE payer=? AND shadow=0"
                " AND kind IN ('debit','trueup') AND id>? AND id<=?", (payer, pos, top)).fetchone()[0]
            total = carried + spent
            units = total // self.unit_micro if total > 0 else 0
            remainder = total - units * self.unit_micro
            if units == 0:  # less than one whole credit: nothing to send, keep the remainder and move on
                self._advance(db, payer, top, remainder)
                return None
            op_id = f"spend:{payer}:{pos}-{top}"
            db.execute("INSERT INTO report_attempts(op_id,payer,to_pos,units,remainder_after,status)"
                       " VALUES(?,?,?,?,?,'pending')", (op_id, payer, top, units, remainder))
        return op_id, units
