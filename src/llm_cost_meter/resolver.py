"""Who pays for this user? Three tiers: TTL cache, then a durable map table, then the backend as authority.

Fail closed: if no tier can answer, raise. Never fall back to a default or "system" payer: that is how
spend once landed on the wrong customer. Many users can resolve to one shared organisation payer; the
balance is keyed by payer, never by user, so a 25-user tenant does not get 25 balances.
"""
import time
from collections.abc import Callable

from .context import UNATTRIBUTED_TENANTS, MissingIdentity
from .ledger import Ledger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS payer_map(
  tenant TEXT NOT NULL, user_id TEXT NOT NULL, payer TEXT NOT NULL,
  PRIMARY KEY (tenant, user_id))
"""


class PayerUnavailable(RuntimeError):
    """The payer could not be resolved and nothing was cached: refuse the work rather than guess."""


class PayerResolver:
    def __init__(self, ledger: Ledger, authority: Callable[[str, str], str], *, ttl: float = 300.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._ledger, self._authority, self._ttl, self._clock = ledger, authority, ttl, clock
        self._cache: dict[tuple[str, str], tuple[str, float]] = {}  # Simplification: per process; Redis in production
        with ledger.transaction() as db:
            db.execute(_SCHEMA)

    def resolve(self, tenant: str, user_id: str) -> str:
        if tenant in UNATTRIBUTED_TENANTS or not user_id:
            raise MissingIdentity(f"cannot resolve a payer for tenant={tenant!r} user={user_id!r}")
        key = (tenant, user_id)
        hit = self._cache.get(key)
        if hit and hit[1] > self._clock():
            return hit[0]
        with self._ledger.transaction() as db:
            row = db.execute("SELECT payer FROM payer_map WHERE tenant=? AND user_id=?", key).fetchone()
        if row:
            return self._remember(key, row[0])
        try:
            payer = self._authority(tenant, user_id)
        except Exception as exc:
            raise PayerUnavailable(f"no payer for {tenant}/{user_id}: authority failed ({exc!r})") from exc
        with self._ledger.transaction() as db:
            db.execute("INSERT OR REPLACE INTO payer_map(tenant,user_id,payer) VALUES(?,?,?)", (*key, payer))
        return self._remember(key, payer)

    def _remember(self, key: tuple[str, str], payer: str) -> str:
        self._cache[key] = (payer, self._clock() + self._ttl)
        return payer
