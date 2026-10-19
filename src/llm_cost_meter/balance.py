"""Live balance cache. The backend billing system owns the purchased balance; this is the hot path."""
import threading
from typing import Protocol

# Refuses a missing key (never invents a balance) and decrements-then-checks atomically.
_DECREMENT_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then return {0, 0} end
local amt = tonumber(ARGV[1])
if not amt or amt <= 0 then return {0, tonumber(redis.call('GET', KEYS[1]))} end
local rem = redis.call('DECRBY', KEYS[1], amt)
if rem < 0 then
  redis.call('INCRBY', KEYS[1], amt)
  return {0, rem + amt}
end
return {1, rem}
"""

# Adjusting a missing key must not mint a balance (a negative true-up would create free credit).
_ADJUST_LUA = """
if redis.call('EXISTS', KEYS[1]) == 0 then return 0 end
if ARGV[2] ~= '' then
  -- a retry after an ambiguous failure (timeout after commit) must not apply twice
  if redis.call('SISMEMBER', KEYS[2], ARGV[2]) == 1 then return 1 end
  redis.call('SADD', KEYS[2], ARGV[2])
  redis.call('EXPIRE', KEYS[2], 604800)
end
redis.call('INCRBY', KEYS[1], tonumber(ARGV[1]))
return 1
"""


class BalanceStore(Protocol):
    def get(self, payer: str) -> int | None: ...
    def set(self, payer: str, micro: int) -> None: ...  # seeding/rebuild only; purchases go via Meter.topup
    def decrement_if_sufficient(self, payer: str, micro: int) -> tuple[bool, int]: ...
    # adjust returns False when there is no such balance (nothing changed); `key` makes a retry idempotent
    def adjust(self, payer: str, delta: int, key: str | None = None) -> bool: ...


class InMemoryBalanceStore:
    def __init__(self) -> None:
        self._b: dict[str, int] = {}
        self._seen: dict[str, set[str]] = {}
        self._lock = threading.Lock()

    def get(self, payer: str) -> int | None:
        return self._b.get(payer)

    def set(self, payer: str, micro: int) -> None:
        """Seeding and rebuild only."""
        with self._lock:
            self._b[payer] = micro

    def decrement_if_sufficient(self, payer: str, micro: int) -> tuple[bool, int]:
        with self._lock:
            cur = self._b.get(payer)
            if micro <= 0 or cur is None or cur < micro:
                return False, cur or 0
            self._b[payer] = cur - micro
            return True, cur - micro

    def adjust(self, payer: str, delta: int, key: str | None = None) -> bool:
        with self._lock:
            if payer not in self._b:
                return False
            if key is not None:
                if key in self._seen.setdefault(payer, set()):
                    return True
                self._seen[payer].add(key)
            self._b[payer] += delta
            return True


class RedisBalanceStore:
    """Keys are `credits:{payer}`; the braces are a hash tag so one payer's keys share a Cluster slot."""

    def __init__(self, client) -> None:
        self._r = client
        self._decrement = client.register_script(_DECREMENT_LUA)
        self._adjust = client.register_script(_ADJUST_LUA)

    @staticmethod
    def _key(payer: str) -> str:
        return f"credits:{{{payer}}}"

    def get(self, payer: str) -> int | None:
        v = self._r.get(self._key(payer))
        return None if v is None else int(v)

    def set(self, payer: str, micro: int) -> None:
        self._r.set(self._key(payer), micro)

    def decrement_if_sufficient(self, payer: str, micro: int) -> tuple[bool, int]:
        ok, remaining = self._decrement(keys=[self._key(payer)], args=[micro])
        return bool(ok), int(remaining)

    def adjust(self, payer: str, delta: int, key: str | None = None) -> bool:
        return bool(self._adjust(keys=[self._key(payer), self._key(payer) + ":applied"], args=[delta, key or ""]))
