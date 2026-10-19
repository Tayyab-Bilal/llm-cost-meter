import pytest

from llm_cost_meter import (
    FakeBillingBackend,
    Ledger,
    MissingIdentity,
    PayerResolver,
    PayerUnavailable,
    turn,
)


class Clock:
    now = 0.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def parts():
    backend, clock = FakeBillingBackend(), Clock()
    backend.register_org("acme", "org-acme", credits=100)
    return backend, clock, PayerResolver(Ledger(), backend.payer_for, ttl=60, clock=clock)


def test_all_users_of_a_tenant_resolve_to_one_shared_organisation_payer(parts, make_meter):
    backend, _, resolver = parts
    payers = {resolver.resolve("acme", f"user-{i}") for i in range(25)}
    assert payers == {"org-acme"}
    # one shared balance, not 25: the credit pool is keyed by payer, never by user
    meter = make_meter(balance=0)
    meter.topup("org-acme", 1_000_000, key="seed")
    assert meter.balances.get("org-acme") == 1_000_000
    assert all(meter.balances.get(f"user-{i}") is None for i in range(25))


def test_cache_hit_skips_the_map_and_the_authority(parts):
    backend, clock, resolver = parts
    resolver.resolve("acme", "u1")
    backend.available = False
    assert resolver.resolve("acme", "u1") == "org-acme"
    assert backend.authority_calls == 1


def test_expired_cache_falls_back_to_the_durable_map_not_the_authority(parts):
    backend, clock, resolver = parts
    resolver.resolve("acme", "u1")
    clock.now = 61  # TTL passed
    backend.available = False
    assert resolver.resolve("acme", "u1") == "org-acme"  # served from the durable map
    assert backend.authority_calls == 1


def test_fails_closed_when_authority_is_down_and_nothing_is_cached(parts):
    backend, _, resolver = parts
    backend.available = False
    with pytest.raises(PayerUnavailable):
        resolver.resolve("acme", "new-user")


def test_unknown_tenant_fails_closed(parts):
    _, _, resolver = parts
    with pytest.raises(PayerUnavailable):
        resolver.resolve("nobody", "u1")


@pytest.mark.parametrize("tenant", ["", "system"])
def test_missing_or_system_tenant_is_refused_before_any_lookup(parts, tenant):
    backend, _, resolver = parts
    with pytest.raises(MissingIdentity):
        resolver.resolve(tenant, "u1")
    assert backend.authority_calls == 0


def test_turn_requires_a_real_tenant_and_payer():
    """Regression: spend once landed on a catch-all "system" tenant because nothing refused a missing one."""
    for kwargs in ({"tenant": "system", "payer": "p"}, {"tenant": "", "payer": "p"}, {"tenant": "t", "payer": ""}):
        with pytest.raises(MissingIdentity), turn(entry_service="chat", **kwargs):
            pass
