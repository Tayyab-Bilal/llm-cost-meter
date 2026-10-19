"""Build-time check: the price table must not drift from a pinned, reviewed snapshot.

A silent rate change makes every downstream number wrong in the same way, so nothing else would notice.
Pin the reviewed rates in a JSON file (`snapshot()` writes one); CI fails when the live table differs, when a
priced model is not pinned, or when a model with an input rate lacks an output rate or a cached rate.
"""
from dataclasses import asdict

from ..pricing import PriceTable


def snapshot(prices: PriceTable) -> dict[str, dict[str, str]]:
    return {m: {k: str(v) for k, v in asdict(r).items() if v is not None} for m, r in sorted(prices.rates.items())}


def check_rate_drift(prices: PriceTable, pinned: dict[str, dict[str, str]], *,
                     no_cache_ok: frozenset[str] = frozenset()) -> list[str]:
    """Returns one problem string per drift. `no_cache_ok` lists models reviewed as having no cached tier."""
    live, problems = snapshot(prices), []
    for model in sorted(pinned.keys() - live.keys()):
        problems.append(f"{model}: pinned but missing from the price table")
    for model in sorted(live.keys() - pinned.keys()):
        problems.append(f"{model}: priced but not pinned; review it and add it to the snapshot")
    for model in sorted(live.keys() & pinned.keys()):
        for axis in sorted(live[model].keys() | pinned[model].keys()):
            was, now = pinned[model].get(axis), live[model].get(axis)
            if was != now:
                problems.append(f"{model}.{axis}: pinned {was}, live {now}")
    for model, rates in live.items():
        if "input_per_mtok" in rates and "output_per_mtok" not in rates:
            problems.append(f"{model}: has an input rate but no output rate")
        if "input_per_mtok" in rates and "cached_input_per_mtok" not in rates and model not in no_cache_ok:
            problems.append(f"{model}: no cached input rate (cached tokens would bill at the full input rate)")
    return problems
