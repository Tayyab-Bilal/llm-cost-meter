"""Every service label must map to a billing bucket, so an unregistered label cannot be unbilled by accident."""
from enum import StrEnum


class Bucket(StrEnum):
    CHARGED = "charged"
    CHARGED_TO_ENTRY_POINT = "charged_to_entry_point"  # sub-agents: billed under the user action
    RECORDED_ZERO = "recorded_zero"  # tracked for cost reports, deliberately free to the user
    SEPARATELY_METERED = "separately_metered"  # pages, minutes, rerank: billed via charge_units


class UnregisteredBillingLabel(KeyError):
    pass


SERVICE_BUCKETS: dict[str, Bucket] = {
    "chat": Bucket.CHARGED,
    "planner": Bucket.CHARGED_TO_ENTRY_POINT,
    "summarizer": Bucket.CHARGED_TO_ENTRY_POINT,
    "health_probe": Bucket.RECORDED_ZERO,
    "doc_parse": Bucket.SEPARATELY_METERED,
    "transcribe": Bucket.SEPARATELY_METERED,
    "embed": Bucket.SEPARATELY_METERED,
    "rerank": Bucket.SEPARATELY_METERED,
}


def bucket_for(label: str) -> Bucket:
    try:
        return SERVICE_BUCKETS[label]
    except KeyError:
        raise UnregisteredBillingLabel(f"service label {label!r} has no billing bucket") from None
