"""Money units. One credit = one USD; money is always an integer count of micro-credits."""
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum

MICRO = 1_000_000


class MeterKind(StrEnum):
    """Separate meters. Units of different kinds are never converted into each other."""

    TOKEN = "token"
    PAGE = "page"
    AUDIO_MINUTE = "audio_minute"
    EMBED_TOKEN = "embed_token"
    VISION_UNIT = "vision_unit"


def to_units(amount: Decimal) -> int:
    """Round once, half up, at the very end of a calculation (floats never reach money)."""
    return int(amount.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def usd_to_units(usd: Decimal | str | int) -> int:
    return to_units(Decimal(usd) * MICRO)
