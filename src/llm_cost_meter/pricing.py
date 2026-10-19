"""Price table. Rates are USD per million tokens, which equals micro-credits per token."""
import logging
from dataclasses import dataclass
from decimal import Decimal

from .units import MeterKind, to_units

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Rates:
    input_per_mtok: Decimal | None = None
    output_per_mtok: Decimal | None = None
    cached_input_per_mtok: Decimal | None = None
    per_page: Decimal | None = None  # USD per unit for the non-token axes
    per_audio_minute: Decimal | None = None
    per_embed_mtok: Decimal | None = None
    per_vision_unit: Decimal | None = None
    per_rerank: Decimal | None = None  # USD per search. Rerank is priced per search; it has no meter of its own.
    # Realtime voice: audio tokens cost far more than text tokens, so they get their own rates.
    audio_input_per_mtok: Decimal | None = None
    audio_output_per_mtok: Decimal | None = None
    cached_audio_input_per_mtok: Decimal | None = None


_UNIT_AXIS = {
    MeterKind.PAGE: "per_page",
    MeterKind.AUDIO_MINUTE: "per_audio_minute",
    MeterKind.EMBED_TOKEN: "per_embed_mtok",
    MeterKind.VISION_UNIT: "per_vision_unit",
}


class PriceTable:
    """In-memory model -> rates. Simplification: a dict; production loads a versioned table (see PriceHistory)."""

    def __init__(self, rates: dict[str, Rates]):
        self.rates = rates

    def has_cached_rate(self, model: str) -> bool:
        return self.rates[model].cached_input_per_mtok is not None

    def cost_tokens(self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> int:
        """`input_tokens` includes the cached ones (provider convention)."""
        r = self.rates[model]
        cached_rate = r.cached_input_per_mtok
        if cached_rate is None:
            if cached_tokens:
                log.warning("no cached rate for %s; billing cached tokens at the input rate", model)
            cached_rate = r.input_per_mtok
        total = (
            Decimal(max(input_tokens - cached_tokens, 0)) * (r.input_per_mtok or 0)
            + Decimal(cached_tokens) * (cached_rate or 0)
            + Decimal(output_tokens) * (r.output_per_mtok or 0)
        )
        return to_units(total)

    def cost_units(self, model: str, kind: MeterKind, units: Decimal | int | str) -> int:
        """Non-token work. Embeddings are priced per million tokens, the rest per unit."""
        if kind is MeterKind.TOKEN:
            raise ValueError("use cost_tokens for TOKEN")
        if isinstance(units, float):
            raise TypeError("units must be int, Decimal or str, never float")
        rate = getattr(self.rates[model], _UNIT_AXIS[kind])
        if rate is None:
            raise KeyError(f"{model} has no rate for {kind.value}")
        per = Decimal(1_000_000) if kind is MeterKind.EMBED_TOKEN else Decimal(1)
        return to_units(Decimal(str(units)) * rate * 1_000_000 / per)

    def cost_rerank(self, model: str, searches: int) -> int:
        """Rerank is billed per search, under no meter of its own (the ledger files it under EMBED_TOKEN)."""
        rate = self.rates[model].per_rerank
        if rate is None:
            raise KeyError(f"{model} has no rerank rate")
        return to_units(Decimal(searches) * rate * 1_000_000)

    def cost_realtime(self, model: str, *, text_in: int = 0, audio_in: int = 0, text_out: int = 0,
                      audio_out: int = 0, cached_text_in: int = 0, cached_audio_in: int = 0) -> int:
        """Realtime voice, priced per modality and cache tier. `text_in` and `audio_in` include their cached
        tokens. Billing every token at the audio rate would overcharge several times over."""
        r = self.rates[model]
        for name, rate in (("audio_input_per_mtok", r.audio_input_per_mtok),
                           ("audio_output_per_mtok", r.audio_output_per_mtok)):
            if rate is None:
                raise KeyError(f"{model} has no {name}")
        text_cached = r.cached_input_per_mtok if r.cached_input_per_mtok is not None else r.input_per_mtok
        audio_cached = (r.cached_audio_input_per_mtok if r.cached_audio_input_per_mtok is not None
                        else r.audio_input_per_mtok)
        total = (
            Decimal(max(text_in - cached_text_in, 0)) * (r.input_per_mtok or 0)
            + Decimal(cached_text_in) * (text_cached or 0)
            + Decimal(max(audio_in - cached_audio_in, 0)) * r.audio_input_per_mtok
            + Decimal(cached_audio_in) * (audio_cached or 0)
            + Decimal(text_out) * (r.output_per_mtok or 0)
            + Decimal(audio_out) * r.audio_output_per_mtok
        )
        return to_units(total)
