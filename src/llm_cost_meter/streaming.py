"""Credit enforcement while a model streams: deduct in batches, cancel when empty, true up at the end."""
import logging

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from .meter import Meter, MeterMode

log = logging.getLogger(__name__)


class InsufficientCredits(BaseException):
    """Raised before any tokens are generated. A refusal is final, so it derives from BaseException on
    purpose: LangChain fallback chains and retry loops catch `Exception`, and a model that was refused for
    lack of credit must not be re-tried on a fallback model. (Once, a refusal fell through to the fallback
    chain and was served anyway.)"""


class CreditsExhausted(Exception):
    """Cancel signal raised mid-stream. `ainvoke_metered` turns it into a truncated result; a plain
    `ainvoke` lets it propagate, with the text delivered so far in `partial_text`. It is an `Exception`
    because LangChain's own batch code mishandles any other kind raised mid-stream. A fallback model that
    starts after it has the same payer, so its pre-flight check is refused with `InsufficientCredits`."""

    def __init__(self, partial_text: str = "") -> None:
        super().__init__("credits exhausted mid-stream")
        self.partial_text = partial_text


def content_text(content) -> str:
    """Message content is a str or a list of blocks (str or {"type": "text", "text": ...})."""
    if isinstance(content, str):
        return content
    return "".join(b if isinstance(b, str) else str(b.get("text", "")) for b in content)


def estimate_tokens(chars: int) -> int:
    return max(chars // 4, 1)


class CreditEnforcer(AsyncCallbackHandler):
    """No reservation: pay as tokens arrive. Fallback chains and error callbacks make
    reserve-then-refund unreliable, so a signed true-up fixes any difference at the end.

    On cancel or provider error we settle what was actually streamed (see `on_llm_error`)."""

    def tap_output_aiter(self, run_id, output):
        """Duck-typing marker. LangChain streams `ainvoke` internally when a handler has these two methods
        (it checks for its private `_StreamingCallbackHandler` protocol), so enforcement and cut-off apply to
        plain `ainvoke` too. They pass the output through untouched. Tied to LangChain internals: a test pins it."""
        return output

    def tap_output_iter(self, run_id, output):
        return output

    raise_error = True  # LangChain swallows handler errors by default; we need them to cancel the stream

    def __init__(self, meter: Meter, service: str, model: str, batch_tokens: int = 64):
        self.meter, self.service, self.model, self.batch = meter, service, model, batch_tokens
        self._calls: dict = {}  # run_id -> per-call state; one handler serves concurrent calls

    async def _guard(self, fn, *args, **kw):
        """OFF and SHADOW must never break the user's call: log and carry on. ENFORCE raises."""
        try:
            return await fn(*args, **kw)
        except (InsufficientCredits, CreditsExhausted):
            raise
        except Exception:
            if self.meter.mode is MeterMode.ENFORCE:
                raise
            log.exception("metering failed in %s mode; call continues unmetered", self.meter.mode.value)

    def _deduct(self, run_id, key_suffix: str, units: int, micro: int) -> bool:
        if not self.meter.debit(f"{run_id}:{key_suffix}", self.service, units, micro):
            return False
        st = self._calls[run_id]
        st["micro"] += micro
        st["units"] += units
        return True

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        await self._guard(self._start, messages, run_id)

    async def _start(self, messages, run_id) -> None:
        chars = sum(len(content_text(m.content)) for batch in messages for m in batch)
        est = estimate_tokens(chars)
        self._calls[run_id] = {"micro": 0, "units": 0, "est_in": est, "out": 0, "pending": 0, "n": 0, "text": []}
        if not self._deduct(run_id, "in", est, self.meter.prices.cost_tokens(self.model, est, 0)):
            del self._calls[run_id]
            raise InsufficientCredits(f"cannot cover estimated input of ~{est} tokens")

    async def on_llm_new_token(self, token: str, *, run_id, **kwargs) -> None:
        await self._guard(self._token, token, run_id)

    async def _token(self, token: str, run_id) -> None:
        st = self._calls.get(run_id)
        if st is None or not token:
            return
        # Simplification: one chunk counts as one token; the true-up uses provider usage, so this only sets batching
        st["out"] += 1
        st["pending"] += 1
        st["text"].append(token)
        if st["pending"] >= self.batch:
            micro = self.meter.prices.cost_tokens(self.model, 0, st["pending"])
            st["n"] += 1
            if not self._deduct(run_id, f"out:{st['n']}", st["pending"], micro):
                st["out"] -= 1  # this chunk is dropped, never delivered, so never billed
                st["pending"] -= 1
                st["text"].pop()
                raise CreditsExhausted("".join(st["text"]))
            st["pending"] = 0

    async def on_llm_end(self, response: LLMResult, *, run_id, **kwargs) -> None:
        await self._guard(self._end, response, run_id)

    async def _end(self, response: LLMResult, run_id) -> None:
        st = self._calls.pop(run_id, None)
        if st is None:
            return
        msg = getattr(response.generations[0][0], "message", None)
        usage = getattr(msg, "usage_metadata", None) or {}
        if not usage:
            log.warning("provider reported no usage for %s; billing the streamed estimate", self.model)
        in_t = usage.get("input_tokens", st["est_in"])
        out_t = usage.get("output_tokens", st["out"])
        cached = (usage.get("input_token_details") or {}).get("cache_read", 0)
        actual = self.meter.prices.cost_tokens(self.model, in_t, out_t, cached)
        self.meter.trueup(f"{run_id}:trueup", self.service, in_t + out_t - st["units"], actual - st["micro"])

    async def on_llm_error(self, error, *, run_id, **kwargs) -> None:
        await self._guard(self._settle, run_id)

    async def _settle(self, run_id) -> None:
        """Cancel or provider failure. Bill what the user actually received: the streamed tokens not yet
        deducted (this may overdraw by less than one batch, like a normal true-up), or, if nothing was
        streamed, give the pre-flight estimate back. Input is not re-priced: no usage report exists."""
        st = self._calls.pop(run_id, None)
        if st is None:
            return
        if st["out"] == 0:
            self.meter.trueup(f"{run_id}:settle", self.service, -st["units"], -st["micro"])
        elif st["pending"]:
            micro = self.meter.prices.cost_tokens(self.model, 0, st["pending"])
            self.meter.trueup(f"{run_id}:settle", self.service, st["pending"], micro)
