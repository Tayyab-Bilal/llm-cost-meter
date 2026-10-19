"""The single choke point: every LLM is built here, so enforcement cannot be forgotten."""
from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models.chat_models import BaseChatModel

from .buckets import Bucket, bucket_for
from .meter import Meter
from .streaming import CreditEnforcer, CreditsExhausted, content_text


def metered_chat_model(service: str, model: str, *, llm_factory: Callable[[str], BaseChatModel],
                    meter: Meter, batch_tokens: int = 64) -> BaseChatModel:
    """`llm_factory` is injected so this module never imports a provider SDK."""
    if bucket_for(service) is Bucket.SEPARATELY_METERED:  # also validates the label
        raise ValueError(f"{service!r} is billed per unit; use Meter.charge_units, not a chat model")
    llm = llm_factory(model)
    enforcer = CreditEnforcer(meter, service, model, batch_tokens)
    if isinstance(llm.callbacks, BaseCallbackManager):
        llm.callbacks.add_handler(enforcer)
    else:
        llm.callbacks = [*(llm.callbacks or []), enforcer]  # keep the caller's own handlers
    return llm


@dataclass
class MeteredResult:
    text: str
    truncated: bool
    usage: dict


async def ainvoke_metered(llm: BaseChatModel, messages) -> MeteredResult:
    """Stream so the enforcer can cancel mid-answer. Running out of credit returns the partial text
    with `truncated=True`; only a pre-flight failure raises (`InsufficientCredits`)."""
    parts: list[str] = []
    usage: dict = {}
    truncated = False
    try:
        async for chunk in llm.astream(messages):
            parts.append(content_text(chunk.content))
            usage = chunk.usage_metadata or usage
    except CreditsExhausted:
        truncated = True
    return MeteredResult("".join(parts), truncated, dict(usage))
