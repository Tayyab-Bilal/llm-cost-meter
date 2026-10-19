"""Deterministic fake chat model for tests and examples. Never talks to a network."""
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from .streaming import content_text


class FakeChat(GenericFakeChatModel):
    """Streams the next scripted reply one word per chunk and ends with a usage chunk, like a provider."""

    cached_tokens: int = 0
    fail_after: int | None = None  # raise a provider error after this many tokens
    emit_usage: bool = True
    reasoning_tokens: int = 0  # hidden thinking tokens: counted inside output_tokens, like providers do

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs: Any):
        reply = next(self.messages)
        text = reply.content if isinstance(reply, AIMessage) else reply
        words = text.split()
        for i, word in enumerate(words):
            if self.fail_after is not None and i >= self.fail_after:
                raise RuntimeError("provider error")
            tok = word + (" " if i < len(words) - 1 else "")
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=tok))
            if run_manager:
                await run_manager.on_llm_new_token(tok, chunk=chunk)
            yield chunk
        if not self.emit_usage:
            return
        in_tokens = max(sum(len(content_text(m.content)) for m in messages) // 4, 1)
        out_tokens = len(words) + self.reasoning_tokens
        usage = {"input_tokens": in_tokens, "output_tokens": out_tokens,
                 "total_tokens": in_tokens + out_tokens,
                 "input_token_details": {"cache_read": min(self.cached_tokens, in_tokens)},
                 "output_token_details": {"reasoning": self.reasoning_tokens}}
        yield ChatGenerationChunk(message=AIMessageChunk(content="", usage_metadata=usage))


def fake_llm_factory(*replies: str, **fields):
    """Returns an `llm_factory` for `metered_chat_model` that replays the scripted replies in order."""
    it = iter(replies)
    return lambda model: FakeChat(messages=it, **fields)
