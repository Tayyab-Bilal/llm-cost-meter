"""A feature that bypasses billing. The AST guard must reject it."""
from anthropic import Anthropic

from llm_cost_meter import metered_chat_model

client = Anthropic()  # direct provider client: nothing meters this


def build_summary_llm(meter, factory):
    # a new label nobody registered in SERVICE_BUCKETS: this is how 151 calls once went unbilled
    return metered_chat_model("meeting_digest", "acme-chat", llm_factory=factory, meter=meter)
