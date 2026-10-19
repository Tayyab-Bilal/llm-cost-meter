from llm_cost_meter import metered_chat_model

llm = metered_chat_model("chat", "m", llm_factory=None, meter=None)
