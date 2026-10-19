from openai import OpenAI

from llm_cost_meter import metered_chat_model

client = OpenAI()
llm = metered_chat_model("shiny_new_feature", "m", llm_factory=None, meter=None)
