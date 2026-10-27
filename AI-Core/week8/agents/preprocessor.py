"""Rewrites messy product text into a short, standard format before it is priced.

The pricers were trained or tuned on product text summarised in this format (Week 6), so
deals scraped from the web are rewritten the same way first. Any LiteLLM model can be
used; the default is Llama 3.2 running locally in Ollama. Set PRICER_PREPROCESSOR_MODEL
in .env to use another, e.g. "groq/openai/gpt-oss-20b".
"""

from litellm import completion
from dotenv import load_dotenv
import os

load_dotenv(override=True)

DEFAULT_MODEL_NAME = os.getenv("PRICER_PREPROCESSOR_MODEL", "ollama/llama3.2")
# gpt-oss models think before answering; keep that short. Other models don't take the setting.
DEFAULT_REASONING_EFFORT = "low" if "gpt-oss" in DEFAULT_MODEL_NAME else None

SYSTEM_PROMPT = """Create a concise description of a product. Respond only in this format. Do not include part numbers.
Title: Rewritten short precise title
Category: eg Electronics
Brand: Brand name
Description: 1 sentence description
Details: 1 sentence on features"""


class Preprocessor:
    """Calls an LLM to rewrite product text, keeping a running total of tokens and cost."""

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        reasoning_effort=DEFAULT_REASONING_EFFORT,
        base_url=None,
    ):
        """
        :param model_name: a LiteLLM model name, e.g. "ollama/llama3.2"
        :param reasoning_effort: passed to models that think, otherwise None
        :param base_url: server address; Ollama models default to http://localhost:11434
        """
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.model_name = model_name
        self.reasoning_effort = reasoning_effort
        self.base_url = base_url
        if "ollama" in model_name and not base_url:
            self.base_url = "http://localhost:11434"

    def messages_for(self, text: str) -> list[dict]:
        """The system prompt describing the format, then the product text."""
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]

    def preprocess(self, text: str) -> str:
        """Return the rewritten text, adding this call's tokens and cost to the totals."""
        messages = self.messages_for(text)
        response = completion(
            messages=messages,
            model=self.model_name,
            reasoning_effort=self.reasoning_effort,
            api_base=self.base_url,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        self.total_cost += response._hidden_params["response_cost"]
        return response.choices[0].message.content
