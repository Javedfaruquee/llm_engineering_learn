"""Lab 2 (Week 1, Day 3): many models compete on one question, and an LLM judges them.

Steps:
  1. Check which providers' API keys are in .env (only OpenAI is required).
  2. Have gpt-5.4-mini write a thought-provoking question with a short answer.
  3. Put that question to models from many providers: OpenAI, Anthropic, Google,
     DeepSeek, Groq, OpenRouter and local Ollama models. They all offer
     OpenAI-compatible endpoints, so the same OpenAI client works for all of them -
     only the base_url and the key change.
  4. Show all the answers to Grok as an "LLM as a judge", which ranks them.

This combines two agentic design patterns: sending one task to several models in
parallel, and having another model evaluate the results.

Setup:
  - OPENAI_API_KEY in .env; any of ANTHROPIC_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY,
    GROQ_API_KEY, GROK_API_KEY, OPENROUTER_API_KEY are optional. A competitor whose
    provider has no key is skipped. Without GROK_API_KEY, FALLBACK_JUDGE judges instead.
  - Ollama (optional, free, local): install from https://ollama.com, then
    `ollama pull llama3.2:1b` (and any other models in COMPETITORS). Only pull models of
    3GB or less unless you know your machine can take more, and avoid ":cloud" models.
    Useful commands: `ollama ls` lists your models, `ollama rm <model>` deletes one.
    Ollama models you haven't pulled are skipped.

Run: python lab2_multi_model_competition_and_judge.py
"""

import json
import os

import requests
from dotenv import load_dotenv
from openai import OpenAI

# ---- Settings ---------------------------------------------------------------------

# Each provider: (OpenAI-compatible base URL, env var holding its key, how many
# characters of the key to show when checking it). OpenAI needs no base URL; Ollama
# runs locally and needs no key.
PROVIDERS = {
    "openai": (None, "OPENAI_API_KEY", 8),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY", 7),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY", 2),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY", 3),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", 4),
    "grok": ("https://api.x.ai/v1", "GROK_API_KEY", 4),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", 6),
    "ollama": ("http://localhost:11434/v1", None, 0),
}
OLLAMA_URL = "http://localhost:11434"

QUESTION_SETTER = ("openai", "gpt-5.4-mini")

# The competitors, as (provider, model). Remove any you don't want to pay for.
COMPETITORS = [
    ("openai", "gpt-5.4-nano"),
    ("anthropic", "claude-sonnet-4-6"),
    ("gemini", "gemini-3.1-flash-lite"),
    ("deepseek", "deepseek-v4-flash"),
    ("groq", "openai/gpt-oss-120b"),           # OpenAI's open-weights model, served fast by Groq
    ("openrouter", "moonshotai/kimi-k2.6"),
    ("ollama", "llama3.2:1b"),
    ("ollama", "gpt-oss:latest"),
    ("ollama", "gemma4:latest"),
]

# Extra request options for particular models.
# reasoning_effort can be none, low, medium, high or xhigh; "none" = answer without thinking.
REQUEST_OPTIONS = {
    "gpt-5.4-nano": {"reasoning_effort": "none"},
}

# Grok is branded "the most truth-seeking large language model in the world", so it judges
JUDGE = ("grok", "grok-4.3")
FALLBACK_JUDGE = ("openai", "gpt-5.4-mini")     # used if there is no Grok key


# ---- Helpers ----------------------------------------------------------------------

def heading(title):
    """Print a section title, so each step's output is easy to find."""
    print(f"\n{'=' * 10} {title} {'=' * 10}\n")


def ask(client, model, prompt):
    """Send prompt to model as a single user message and return the reply text."""
    messages = [{"role": "user", "content": prompt}]
    options = REQUEST_OPTIONS.get(model, {})
    response = client.chat.completions.create(model=model, messages=messages, **options)
    return response.choices[0].message.content


# ---- Steps ------------------------------------------------------------------------

def check_keys():
    """Print which provider keys are set, showing only the first few characters of each."""
    heading("API keys")
    for provider, (_, key_name, prefix_length) in PROVIDERS.items():
        if key_name is None:
            continue                    # Ollama: no key needed
        key = os.getenv(key_name)
        optional = "" if provider == "openai" else " (and this is optional)"
        if key:
            print(f"{provider}: key exists and begins {key[:prefix_length]}")
        else:
            print(f"{provider}: key not set{optional}")


def make_clients():
    """One OpenAI client per provider that has a key (plus Ollama, which needs none).

    Every client is the same OpenAI class; base_url points it at the other provider.
    """
    clients = {}
    for provider, (base_url, key_name, _) in PROVIDERS.items():
        if key_name is None:
            clients[provider] = OpenAI(base_url=base_url, api_key="ollama")  # any non-empty key works
        elif os.getenv(key_name):
            clients[provider] = OpenAI(api_key=os.getenv(key_name), base_url=base_url)
    return clients


def local_ollama_models():
    """The names of the models pulled into Ollama, or an empty list if Ollama isn't running."""
    heading("Ollama")
    try:
        models = requests.get(f"{OLLAMA_URL}/v1/models", timeout=3).json()
    except requests.RequestException:
        print(f"Ollama is not running at {OLLAMA_URL} - skipping local models (start it with `ollama serve`)")
        return []
    names = [model.get("id") for model in models.get("data", [])]
    print("Ollama is running. Local models:", ", ".join(names) or "none")
    return names


def make_question(clients):
    """Have the question setter write a challenging question with a short answer."""
    provider, model = QUESTION_SETTER
    heading(f"{model} writes the question")
    request = """
Please come up with a challenging, nuanced question with a succinct answer,
that I can ask a number of LLMs to evaluate their intelligence.
Not a mathematical puzzle, but more of a thought-provoking question that requires intelligent insight.
Include in your question that the answer must be short.
"""
    request += "Answer only with the question, no explanation."
    question = ask(clients[provider], model, request)
    print(question)
    return question


def run_competition(clients, question, ollama_models):
    """Put the question to every competitor that can be reached.

    Skips competitors whose provider has no key, Ollama models that aren't pulled,
    and any call that fails (printing why), so one problem doesn't stop the lab.
    Returns (competitor model names, their answers), in the same order.
    """
    competitors, answers = [], []
    for provider, model in COMPETITORS:
        if provider not in clients:
            print(f"Skipping {model}: no {provider} key")
            continue
        if provider == "ollama" and model not in ollama_models:
            print(f"Skipping {model}: not pulled into Ollama (run `ollama pull {model}`)")
            continue
        heading(f"{model} ({provider})")
        try:
            answer = ask(clients[provider], model, question)
        except Exception as error:
            print(f"Skipping {model}: {error}")
            continue
        print(answer)
        competitors.append(model)
        answers.append(answer)
    return competitors, answers


def make_judge_prompt(question, competitors, answers):
    """Build the judge's prompt: the question, then every answer numbered from 1.

    The answers are anonymous - only numbered - so the judge can't favour a brand.
    The judge must reply with JSON listing the competitor numbers, best first.
    """
    together = ""
    for index, answer in enumerate(answers):
        together += f"# Response from competitor {index + 1}\n\n"
        together += answer + "\n\n"

    # Double braces {{ }} are literal braces inside the f-string
    return f"""You are judging a competition between {len(competitors)} competitors.
Each model has been given this question:

{question}

Your job is to evaluate each response for clarity and strength of argument, and rank them in order of best to worst.
Respond with JSON, and only JSON, with the following format:
{{"results": ["best competitor number", "second best competitor number", "third best competitor number", ...]}}

Here are the responses from each competitor:

{together}

Now respond with the JSON with the ranked order of the competitors, nothing else. Do not include markdown formatting or code blocks."""


def judge_competition(clients, judge_prompt):
    """Ask the judge (or the fallback judge, if there's no Grok key) to rank the answers.

    Returns the ranked list of competitor numbers (as strings, "1" = first competitor).
    """
    provider, model = JUDGE if JUDGE[0] in clients else FALLBACK_JUDGE
    heading(f"Judgement time! {model} ranks the answers")
    results = ask(clients[provider], model, judge_prompt)
    print(results)
    return json.loads(results)["results"]


def print_ranking(competitors, ranks):
    """Turn the judge's competitor numbers back into model names, best first."""
    heading("Results")
    for index, result in enumerate(ranks):
        competitor = competitors[int(result) - 1]
        print(f"Rank {index + 1}: {competitor}")


# ---- Main -------------------------------------------------------------------------

def main():
    """Run the lab's steps in order."""
    load_dotenv(override=True)
    check_keys()
    clients = make_clients()
    ollama_models = local_ollama_models()

    question = make_question(clients)
    competitors, answers = run_competition(clients, question, ollama_models)

    judge_prompt = make_judge_prompt(question, competitors, answers)
    heading("The judge's prompt")
    print(judge_prompt)
    ranks = judge_competition(clients, judge_prompt)
    print_ranking(competitors, ranks)


if __name__ == "__main__":
    main()
