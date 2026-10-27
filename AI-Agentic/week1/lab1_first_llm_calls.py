"""Lab 1 (Week 1, Day 2): first calls to an LLM with the OpenAI API.

Steps:
  1. Load the API keys from .env and check the OpenAI key is there.
  2. Ask a very cheap model for a fun fact - the simplest possible LLM call.
  3. Chain three calls: one model writes a hard IQ-style question, a second call
     answers it, and a stronger model judges whether the answer is right. Feeding
     one LLM's output into the next is the first step towards agentic AI.
  4. Exercise: find an Agentic AI business opportunity with three chained calls -
     pick a business area, find a pain point in it, propose an agentic solution.

Setup: OPENAI_API_KEY in the .env file in the project root (load_dotenv looks in the
current folder and then each parent folder). To use another provider (Gemini,
DeepSeek, Ollama...), see guide 9 in the course repo; only the OpenAI() line changes.

Run: python lab1_first_llm_calls.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

# ---- Settings ---------------------------------------------------------------------

FAST_MODEL = "gpt-5.4-nano"             # very cheap; fine for simple requests
MID_MODEL = "gpt-5.4-mini"              # still cheap, but better at reasoning
JUDGE_MODEL = "gpt-5.4"                 # the strongest of the three, used to mark answers


# ---- Helpers ----------------------------------------------------------------------

def heading(title):
    """Print a section title, so each step's output is easy to find."""
    print(f"\n{'=' * 10} {title} {'=' * 10}\n")


def ask(openai, model, prompt):
    """Send prompt to model as a single user message and return the reply text.

    Messages use the OpenAI chat format: a list of {"role": ..., "content": ...}
    dicts. Here there is just one, from the user.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


# ---- Steps ------------------------------------------------------------------------

def check_openai_key():
    """Load .env into environment variables and report whether OPENAI_API_KEY is set.

    If no .env is found, the usual cause is that it wasn't saved after adding the key,
    or isn't named exactly ".env", or isn't in the project root.
    """
    heading("Checking the API key")
    if not load_dotenv(override=True):
        print("No .env file found - is it saved, named exactly .env, and in the project root?")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        # Show only the start of the key - never print a whole key
        print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        print("OpenAI API Key not set - please head to the troubleshooting guide in the setup folder")


def fun_fact(openai):
    """The simplest possible call: ask the cheapest model for a fun fact."""
    heading(f"A fun fact from {FAST_MODEL}")
    print(ask(openai, FAST_MODEL, "Tell me a fun fact"))


def question_answer_judge(openai):
    """Chain three LLM calls: write a hard question, answer it, then judge the answer."""
    heading(f"{MID_MODEL} writes a hard question")
    question = ask(
        openai,
        MID_MODEL,
        "Please propose a hard, challenging question to assess someone's IQ. Respond only with the question.",
    )
    print(question)

    heading(f"{MID_MODEL} answers it")
    answer = ask(openai, MID_MODEL, question)
    print(answer)

    heading(f"{JUDGE_MODEL} judges the answer")
    # Both the question and the answer go to the judge in one prompt
    message = f"""
Here is a question:
{question}

And here is a possible answer that might be correct or incorrect:
{answer}

Please evaluate if the answer is correct or incorrect.
"""
    print(ask(openai, JUDGE_MODEL, message))


def exercise_business_opportunity(openai):
    """Exercise: three chained calls, each building on the previous reply.

    1. Pick a business area worth exploring for Agentic AI.
    2. Find a pain point in that area that might be ripe for an agentic solution.
    3. Propose the Agentic AI solution.
    (The notebook leaves this as a template for you to fill in; this is one way to do it.)
    """
    heading("Exercise: 1. a business area")
    business_area = ask(
        openai,
        MID_MODEL,
        "Pick a business area that might be worth exploring for an Agentic AI opportunity. "
        "Respond only with the business area and one sentence on why.",
    )
    print(business_area)

    heading("Exercise: 2. a pain point")
    pain_point = ask(
        openai,
        MID_MODEL,
        f"Here is a business area:\n{business_area}\n\n"
        "Present one pain point in this industry - something challenging that might be ripe "
        "for an Agentic AI solution. Respond in a short paragraph.",
    )
    print(pain_point)

    heading("Exercise: 3. an agentic solution")
    print(ask(
        openai,
        MID_MODEL,
        f"Business area:\n{business_area}\n\nPain point:\n{pain_point}\n\n"
        "Propose an Agentic AI solution to this pain point: what the agents do, "
        "which tools they use, and where a human stays in the loop.",
    ))


# ---- Main -------------------------------------------------------------------------

def main():
    """Run the lab's steps in order."""
    check_openai_key()
    openai = OpenAI()                   # reads OPENAI_API_KEY from the environment
    fun_fact(openai)
    question_answer_judge(openai)
    exercise_business_opportunity(openai)


if __name__ == "__main__":
    main()
