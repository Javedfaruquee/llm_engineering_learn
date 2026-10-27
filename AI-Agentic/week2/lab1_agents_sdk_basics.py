"""Lab 1 (Week 2, Day 1): first look at the OpenAI Agents SDK.

The SDK is very lightweight: an Agent is a name, instructions and a model, and the
Runner runs the agent loop for you (calling the model, running tools, repeating).

Steps:
  Part 1 - A simple agent: tell a joke, look at the raw messages, wrap the run in a
           trace (see it at https://platform.openai.com/traces), and stream the reply
           word by word.
  Part 2 - A tool: turn a plain Python function into a tool with @function_tool -
           the SDK writes the JSON description from the type hints and docstring -
           and give it to an agent that sends you a push notification.
  Part 3 - Memory: each Runner.run() starts fresh, so the agent forgets your name.
           Fix it by passing the conversation back in yourself, or with a session
           (SQLiteSession) that stores the conversation for you.

Setup:
  - OPENAI_API_KEY in .env.
  - PUSHOVER_USER and PUSHOVER_TOKEN in .env for Part 2 (see Week 1, Lab 4).
  - The SDK's package is `openai-agents` (pip install openai-agents), imported as
    `agents`. Careful: `pip install agents` installs an unrelated older library.
  - Docs: https://openai.github.io/openai-agents-python/

Run: python lab1_agents_sdk_basics.py
"""

import asyncio
import os

import requests
from agents import Agent, Runner, SQLiteSession, function_tool, trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from messenger import push

load_dotenv(override=True)

# ---- Settings ---------------------------------------------------------------------

MODEL_NAME = "gpt-5.4-mini"

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


# ---- Helpers ----------------------------------------------------------------------

def heading(title):
    """Print a section title, so each step's output is easy to find."""
    print(f"\n{'=' * 10} {title} {'=' * 10}\n")


async def ask(agent, prompt, **options):
    """Run agent on prompt (a string, or a list of messages), print and return the result.

    options are passed on to Runner.run, e.g. session=...
    """
    result = await Runner.run(agent, prompt, **options)
    print(result.final_output)
    return result


# ---- Part 1: a simple agent ---------------------------------------------------------

async def simple_agent():
    """Make a joke-telling agent; run it plain, inside a trace, and streamed."""
    agent = Agent(name="Jokester", instructions="You are a joke teller", model=MODEL_NAME)

    heading("A joke")
    result = await ask(agent, "Tell a joke about Autonomous AI Agents")

    heading("The messages behind it")
    # The full conversation the run produced: our input plus the model's reply
    print(result.to_input_list())

    heading("A joke, inside a trace")
    # A trace groups everything in the block into one entry on the OpenAI traces page
    with trace("Telling a joke"):
        await ask(agent, "Tell a joke about Autonomous AI Agents")
    print("\nNow look at the trace: https://platform.openai.com/traces")

    heading("Streaming 5 jokes")
    result = Runner.run_streamed(agent, input="Please tell me 5 jokes about AI Agents.")
    async for event in result.stream_events():
        # Print each piece of text as it arrives; skip the other event types
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()


# ---- Part 2: a tool ---------------------------------------------------------------

def check_pushover_keys():
    """Report whether the Pushover keys are set and look right (user starts u, token a)."""
    if pushover_user:
        if pushover_user.startswith("u"):
            print("Pushover user found and looks good")
        else:
            print("Pushover user found but doesn't start with u")
    else:
        print("Pushover user not found")

    if pushover_token:
        if pushover_token.startswith("a"):
            print("Pushover token found and looks good")
        else:
            print("Pushover token found but doesn't start with a")
    else:
        print("Pushover token not found")


# @function_tool turns the function into a tool: the name, the docstring (as the
# description) and the type hints (as the argument schema) become the JSON the model sees
@function_tool
def push_tool(message: str) -> str:
    """ Send the given message to the user as a push notification """
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    result = requests.post(pushover_url, data=payload).status_code
    return f"Push sent with API status code {result}"


async def agent_with_tool():
    """Send a test push, show the tool the SDK built, then let an agent use it."""
    heading("Pushover")
    check_pushover_keys()
    push("HEY!!")                       # a plain function call - check your phone

    heading("The push tool")
    print(f"Name: {push_tool.name}\nDescription: {push_tool.description}")
    print(f"Arguments: {push_tool.params_json_schema}")

    heading("An agent that sends pushes")
    notifier = Agent(name="Notifier", model=MODEL_NAME, instructions="You notify the user upon request", tools=[push_tool])
    with trace("Pizza has arrived"):
        await ask(notifier, "Notify the user that the pizza is here")


# ---- Part 3: memory ---------------------------------------------------------------

async def memory():
    """Show that runs forget each other, then two ways to give the agent memory."""
    agent = Agent(name="Assistant", model=MODEL_NAME)

    heading("No memory: two separate runs")
    await ask(agent, "Hi there. My name is Ed.")
    await ask(agent, "What's my name?")     # it can't know

    heading("Memory approach 1: pass the conversation back in yourself")
    response = await ask(agent, "Hi there. My name is Ed.")
    # to_input_list() is the conversation so far; add the next question to it
    next_input = response.to_input_list() + [{"role": "user", "content": "What's my name?"}]
    print(next_input)
    await ask(agent, next_input)

    heading("Memory approach 2: a session stores the conversation")
    # In memory only; SQLiteSession("12345", "memory.db") would keep it on disk
    session = SQLiteSession("12346")
    await ask(agent, "Hi there. My name is Ed.", session=session)
    await ask(agent, "What's my name?", session=session)


# ---- Main -------------------------------------------------------------------------

async def main():
    """Run the three parts of the lab in order."""
    await simple_agent()
    await agent_with_tool()
    await memory()


if __name__ == "__main__":
    asyncio.run(main())
