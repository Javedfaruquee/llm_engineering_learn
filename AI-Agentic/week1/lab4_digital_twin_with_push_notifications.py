"""Lab 4 (Week 1, Day 5): the full digital twin, with tools that send push notifications.

Builds on Lab 3. The twin now has two tools, and both alert you on your phone via
Pushover:
  - record_user_details: a visitor wants to get in touch and gave their email
  - record_unknown_question: the twin was asked something it couldn't answer, so you
    know what to add to its profile
Tool calls are dispatched by name through a dictionary instead of an if/elif chain,
so adding a tool means adding one entry.

Steps:
  1. Check the Pushover keys and send a test push.
  2. Read the LinkedIn PDF and summary, and build the twin's system prompt.
  3. Launch the twin as a Gradio chat app with an agent loop.

Setup:
  - OPENAI_API_KEY in .env.
  - Pushover (free): sign up at https://pushover.net, click "Create an
    Application/API Token" (name it e.g. Agents), and add to .env:
        PUSHOVER_USER=<the user key on your Pushover home screen; starts with u>
        PUSHOVER_TOKEN=<your new application's token; starts with a>
    Then "Add Phone, Tablet or Desktop" to install the Pushover app on your phone.
  - Your own twin/linkedin.pdf and twin/summary.txt (see Lab 3).

This is the same app as twin/app.py, which splits it into modules: context.py
(system prompt), tools.py (tools), styles.py (look and feel), app.py (chat and UI).
To deploy it for free, see RENDER_INSTRUCTIONS.md (HuggingFace Spaces is no longer free).

Run: python lab4_digital_twin_with_push_notifications.py
The app opens in your browser; press Ctrl+C in the terminal to stop it.
"""

import json
import os
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(override=True)

# ---- Settings ---------------------------------------------------------------------

TWIN_DIR = Path(__file__).parent / "twin"   # holds linkedin.pdf and summary.txt
MODEL = "gpt-5.4-mini"

pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


# ---- Step 1: push notifications ---------------------------------------------------

def check_pushover_keys():
    """Report whether the Pushover user key and app token are set and look right.

    Pushover user keys start with "u" and application tokens with "a".
    """
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


def push(message):
    """Send message as a push notification to your phone (and print it)."""
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)


# ---- The tools --------------------------------------------------------------------
# Each returns "OK", which is sent back to the model as the tool's result.

def record_user_details(email, name="Name not provided", notes="not provided"):
    """Tool: push an alert that a visitor wants to be in touch, with their details."""
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return "OK"


def record_unknown_question(question):
    """Tool: push an alert with a question the twin couldn't answer."""
    push(f"Recording {question} asked that I couldn't answer")
    return "OK"


# The descriptions the model sees: name, when to use it, and its arguments (JSON schema)
record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {
                "type": "string",
                "description": "Any additional info about the conversation that's worth recording to give context",
            },
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"},
        },
        "required": ["question"],
        "additionalProperties": False,
    },
}

TOOLS = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json},
]

# Tool name -> function. The notebook looks tools up in globals(), which would let the
# model call any function in the program; this dictionary allows only these two.
TOOL_MAP = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question,
}


def handle_tool_calls_with_manual_if(tool_calls):
    """Run the tool calls the model asked for, choosing each tool with an if statement.

    Kept to show what tool dispatch boils down to; handle_tool_calls below is the
    tidier version the app uses.
    Returns one "tool" message per call, holding that tool's result.
    """
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)

        # THE BIG IF STATEMENT!!!
        if tool_name == "record_user_details":
            result = record_user_details(**arguments)
        elif tool_name == "record_unknown_question":
            result = record_unknown_question(**arguments)

        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results


def handle_tool_calls(tool_calls):
    """Run the tool calls the model asked for, looking each tool up by name in TOOL_MAP.

    Returns one "tool" message per call, holding that tool's result.
    """
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = TOOL_MAP.get(tool_name)
        result = tool(**arguments) if tool else "No tool found"
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results


# ---- Step 2: the twin's system prompt ----------------------------------------------

def load_profile():
    """Return (LinkedIn text, summary text) read from the twin folder."""
    reader = PdfReader(TWIN_DIR / "linkedin.pdf")
    linkedin = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:                        # skip pages with no text, e.g. images only
            linkedin += text

    with open(TWIN_DIR / "summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()
    return linkedin, summary


def make_system_prompt(summary, linkedin):
    """The twin's system prompt: as in Lab 3, plus rules telling it when to use its tools."""
    return f"""

# Your role

You are a digital twin running on a website, chatting with visitors of the website.
You represent the person who's website you are on.
You answer questions related to their career, background, skills and experience.

Here are the details of the person you are representing:

{summary}

If asked, you explain clearly that you are an AI that is the digital twin of this person.

# Context

Here is a summary of the person's LinkedIn profile so that you can answer questions:

{linkedin}

# Rules

Engage with the user. Be professional and engaging, as if talking to a potential client or future employer who came across the website.
Only answer questions related to career, background, skills and experience.
If the user asks about something unrelated, then steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

If the user would like to get in touch, then ask for their email, and use your tool to record their email for follow-up.

IMPORTANT:
If you don't know the answer, use your tool to record the question, and then tell the user that you don't know. Never make up an answer.
"""


# ---- Step 3: the chat function with an agent loop ------------------------------------

def make_chat(openai, system_prompt):
    """Return the chat(message, history) function Gradio calls for each visitor message.

    The agent loop: while the model asks for tools, run them, add the results to the
    conversation and call the model again; when it replies with text, return that.
    """

    def chat(message, history):
        """Reply to the visitor's message, running any tools the model asks for."""
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            results = handle_tool_calls(message.tool_calls)
            messages.append(message)    # the model's tool request, then the results
            messages.extend(results)
            response = openai.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        return response.choices[0].message.content

    return chat


# ---- Main -------------------------------------------------------------------------

def main():
    """Check Pushover, build the twin, and launch it in the browser."""
    check_pushover_keys()
    push("HEY!!")                       # a test push - check it arrives on your phone

    linkedin, summary = load_profile()
    system_prompt = make_system_prompt(summary, linkedin)
    chat = make_chat(OpenAI(), system_prompt)

    print("Launching the digital twin in your browser (Ctrl+C to stop)")
    gr.ChatInterface(chat).launch(inbrowser=True)


if __name__ == "__main__":
    main()
