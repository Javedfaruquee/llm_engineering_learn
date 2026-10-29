"""Lab 3 (Week 1, Day 4): a digital twin chatbot, then tools and a hand-built agent loop.

A "digital twin" is a chatbot on your website that answers visitors' questions about
your career, as you. It is built from your LinkedIn profile (twin/linkedin.pdf) and a
short summary (twin/summary.txt) put into the system prompt.

Steps:
  1. Read the LinkedIn PDF and the summary.
  2. A refresher on how chat works: the system prompt sets the scene, and the model
     has no memory - it "remembers" only because the whole conversation is sent
     every time. Shown with three small conversations.
  3. Build the twin's system prompt and test it with one question.
  4. Launch the twin as a Gradio chat app, in one of three versions:
       plain  - just the model and the system prompt
       tool   - the model can call a tool to record a visitor's email (handled once)
       loop   - the same tool, but handled in a loop: the model may call tools as
                many times as it likes before replying. That loop is an agent loop,
                built without any agent framework.

Setup:
  - OPENAI_API_KEY in .env.
  - Replace twin/linkedin.pdf with your own LinkedIn profile (Profile > More > Save to
    PDF), or any PDF such as your resume, and rewrite twin/summary.txt about you.
  - Using a model other than OpenAI? Some reject Gradio's extra history fields; add
    `history = [{"role": h["role"], "content": h["content"]} for h in history]`
    at the start of each chat function.

Run: python lab3_digital_twin_chat_and_tools.py [plain|tool|loop]   (default: loop)
The app opens in your browser; press Ctrl+C in the terminal to stop it.
"""

import argparse
import json
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# ---- Settings ---------------------------------------------------------------------

HERE = Path(__file__).parent
TWIN_DIR = HERE / "twin"                    # holds linkedin.pdf and summary.txt
EMAILS_FILE = HERE / "emails.txt"           # where the email tool records addresses

SMALL_MODEL = "gpt-5.4-nano"                # for the conversation-memory demo
MODEL = "gpt-5.4-mini"                      # for the twin itself


# ---- Helpers ----------------------------------------------------------------------

def heading(title):
    """Print a section title, so each step's output is easy to find."""
    print(f"\n{'=' * 10} {title} {'=' * 10}\n")


def reply_to(openai, model, messages, tools=None):
    """Send a list of chat messages to model and return the full response.

    tools, if given, is the list of tool definitions the model may call.
    """
    if tools:
        return openai.chat.completions.create(model=model, messages=messages, tools=tools)
    return openai.chat.completions.create(model=model, messages=messages)


# ---- Step 1: the person's details --------------------------------------------------

def load_profile():
    """Return (LinkedIn text, summary text) read from the twin folder.

    PdfReader pulls the text out of each page of the PDF; pages with no text
    (e.g. images only) are skipped.
    """
    reader = PdfReader(TWIN_DIR / "linkedin.pdf")
    linkedin = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            linkedin += text

    with open(TWIN_DIR / "summary.txt", "r", encoding="utf-8") as f:
        summary = f.read()
    return linkedin, summary


# ---- Step 2: how chat works --------------------------------------------------------

def demo_conversation_memory(openai):
    """Show the system prompt at work, and that the model only knows what it is sent.

    1. A helpful assistant greets Ed.
    2. Same greeting, but the system prompt makes it snarky.
    3. Asked "What's my name?" with no history, it can't know.
    4. With the earlier exchange included, it can - the illusion of memory.
    """
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hi, my name is Ed"},
        ],
        [
            {"role": "system", "content": "You are a snarky, witty assistant"},
            {"role": "user", "content": "Hi, my name is Ed"},
        ],
        [
            {"role": "system", "content": "You are a snarky, witty assistant"},
            {"role": "user", "content": "What's my name?"},
        ],
        [
            {"role": "system", "content": "You are a snarky, witty assistant"},
            {"role": "user", "content": "Hi, my name is Ed"},
            {"role": "assistant", "content": "Well hi there, Ed. It's nice to meet you."},
            {"role": "user", "content": "What's my name?"},
        ],
    ]
    for number, messages in enumerate(conversations, start=1):
        heading(f"Conversation {number}: system prompt '{messages[0]['content']}', last message '{messages[-1]['content']}'")
        response = reply_to(openai, SMALL_MODEL, messages)
        print(response.choices[0].message.content)


# ---- Step 3: the twin's system prompt ----------------------------------------------

def make_system_prompt(summary, linkedin):
    """The twin's system prompt: its role, the person's details, and rules of behaviour."""
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
Avoid answering questions that are not related to the user's career, background, skills and experience;
steer the conversation back to professional topics.

Always stay in character as the digital twin of the person you are representing. Represent the person.

IMPORTANT: If you don't know the answer, say so. Never make up an answer.
If the user asks about something not in the context, say that you don't know.
"""


# ---- Step 4: the tool ------------------------------------------------------------
# A tool is just a Python function plus a JSON description of it. The model never runs
# the function: it replies asking for a call, our code runs it and sends back the result.

def record_email_tool(email):
    """The tool: append the visitor's email address to emails.txt."""
    print(f"Tool called to record an email: {email}")
    with open(EMAILS_FILE, "a", encoding="utf-8") as f:
        f.write(email + "\n")
    return "Email received"


# The description the model sees: the tool's name, what it's for, and its arguments
# (a JSON schema)
record_email_tool_json = {
    "name": "record_email_tool",
    "description": "Use this tool to record that a user provided their email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"}
        },
        "required": ["email"],
        "additionalProperties": False,
    },
}

TOOLS = [{"type": "function", "function": record_email_tool_json}]


# ---- Step 4: the three versions of the chat function --------------------------------
# Gradio calls chat(message, history) for each visitor message, where history is the
# conversation so far in OpenAI's message format. Each make_* function returns a chat
# function that uses the given client and system prompt.

def make_chat_plain(openai, system_prompt):
    """Version 1: no tools. System prompt + history + the new message, one call."""

    def chat(message, history):
        """Reply to the visitor's message."""
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        response = reply_to(openai, MODEL, messages)
        return response.choices[0].message.content

    return chat


def make_chat_with_one_tool_call(openai, system_prompt):
    """Version 2: the model may ask to record an email; we handle one such request.

    Tool calling is just an "if": if the model stopped because it wants a tool
    (finish_reason == "tool_calls"), run the tool, add its result to the messages,
    and call the model again. This version assumes the only tool is the email tool,
    and that the model asks for just one call.
    """

    def chat(message, history):
        """Reply to the visitor's message, recording their email if the model asks to."""
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        response = reply_to(openai, MODEL, messages, TOOLS)

        if response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_call = message.tool_calls[0]
            email = json.loads(tool_call.function.arguments).get("email")
            record_email_tool(email)
            # The model's tool request, then the tool's result, tied together by the call id
            messages.append(message)
            messages.append({"role": "tool", "content": "Email recorded", "tool_call_id": tool_call.id})
            response = reply_to(openai, MODEL, messages, TOOLS)

        return response.choices[0].message.content

    return chat


def make_chat_with_agent_loop(openai, system_prompt):
    """Version 3: our first agent loop, without an agent framework.

    Two changes from version 2:
      - "if" becomes "while": keep running tools and calling the model until it
        replies without asking for a tool.
      - Each reply may ask for several tool calls; handle them all with a for loop.
    Agent frameworks do exactly this for you behind the scenes.
    """

    def chat(message, history):
        """Reply to the visitor's message, running as many tool calls as the model asks for."""
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": message}]
        response = reply_to(openai, MODEL, messages, TOOLS)

        while response.choices[0].finish_reason == "tool_calls":
            message = response.choices[0].message
            messages.append(message)
            for tool_call in message.tool_calls:
                email = json.loads(tool_call.function.arguments).get("email")
                record_email_tool(email)
                messages.append({"role": "tool", "content": "Email recorded", "tool_call_id": tool_call.id})
            response = reply_to(openai, MODEL, messages, TOOLS)

        return response.choices[0].message.content

    return chat


CHAT_VERSIONS = {
    "plain": make_chat_plain,
    "tool": make_chat_with_one_tool_call,
    "loop": make_chat_with_agent_loop,
}


# ---- Main -------------------------------------------------------------------------

def main():
    """Run the lab's steps, then launch the chosen version of the twin in the browser."""
    parser = argparse.ArgumentParser(description="Lab 3: the digital twin")
    parser.add_argument("version", nargs="?", default="loop", choices=CHAT_VERSIONS,
                        help="which chat function to launch (default: loop)")
    version = parser.parse_args().version

    load_dotenv(override=True)
    openai = OpenAI()

    heading("Profile")
    linkedin, summary = load_profile()
    print(summary)
    print(f"LinkedIn profile: {len(linkedin):,} characters. It starts:\n{linkedin[:500]}...")

    demo_conversation_memory(openai)

    system_prompt = make_system_prompt(summary, linkedin)
    heading("The twin's system prompt")
    print(system_prompt)

    chat = CHAT_VERSIONS[version](openai, system_prompt)
    heading(f"Test of the '{version}' twin: Please summarize who you are")
    print(chat("Please summarize who you are", []))

    heading(f"Launching the '{version}' twin in your browser (Ctrl+C to stop)")
    gr.ChatInterface(chat).launch(inbrowser=True)


if __name__ == "__main__":
    main()
