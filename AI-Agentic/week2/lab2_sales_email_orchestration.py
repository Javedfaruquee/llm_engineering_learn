"""Lab 2 (Week 2, Day 2): orchestrating several agents - by code, and by an LLM.

The project: an automated sales team for ComplAI, a (made up) company selling an AI
tool for SOC2 compliance. Three sales agents write cold emails in different styles,
and the best one is sent.

Steps:
  Part 1 - Messaging: send a test message by email, or by push notification if email
           isn't set up.
  Part 2 - Orchestration by code: our code decides the steps. Run the three sales
           agents in parallel, have a picker agent choose the best email, then a
           sender agent that must use a tool to send it. Predictable and reliable.
  Part 3 - Orchestration by an LLM: a Sales Manager agent decides the steps.
           3a: the sales agents are turned into tools the manager calls
               (control passes A -> B -> A).
           3b: the manager hands off to the sender agent, which takes over
               (control passes A -> B). More powerful, less predictable.

Setup:
  - OPENAI_API_KEY in .env.
  - Email (optional, else push notifications are used): in .env set
        EMAIL_SMTP_SERVER   e.g. smtp.gmail.com, smtp-mail.outlook.com, smtp.mail.me.com
        EMAIL_APP_PASSWORD  an app password from your email provider (for Gmail, turn on
                            2-step verification, then https://myaccount.google.com/apppasswords;
                            16 characters, no spaces)
        EMAIL_ADDRESS       your address; emails are sent from and to it
    Or PUSHOVER_USER and PUSHOVER_TOKEN for push notifications.
  - Graphviz (optional) to draw diagrams of the agents with --graph:
    `winget install graphviz` on Windows, `brew install graphviz` on Mac.

Note: this sends several real emails (or pushes), and they may land in your spam folder.
LLM orchestration can be unreliable; if it misbehaves, iterate on the prompts or use a
bigger model. Traces: https://platform.openai.com/traces

Run: python lab2_sales_email_orchestration.py [--graph]
"""

import argparse
import asyncio
import os

from agents import Agent, ModelSettings, Runner, function_tool, trace
from dotenv import load_dotenv
from openai.types.responses import ResponseTextDeltaEvent

from messenger import push, send_email

load_dotenv(override=True)

# ---- Settings ---------------------------------------------------------------------

MODEL_NAME = "gpt-5.4-mini"

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_SMTP_SERVER = os.getenv("EMAIL_SMTP_SERVER")
EMAIL_APP_PASSWORD = os.getenv("EMAIL_APP_PASSWORD")
# Send email only if all three settings are there; otherwise fall back to pushes
USE_EMAIL = bool(EMAIL_ADDRESS and EMAIL_SMTP_SERVER and EMAIL_APP_PASSWORD)

COLD_EMAIL_REQUEST = "Write a cold sales email"


# ---- Helpers ----------------------------------------------------------------------

def heading(title):
    """Print a section title, so each step's output is easy to find."""
    print(f"\n{'=' * 10} {title} {'=' * 10}\n")


# ---- Part 1: messaging --------------------------------------------------------------

def check_messaging_settings():
    """Report which email and Pushover settings are present, and which channel will be used."""
    print("Email address is set" if EMAIL_ADDRESS else "Email address is not set")
    print("SMTP server is set" if EMAIL_SMTP_SERVER else "SMTP server is not set")
    print("App password is set" if EMAIL_APP_PASSWORD else "App password is not set")
    print("Pushover keys are set" if os.getenv("PUSHOVER_USER") and os.getenv("PUSHOVER_TOKEN") else "Pushover keys are not set")
    if USE_EMAIL:
        print("Email is set up and we will try using it")
    else:
        print("Email is not set up; we will send push notifications instead")


def send_message(subject, text_body, html_body):
    """Send by email if it's set up, otherwise as a push notification (text only)."""
    if USE_EMAIL:
        send_email(subject, text_body, html_body)
    else:
        push(f"Subject: {subject}\n\n{text_body}")


@function_tool
def send_email_tool(subject: str, text_body: str, html_body: str) -> str:
    """
    Send out an email with the given subject and body to all sales prospects

    Args:
        subject: The subject of the email
        text_body: The body of the email as plain text
        html_body: The HTML body of the email
    """
    send_message(subject, text_body, html_body)
    return "Email sent successfully"


# ---- The agents -------------------------------------------------------------------

INTRO = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI.
You write emails.
"""

# Same job, three styles
sales_agent1 = Agent(name="Professional Sales Agent", model=MODEL_NAME,
                     instructions=INTRO + "Your email style is professional, serious, with gravitas and credibility.")
sales_agent2 = Agent(name="Humorous Sales Agent", model=MODEL_NAME,
                     instructions=INTRO + "Your email style is witty, engaging, and humorous.")
sales_agent3 = Agent(name="Executive Sales Agent", model=MODEL_NAME,
                     instructions=INTRO + "Your email style is concise, to the point, in the style of a busy senior executive.")
SALES_AGENTS = [sales_agent1, sales_agent2, sales_agent3]

# Picks the best email and replies with it
sales_picker = Agent(name="Sales_picker", model=MODEL_NAME, instructions="""
You pick the best cold sales email from the given options.
Imagine you are a customer and pick the one you are most likely to respond to.
Do not give an explanation; reply with the selected email only.
""")

# Picks the best email and sends it; tool_choice="required" forces it to call a tool
sales_sender = Agent(
    name="Sales Sender",
    model=MODEL_NAME,
    instructions="""
You pick the best cold sales email from the given options.
Imagine you are a customer and pick the one you are most likely to respond to.
Then use your tool to send the email.
""",
    tools=[send_email_tool],
    model_settings=ModelSettings(tool_choice="required"),
)

# Each sales agent wrapped as a tool, so a manager agent can call it
TOOL_DESCRIPTION = "Use this tool to write a sales email. In the input, just instruct it to write a sales email."
sales_writer_tools = [
    agent.as_tool(tool_name=f"sales_email_writer_{number}", tool_description=TOOL_DESCRIPTION)
    for number, agent in enumerate(SALES_AGENTS, start=1)
]


# ---- Part 2: orchestration by code ------------------------------------------------

async def write_three_emails():
    """Run the three sales agents at the same time; return their emails as one text."""
    results = await asyncio.gather(*(Runner.run(agent, COLD_EMAIL_REQUEST) for agent in SALES_AGENTS))
    outputs = [result.final_output for result in results]
    return "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)


async def orchestrate_by_code():
    """Stream one email, then: three in parallel -> pick the best -> pick and send the best."""
    heading("One cold email, streamed")
    result = Runner.run_streamed(sales_agent1, input=COLD_EMAIL_REQUEST)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()

    heading("Three emails in parallel")
    with trace("Parallel cold emails"):
        emails = await write_three_emails()
    print(emails)

    heading("The picker chooses the best")
    with trace("Sales selection workflow"):
        emails = await write_three_emails()
        best = await Runner.run(sales_picker, emails)
    print(f"Best sales email:\n{best.final_output}")

    heading("The sender chooses and sends the best")
    print(f"The tool the SDK built from send_email_tool:\n{send_email_tool.params_json_schema}\n")
    with trace("Sales selection workflow with sending"):
        emails = await write_three_emails()
        response = await Runner.run(sales_sender, emails)
    print(f"Final response:\n{response.final_output}")


# ---- Part 3: orchestration by an LLM ----------------------------------------------

MANAGER_TASK_STEP_1 = """
Follow these steps:

1. Generate Drafts: Use each of the three sales_email_writer tools to generate different email drafts.
Just instruct each to write a sales email; no further details are needed.
Do not proceed until all three drafts are ready, one from each tool.
"""


def draw(agent, filename):
    """Save a diagram of the agent, its tools and handoffs as filename.png (needs Graphviz)."""
    try:
        from agents.extensions.visualization import draw_graph
        draw_graph(agent, filename=filename)
        print(f"Saved a diagram of {agent.name} to {filename}.png")
    except Exception as error:
        print(f"Couldn't draw the diagram (is Graphviz installed?): {error}")


async def manager_with_tools(graph):
    """3a: the Sales Manager calls the three writers as tools, then sends the best email itself."""
    heading("3a: Sales Manager with agents as tools")
    sales_manager = Agent(
        name="Sales Manager",
        instructions="""
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_writer tools.
""",
        tools=sales_writer_tools + [send_email_tool],
        model=MODEL_NAME,
    )
    task = MANAGER_TASK_STEP_1 + """
2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.

3. Use your tool to send the best email (and only the best email) to the user. Only send 1 email.
"""
    if graph:
        draw(sales_manager, "sales_manager_tools")
    with trace("Sales manager"):
        result = await Runner.run(sales_manager, task)
    print(result.final_output)


async def manager_with_handoff(graph):
    """3b: the Sales Manager gathers three drafts, then hands off to the Sales Sender."""
    heading("3b: Sales Manager with a handoff")
    sales_manager = Agent(
        name="Sales Manager",
        instructions="""
You are a Sales Manager at ComplAI. You get your sales team to draft emails, then send them all to a sales picker.
""",
        tools=sales_writer_tools,
        handoffs=[sales_sender],
        model=MODEL_NAME,
    )
    task = MANAGER_TASK_STEP_1 + """
2. Handoff to the sales sender to choose and send the best email.
"""
    if graph:
        draw(sales_manager, "sales_manager_handoff")
    with trace("Sales manager"):
        result = await Runner.run(sales_manager, task)
    print(result.final_output)


# ---- Main -------------------------------------------------------------------------

async def main():
    """Run the three parts of the lab in order."""
    parser = argparse.ArgumentParser(description="Lab 2: orchestrating sales agents")
    parser.add_argument("--graph", action="store_true", help="save diagrams of the Sales Manager (needs Graphviz)")
    graph = parser.parse_args().graph

    heading("Part 1: messaging")
    check_messaging_settings()
    send_message("Big news", "Communications are a go!", "<html><body>Communications are a <strong>go!</strong></body></html>")

    await orchestrate_by_code()
    await manager_with_tools(graph)
    await manager_with_handoff(graph)
    print("\nCheck the traces at https://platform.openai.com/traces, and your inbox (and spam folder)!")


if __name__ == "__main__":
    asyncio.run(main())
