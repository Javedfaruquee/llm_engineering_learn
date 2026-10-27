"""Lab 3 (Week 2, Day 3): other models, structured outputs and guardrails.

Steps:
  Part 1 - Different models: the Lab 2 Sales Manager again, but its three sales agents
           run on Gemini, Kimi (via OpenRouter) and GPT-OSS (via Groq). Any provider
           with an OpenAI-compatible endpoint works in 3 steps: base URL -> AsyncOpenAI
           client -> OpenAIChatCompletionsModel.
  Part 2 - Structured outputs: instead of text, an agent replies with a Python object
           (a pydantic class). The SDK sends the class's JSON schema to the model and
           fills in the object from the model's JSON reply. Not every provider supports it.
  Part 3 - Guardrails: an output guardrail checks a sales agent's email with the
           structured-output checker and blocks it if it's unprofessional or has
           placeholders. Then the same check done by hand, which works in any framework.
           Gotcha: input guardrails only run on the first agent's input, and output
           guardrails only on the last agent's output.
  Optional extras (flags):
    --sandbox  a SandboxAgent fixes the bug in code/orders.py in a sandboxed workspace
               and writes the fixed file to output/. Needs Mac, Linux or Windows + WSL2
               (it is skipped on plain Windows).
    --mcp      an MCP teaser: an agent answers a question about the SDK, first alone,
               then with the Context7 documentation MCP server to look things up.

Setup:
  - OPENAI_API_KEY in .env; GOOGLE_API_KEY, OPENROUTER_API_KEY and GROQ_API_KEY for the
    three sales agents in Part 1 and the cowboy agent in Part 3 (which uses Gemini).
  - Email or Pushover settings as in Lab 2; messages are sent by email (USE_EMAIL).
  - Traces: https://platform.openai.com/traces

Run: python lab3_models_structured_outputs_guardrails.py [--sandbox] [--mcp]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    Runner,
    function_tool,
    output_guardrail,
    trace,
)
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from messenger import push, send_email

load_dotenv(override=True)

# ---- Settings ---------------------------------------------------------------------

HERE = Path(__file__).parent
MODEL_NAME = "gpt-5.4-mini"
USE_EMAIL = True                        # False: send push notifications instead

# Step 1 of using another provider: its OpenAI-compatible base URL
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Each key: (env var, how many characters to show when checking it)
API_KEYS = {
    "OpenAI": ("OPENAI_API_KEY", 8),
    "Google": ("GOOGLE_API_KEY", 2),
    "OpenRouter": ("OPENROUTER_API_KEY", 6),
    "Groq": ("GROQ_API_KEY", 4),
}

SALES_INSTRUCTIONS = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI.
You write compelling sales emails that are likely to get a response.
"""


# ---- Helpers ----------------------------------------------------------------------

def heading(title):
    """Print a section title, so each step's output is easy to find."""
    print(f"\n{'=' * 10} {title} {'=' * 10}\n")


def check_keys():
    """Print which API keys are set, showing only their first few characters."""
    for provider, (key_name, prefix_length) in API_KEYS.items():
        key = os.getenv(key_name)
        optional = "" if provider == "OpenAI" else " (and this is optional)"
        if key:
            print(f"{provider} API Key exists and begins {key[:prefix_length]}")
        else:
            print(f"{provider} API Key not set{optional}")


def make_models():
    """Steps 2 and 3 for each provider: an async client, then a model the SDK can use.

    Returns (gemini_model, kimi_model, oss_model).
    """
    gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=os.getenv("GOOGLE_API_KEY"))
    openrouter_client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=os.getenv("OPENROUTER_API_KEY"))
    groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=os.getenv("GROQ_API_KEY"))

    gemini_model = OpenAIChatCompletionsModel(model="gemini-3.1-flash-lite", openai_client=gemini_client)
    kimi_model = OpenAIChatCompletionsModel(model="moonshotai/kimi-k2.6", openai_client=openrouter_client)
    oss_model = OpenAIChatCompletionsModel(model="openai/gpt-oss-120b", openai_client=groq_client)
    return gemini_model, kimi_model, oss_model


def send_message(subject, text_body, html_body):
    """Send by email, or as a push notification (text only) if USE_EMAIL is False."""
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


# ---- Part 1: different models -----------------------------------------------------

async def sales_manager_across_models(gemini_model, kimi_model, oss_model):
    """Lab 2's Sales Manager, with its three sales agents on three different providers."""
    heading("Part 1: test message")
    send_message("Yet another test", "Hooray!", "<html><body><h1>Hooray!</h1></body></html>")

    sales_agents = [
        Agent(name="Gemini Sales Agent", instructions=SALES_INSTRUCTIONS, model=gemini_model),
        Agent(name="Kimi Sales Agent", instructions=SALES_INSTRUCTIONS, model=kimi_model),
        Agent(name="GPT-OSS Sales Agent", instructions=SALES_INSTRUCTIONS, model=oss_model),
    ]
    description = "Use this tool to write a sales email. In the input, just instruct it to write a sales email."
    tools = [
        agent.as_tool(tool_name=f"sales_agent{number}", tool_description=description)
        for number, agent in enumerate(sales_agents, start=1)
    ] + [send_email_tool]

    instructions = """
You are a Sales Manager at ComplAI. Your goal is to find the single best cold sales email using the sales_agent tools.
"""
    task = """
Follow these steps:

1. Generate Drafts: Use each of the three sales_agent tools to generate different email drafts.
Just instruct each to write a sales email; no further details are needed.
Do not proceed until all three drafts are ready, one from each tool.

2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.

3. Use your tool to send the best email (and only the best email) to the user. Only send 1 email.
"""
    sales_manager = Agent(name="Sales Manager", instructions=instructions, tools=tools, model=MODEL_NAME)

    heading("Part 1: Sales Manager across different models")
    with trace("Sales Manager across different models"):
        result = await Runner.run(sales_manager, task)
    print(result.final_output)


# ---- Part 2: structured outputs ---------------------------------------------------

class EmailReview(BaseModel):
    """The checker's verdict on an email. The Field descriptions are instructions to the model."""

    is_professional: bool = Field(description="Whether the email is professional and appropriate")
    number_of_sentences: int = Field(description="The number of sentences in the body of the email, not including the greeting and signature")
    contains_placeholders: bool = Field(description="Whether the email contains placeholders for personalization")


# output_type makes the agent reply with an EmailReview instead of text
checker = Agent(name="Checker", instructions="You review potential sales emails", model=MODEL_NAME, output_type=EmailReview)

BAD_EMAIL = """
Hi [first_name],

I'm hitting you up to see if you'd like to buy our product. It's really great. You'll miss out if you don't buy it.

Laters.

Ed
"""


async def structured_outputs():
    """Show the JSON schema sent to the model, then review a deliberately bad email."""
    heading("Part 2: the schema the model is given")
    print(EmailReview.model_json_schema())

    heading("Part 2: the checker reviews a bad email")
    result = await Runner.run(checker, BAD_EMAIL)
    review = result.final_output         # an EmailReview object, not text
    print(review)
    print(f"is_professional = {review.is_professional}")


# ---- Part 3: guardrails -----------------------------------------------------------

def is_problem(review: EmailReview) -> bool:
    """An email fails the check if it has placeholders or isn't professional."""
    return review.contains_placeholders or not review.is_professional


@output_guardrail
async def email_guardrail(ctx, agent, message):
    """Output guardrail: review the agent's final email; trip the wire if it fails the check."""
    result = await Runner.run(checker, message, context=ctx.context)
    review = result.final_output
    return GuardrailFunctionOutput(output_info={"review": review}, tripwire_triggered=is_problem(review))


async def guardrails(gemini_model):
    """A cowboy-speaking sales agent, checked by the framework's guardrail, then by hand."""
    cowboy_instructions = SALES_INSTRUCTIONS + "\nSpeak like a cowboy"

    heading("Part 3: guardrail from the framework")
    sales_agent_cowboy = Agent(name="Cowboy", instructions=cowboy_instructions, model=gemini_model, output_guardrails=[email_guardrail])
    try:
        result = await Runner.run(sales_agent_cowboy, "Write a cold sales email")
        print(result.final_output)
    except OutputGuardrailTripwireTriggered as tripwire:
        # When the guardrail trips, the run raises this instead of returning the email
        review = tripwire.guardrail_result.output.output_info["review"]
        print(f"The guardrail blocked the email: {review}")

    heading("Part 3: the same check, done by hand")
    simple_cowboy = Agent(name="Simple Cowboy", instructions=cowboy_instructions, model=gemini_model)
    result = await Runner.run(simple_cowboy, "Write a cold sales email")
    email = result.final_output
    print(email)
    result = await Runner.run(checker, email)
    if is_problem(result.final_output):
        print("The email is not professional or has placeholders and will not be sent")
    else:
        print("Email is good")


# ---- Optional extra: sandbox agents -------------------------------------------------

async def sandbox_bug_fix():
    """A SandboxAgent reviews code/orders.py, finds the bug, and writes a fixed copy to output/.

    Three things to set up: a Manifest (the workspace - our code folder, plus permission
    to write to output/), Capabilities (what the agent may do) and a SandboxRunConfig
    (where it runs - here, a local Unix sandbox). Docs:
    https://openai.github.io/openai-agents-python/sandbox_agents/
    """
    heading("Optional: sandbox agent fixes a bug")
    if sys.platform == "win32":
        print("Skipped: sandbox agents need Mac, Linux or Windows + WSL2 (run this lab inside WSL)")
        return

    from agents.run import RunConfig
    from agents.sandbox import Manifest, SandboxAgent, SandboxPathGrant, SandboxRunConfig
    from agents.sandbox.capabilities import Capabilities
    from agents.sandbox.entries import LocalDir
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    code_dir = (HERE / "code").resolve()
    output_dir = (HERE / "output").resolve()
    output_dir.mkdir(exist_ok=True)

    instructions = f"""
You are a software engineer that fixes bugs.
Review files in the sandbox code directory.

Write the fixed version of the file to this host output directory:
{output_dir}

Use full file paths when writing output.
Respond with a summary of what you did.
"""
    manifest = Manifest(entries={"code": LocalDir(src=code_dir)}, extra_path_grants=[SandboxPathGrant(path=str(output_dir))])
    capabilities = Capabilities.default()
    print(capabilities)
    run_config = RunConfig(sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()), workflow_name="Sandbox coding example")
    agent = SandboxAgent(name="Engineer", instructions=instructions, model=MODEL_NAME, default_manifest=manifest, capabilities=capabilities)

    result = await Runner.run(agent, "Fix the bug in the code", run_config=run_config)
    print(result.final_output)


# ---- Optional extra: MCP teaser ---------------------------------------------------

async def mcp_teaser():
    """Ask about a brand-new SDK feature without and then with the Context7 MCP server.

    Context7 serves up-to-date library documentation; with it, the agent can look the
    answer up instead of guessing from (older) training data.
    """
    from agents.mcp import MCPServerStreamableHttp

    task = """
In the new SandboxAgents feature in the OpenAI Agents SDK as of May 2026, what is the role of the Manifest object?
Always be accurate. If you don't know the answer, say so.
"""
    heading("Optional MCP teaser: without MCP")
    agent = Agent(name="Expert", instructions="Answer the question", model="gpt-4o-mini")
    result = await Runner.run(agent, task)
    print(result.final_output)

    heading("Optional MCP teaser: with the Context7 MCP server")
    params = {"url": "https://mcp.context7.com/mcp", "timeout": 60}
    async with MCPServerStreamableHttp(name="Context7", params=params) as server:
        agent = Agent(name="Expert", instructions="Use Context7 to answer the question", mcp_servers=[server], model="gpt-4o-mini")
        result = await Runner.run(agent, task)
    print(result.final_output)


# ---- Main -------------------------------------------------------------------------

async def main():
    """Run the three parts, then any optional extras asked for on the command line."""
    parser = argparse.ArgumentParser(description="Lab 3: models, structured outputs, guardrails")
    parser.add_argument("--sandbox", action="store_true", help="also run the sandbox agent extra (not on plain Windows)")
    parser.add_argument("--mcp", action="store_true", help="also run the MCP teaser")
    args = parser.parse_args()

    heading("API keys")
    check_keys()
    gemini_model, kimi_model, oss_model = make_models()

    await sales_manager_across_models(gemini_model, kimi_model, oss_model)
    await structured_outputs()
    await guardrails(gemini_model)
    if args.sandbox:
        await sandbox_bug_fix()
    if args.mcp:
        await mcp_teaser()
    print("\nCheck the traces at https://platform.openai.com/traces")


if __name__ == "__main__":
    asyncio.run(main())
