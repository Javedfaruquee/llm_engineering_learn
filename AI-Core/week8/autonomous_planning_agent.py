"""Week 8, Day 4: an autonomous planning agent that drives the other agents with tool calls.

Instead of hard-coding the workflow (scan -> price -> notify), we give an LLM three tools
and a goal, and let it decide which tool to call, with what arguments, and when to stop.

Steps:
    1. Get a fixed set of example deals from the ScannerAgent (no scraping, no LLM call).
    2. Define three FAKE tools - scan, estimate value, notify - that just print and return
       canned results, so the tool-calling loop can be watched cheaply and safely.
    3. Run the loop: call the model with the tools; while it asks for tool calls, run them
       and send the results back; stop when it answers normally.
    4. Run the real AutonomousPlanningAgent, whose tools call the real Scanner, Ensemble
       and Messaging agents - so it scrapes live deals, prices them and sends a push.

Needs OPENAI_API_KEY, the products_vectorstore built on Day 2, the Modal pricer-service
from Day 1, and the Pushover keys from Day 3 for the real run.
"""

import json
import logging

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

from agents.autonomous_planning_agent import AutonomousPlanningAgent
from agents.scanner_agent import ScannerAgent


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

MODEL = AutonomousPlanningAgent.MODEL       # gpt-5.1
DB = "products_vectorstore"                 # Chroma database built on Day 2
COLLECTION = "products"

# The tool definitions and prompts are the same ones the real agent uses
TOOLS = [
    {"type": "function", "function": AutonomousPlanningAgent.scan_function},
    {"type": "function", "function": AutonomousPlanningAgent.estimate_function},
    {"type": "function", "function": AutonomousPlanningAgent.notify_function},
]


# ---------------------------------------------------------------------------
# Fake tools
# ---------------------------------------------------------------------------

def make_fake_tools(test_results):
    """Build the three fake tools, keyed by the names the model will call them by.

    Args:
        test_results: A DealSelection of example deals, returned by the fake scan.

    Returns:
        A dict of tool name -> function.
    """
    def scan_the_internet_for_bargains() -> str:
        """This tool scans the internet for great deals and gets a curated list of promising deals"""
        print("Fake function to scan the internet - this returns a hardcoded set of deals")
        return test_results.model_dump_json()

    def estimate_true_value(description: str) -> str:
        """This tool estimates the true value of a product based on a text description of it"""
        print(f"Fake function to estimating true value of {description[:20]}... - this always returns $300")
        return f"Product {description} has an estimated true value of $300"

    def notify_user_of_deal(description: str, deal_price: float, estimated_true_value: float, url: str) -> str:
        """This tool notifies the user of a great deal, given a description of it, the price of the deal, and the estimated true value"""
        print(f"Fake function to notify user of {description} which costs {deal_price} and estimate is {estimated_true_value}")
        return "notification sent ok"

    return {tool.__name__: tool for tool in (scan_the_internet_for_bargains, estimate_true_value, notify_user_of_deal)}


def handle_tool_calls(message, tools):
    """Run every tool call the model asked for in this message.

    Returns:
        One "tool" message per call, carrying the result back to the model under the
        call's id. Unknown tool names get an empty result.
    """
    results = []
    for tool_call in message.tool_calls:
        tool = tools.get(tool_call.function.name)
        arguments = json.loads(tool_call.function.arguments)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results


def run_tool_loop(openai, tools):
    """Let the model work towards its goal using the given tools, until it stops calling them.

    The goal (in the agent's prompts): scan for deals, estimate each one's true value,
    notify the user of the single best bargain, then reply OK.

    Returns:
        The model's final text reply.
    """
    messages = [
        {"role": "system", "content": AutonomousPlanningAgent.system_message},
        {"role": "user", "content": AutonomousPlanningAgent.user_message},
    ]
    while True:
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
        choice = response.choices[0]
        if choice.finish_reason != "tool_calls":
            return choice.message.content
        # Keep the model's request and our results in the conversation, then go round again
        messages.append(choice.message)
        messages.extend(handle_tool_calls(choice.message, tools))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Watch the tool loop with fake tools, then run the real autonomous agent."""
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")    # show the agents' logs

    test_results = ScannerAgent().test_scan()
    print(test_results)

    fake_tools = make_fake_tools(test_results)
    fake_tools["notify_user_of_deal"]("a new iphone", 100, 1000, "https://www.apple.com/iphone")   # try one
    print(json.dumps(TOOLS, indent=2))
    print(run_tool_loop(OpenAI(), fake_tools))

    # Now with the real tools: live scraping, real pricing, a real push notification
    collection = chromadb.PersistentClient(path=DB).get_or_create_collection(COLLECTION)
    print(AutonomousPlanningAgent(collection).plan())


if __name__ == "__main__":
    main()
