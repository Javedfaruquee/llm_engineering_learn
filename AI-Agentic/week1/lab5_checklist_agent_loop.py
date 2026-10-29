"""Lab 5 (Week 1, extra): a visible agent loop that plans with a checklist.

The digital twin's agent loop ran out of sight, inside each chat reply. Here the loop is
the whole program, with two ideas borrowed from Claude Code:
  1. A terminal UI: output is formatted with the rich library (colours, strikethrough).
  2. A checklist tool: the model first writes a plan as a checklist, then ticks off
     each item as it works through it. This makes it take many small, visible steps
     instead of answering in one go.

Steps:
  1. Try the checklist tools by hand: create a list and tick off an item.
  2. Give the model the tools and a word problem (when do two trains meet?), and let
     the agent loop run until the model replies with the solution.

Setup: OPENAI_API_KEY in .env; the rich package (pip install rich).

Run: python lab5_checklist_agent_loop.py
"""

import json

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

# ---- Settings ---------------------------------------------------------------------

MODEL = "gpt-5.5"

SYSTEM_MESSAGE = """
You are given a problem to solve, by using your checklist tools to plan a list of steps, then carrying out each step in turn.
Now create a plan, set the checklist, carry out the steps, and reply with the solution.
If any quantity isn't provided in the question, then include a step to come up with a reasonable estimate.
Provide your solution in Rich console markup without code blocks.
Do not ask the user questions or clarification; respond only with the answer after using your tools.
"""

USER_MESSAGE = """
A train leaves Boston at 2:00 pm traveling 60 mph.
Another train leaves New York at 3:00 pm traveling 80 mph toward Boston.
When do they meet?
"""


# ---- Terminal output --------------------------------------------------------------

def show(text):
    """Print text with rich, which renders markup like [green]...[/green].

    Falls back to plain print if rich can't render it (e.g. malformed markup).
    """
    try:
        Console().print(text)
    except Exception:
        print(text)


# ---- The checklist: the state the tools work on -------------------------------------
# Two parallel lists: item descriptions, and whether each one is done.

checklist = []
completed = []


def reset_checklist():
    """Empty the checklist, ready for a new problem."""
    checklist.clear()
    completed.clear()


def get_checklist_report() -> str:
    """Show the checklist (done items in green and struck through) and return it as text."""
    result = ""
    for index, item in enumerate(checklist):
        if completed[index]:
            result += f"Checklist #{index + 1}: [green][strike]{item}[/strike][/green]\n"
        else:
            result += f"Checklist #{index + 1}: {item}\n"
    show(result)
    return result


# ---- The tools --------------------------------------------------------------------
# Both return the updated checklist, so the model always sees where it's up to.

def create_checklist(descriptions: list[str]) -> str:
    """Tool: add items to the checklist (none done yet) and return the full list."""
    checklist.extend(descriptions)
    completed.extend([False] * len(descriptions))
    return get_checklist_report()


def mark_complete(index: int, completion_notes: str) -> str:
    """Tool: tick off item number index (counting from 1), showing the model's notes on it."""
    if 1 <= index <= len(checklist):
        completed[index - 1] = True
    else:
        return "No checklist at this index."
    show(completion_notes)
    return get_checklist_report()


# The descriptions the model sees: name, purpose and arguments (JSON schema) of each tool
create_checklist_json = {
    "name": "create_checklist",
    "description": "Add new checklist from a list of descriptions and return the full list",
    "parameters": {
        "type": "object",
        "properties": {
            "descriptions": {
                "type": "array",
                "items": {"type": "string"},
                "title": "Descriptions of checklist items",
            }
        },
        "required": ["descriptions"],
        "additionalProperties": False,
    },
}

mark_complete_json = {
    "name": "mark_complete",
    "description": "Mark complete the checklist item at the given position (starting from 1) and return the full list",
    "parameters": {
        "properties": {
            "index": {
                "description": "The 1-based index of the checklist item to mark as complete",
                "title": "Index",
                "type": "integer",
            },
            "completion_notes": {
                "description": "Notes about how you completed the checklist item in rich console markup",
                "title": "Completion Notes",
                "type": "string",
            },
        },
        "required": ["index", "completion_notes"],
        "type": "object",
        "additionalProperties": False,
    },
}

TOOLS = [
    {"type": "function", "function": create_checklist_json},
    {"type": "function", "function": mark_complete_json},
]

# Tool name -> function; only these can be called (the notebook used globals())
TOOL_MAP = {
    "create_checklist": create_checklist,
    "mark_complete": mark_complete,
}


# ---- The agent loop ---------------------------------------------------------------

def handle_tool_calls(tool_calls):
    """Run the tool calls the model asked for; return one "tool" message per call with its result."""
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = TOOL_MAP.get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results


def loop(openai, messages):
    """The agent loop: call the model; while it asks for tools, run them and call it again.

    Each pass the model sees everything so far - its plan, the tool results - and
    decides the next step. When it replies without a tool call, show that reply.
    """
    response = openai.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message
        results = handle_tool_calls(message.tool_calls)
        messages.append(message)        # the model's tool request, then the results
        messages.extend(results)
        response = openai.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    show(response.choices[0].message.content)


# ---- Steps ------------------------------------------------------------------------

def demo_checklist_tools():
    """Try the tools by hand, with no LLM: make a 3-item list and tick off the first item."""
    show("[bold]Trying the checklist tools by hand[/bold]")
    create_checklist(["Buy groceries", "Finish week 1", "Eat banana"])
    mark_complete(1, "bought")


def solve_problem(openai):
    """Hand the model the train problem and let the agent loop plan and solve it."""
    show("[bold]The agent solves the train problem[/bold]")
    reset_checklist()
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_MESSAGE},
    ]
    loop(openai, messages)


# ---- Main -------------------------------------------------------------------------

def main():
    """Run the lab's steps in order."""
    load_dotenv(override=True)
    demo_checklist_tools()
    solve_problem(OpenAI())             # starts by clearing the demo's checklist


if __name__ == "__main__":
    main()
