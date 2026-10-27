from typing import Optional, List
from agents.agent import Agent
from agents.deals import Deal, Opportunity
from agents.scanner_agent import ScannerAgent
from agents.ensemble_agent import EnsembleAgent
from agents.messaging_agent import MessagingAgent
from openai import OpenAI
import json


class AutonomousPlanningAgent(Agent):
    """
    Does the same job as PlanningAgent, but an LLM decides the steps (Day 4).
    The model is given three tools - scan for deals, estimate a product's value, notify
    the user - and a goal, and calls the tools itself in a loop until it is done.
    """

    name = "Autonomous Planning Agent"
    color = Agent.GREEN
    MODEL = "gpt-5.1"

    def __init__(self, collection):
        """
        Create instances of the 3 Agents that this planner coordinates across
        :param collection: the Chroma "products" collection, passed on to the EnsembleAgent
        """
        self.log("Autonomous Planning Agent is initializing")
        self.scanner = ScannerAgent()
        self.ensemble = EnsembleAgent(collection)
        self.messenger = MessagingAgent()
        self.openai = OpenAI()
        self.memory = None                  # past opportunities, set for each run by plan()
        self.opportunity = None             # the deal the model notified about this run
        self.log("Autonomous Planning Agent is ready")

    # ---- The tools: called by handle_tool_call when the model asks for them ----------
    # Each returns a string, which is sent back to the model as the tool's result.

    def scan_the_internet_for_bargains(self) -> str:
        """
        Run the tool to scan
        Returns the selected deals as JSON, skipping any in memory
        """
        self.log("Autonomous Planning agent is calling scanner")
        results = self.scanner.scan(memory=self.memory)
        return results.model_dump_json() if results else "No deals found"

    def estimate_true_value(self, description: str) -> str:
        """
        Run the tool to estimate true value
        Prices the described product with the EnsembleAgent
        """
        self.log("Autonomous Planning agent is estimating value via Ensemble Agent")
        estimate = self.ensemble.price(description)
        return f"The estimated true value of {description} is {estimate}"

    def notify_user_of_deal(
        self, description: str, deal_price: float, estimated_true_value: float, url: str
    ) -> str:
        """
        Run the tool to notify the user
        Sends the push notification and records the deal as this run's Opportunity.
        Only the first call in a run does anything, in case the model calls it twice.
        """
        if self.opportunity:
            self.log("Autonomous Planning agent is trying to notify the user a 2nd time; ignoring")
        else:
            self.log("Autonomous Planning agent is notifying user")
            self.messenger.notify(description, deal_price, estimated_true_value, url)
            deal = Deal(product_description=description, price=deal_price, url=url)
            discount = estimated_true_value - deal_price
            self.opportunity = Opportunity(
                deal=deal, estimate=estimated_true_value, discount=discount
            )
        return "Notification sent ok"

    # ---- Tool definitions: JSON schemas telling the model what each tool does ------------
    # The model reads the descriptions to decide when to call a tool and with what arguments.

    scan_function = {
        "name": "scan_the_internet_for_bargains",
        "description": "Returns top bargains scraped from the internet along with the price each item is being offered for",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }

    estimate_function = {
        "name": "estimate_true_value",
        "description": "Given the description of an item, estimate how much it is actually worth",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item to be estimated",
                },
            },
            "required": ["description"],
            "additionalProperties": False,
        },
    }

    notify_function = {
        "name": "notify_user_of_deal",
        "description": "Send the user a push notification about the single most compelling deal; only call this one time",
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "The description of the item itself scraped from the internet",
                },
                "deal_price": {
                    "type": "number",
                    "description": "The price offered by this deal scraped from the internet",
                },
                "estimated_true_value": {
                    "type": "number",
                    "description": "The estimated actual value that this is worth",
                },
                "url": {
                    "type": "string",
                    "description": "The URL of this deal as scraped from the internet",
                },
            },
            "required": ["description", "deal_price", "estimated_true_value", "url"],
            "additionalProperties": False,
        },
    }

    def get_tools(self):
        """
        Return the json for the tools to be used
        In the format OpenAI's chat completions API expects
        """
        return [
            {"type": "function", "function": self.scan_function},
            {"type": "function", "function": self.estimate_function},
            {"type": "function", "function": self.notify_function},
        ]

    def handle_tool_call(self, message):
        """
        Actually call the tools associated with this message
        :param message: the model's reply, containing one or more tool calls
        :return: one "tool" role message per call, holding that tool's result
        """
        mapping = {
            "scan_the_internet_for_bargains": self.scan_the_internet_for_bargains,
            "estimate_true_value": self.estimate_true_value,
            "notify_user_of_deal": self.notify_user_of_deal,
        }
        results = []
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            tool = mapping.get(tool_name)
            result = tool(**arguments) if tool else ""      # unknown tool -> empty result
            results.append({"role": "tool", "content": result, "tool_call_id": tool_call.id})
        return results

    # ---- The prompts: the goal, and the steps the model should take ----------------------

    system_message = "You find great deals on bargain products using your tools, and notify the user of the best bargain."
    user_message = """
    First, use your tool to scan the internet for bargain deals. Then for each deal, use your tool to estimate its true value.
    Then pick the single most compelling deal where the price is much lower than the estimated true value, and use your tool to notify the user.
    Then just reply OK to indicate success.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    def plan(self, memory: List[Opportunity] = []) -> Optional[Opportunity]:
        """
        Run the full workflow, providing the LLM with tools to surface scraped deals to the user
        Calls the model repeatedly: while it asks for tools, run them and send back the
        results; when it replies without a tool call, the run is over.
        :param memory: the opportunities surfaced in the past; their deals are skipped
        :return: an Opportunity if one was surfaced, otherwise None
        """
        self.log("Autonomous Planning Agent is kicking off a run")
        self.memory = memory
        self.opportunity = None
        messages = self.messages[:]         # a copy, so each run starts from the same prompts
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model=self.MODEL, messages=messages, tools=self.get_tools()
            )
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                results = self.handle_tool_call(message)
                messages.append(message)    # the model's tool request, then the results
                messages.extend(results)
            else:
                done = True
        reply = response.choices[0].message.content
        self.log(f"Autonomous Planning Agent completed with: {reply}")
        return self.opportunity
