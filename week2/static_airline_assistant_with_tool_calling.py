import os
import json
import requests
import gradio as gr
from IPython.display import Markdown, display
from dotenv import load_dotenv
from openai import OpenAI

# Read the API keys from the .env file
load_dotenv(override=True)
openai_api_key = os.getenv("OPENAI_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Which model each bot uses
gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"
gemini_model = "gemini-3.5-flash-lite" 

# Connect to OpenAI, Anthropic and Google; comment out the Claude or Google lines if you're 
# not using them

openai = OpenAI()

# Create one client per bot.
openai_client = OpenAI(api_key=openai_api_key)
anthropic_client = OpenAI(api_key=anthropic_api_key, base_url="https://api.anthropic.com/v1/")
gemini_client = OpenAI(api_key=google_api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

system_message = """
You are a helpful assistant for an Airline called FlightAI.
Give short, courteous answers, no more than 1 sentence.
Always be accurate. If you don't know the answer, say so.
Use get_ticket_price to look up a price. If the customer tells you the price for a
city we don't have yet, use add_ticket_price to save it.
"""

ticket_prices = {"london": "$799", "paris": "$899", "tokyo": "$1400", "berlin": "$499"}

# The two functions that do the work. Their parameter names match the properties
# in the schemas below, which is what lets us call them with **arguments.

def get_ticket_price(destination_city):
    print(f"Tool called for city {destination_city}")
    price = ticket_prices.get(destination_city.strip().lower(), "Unknown ticket price")
    return f"The price of a ticket to {destination_city} is {price}"

def add_ticket_price(destination_city, price):
    print(f"Tool called to add city {destination_city} at {price}")
    ticket_prices[destination_city.strip().lower()] = price.strip()
    return f"The price of a ticket to {destination_city} is now {price}"

# There's a particular dictionary structure that's required to describe our functions.
# Both tools live in this one list:

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_price",
            "description": "Get the price of a return ticket to the destination city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The city that the customer wants to travel to",
                    },
                },
                "required": ["destination_city"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_ticket_price",
            "description": "Add a new destination city and its return ticket price to the price list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_city": {
                        "type": "string",
                        "description": "The city to add to the ticket price list",
                    },
                    "price": {
                        "type": "string",
                        "description": "The price of a return ticket to that city, for example $499",
                    },
                },
                "required": ["destination_city", "price"],
                "additionalProperties": False
            }
        }
    },
]

# Look up which function to run by name, instead of a long if/elif chain:

tool_functions = {
    "get_ticket_price": get_ticket_price,
    "add_ticket_price": add_ticket_price,
}

# We have to write that function handle_tool_call:

def handle_tool_calls(message):
    responses = []
    for tool_call in message.tool_calls:
        function = tool_functions[tool_call.function.name]
        arguments = json.loads(tool_call.function.arguments)
        responses.append({
            "role": "tool",
            "content": function(**arguments),
            "tool_call_id": tool_call.id
        })
    return responses

def chat(message, history):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = gemini_client.chat.completions.create(model=gemini_model, messages=messages, tools=tools)

    while response.choices[0].finish_reason=="tool_calls":
        message = response.choices[0].message
        responses = handle_tool_calls(message)
        messages.append(message)
        messages.extend(responses)
        response = gemini_client.chat.completions.create(model=gemini_model, messages=messages, tools=tools)
    
    return response.choices[0].message.content

gr.ChatInterface(fn=chat).launch(inbrowser=True)
