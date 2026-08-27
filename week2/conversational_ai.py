import os
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

system_message = "You are a helpful assistant in a clothes store. You should try to gently encourage \
the customer to try items that are on sale. Hats are 60% off, and most other items are 50% off. \
For example, if the customer says 'I'm looking to buy a hat', \
you could reply something like, 'Wonderful - we have lots of hats - including several that are part of our sales event.'\
Encourage the customer to buy hats if they are unsure what to get."

def chat(message, history):
 
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    print(f"History: {history}")
    relevant_system_message = system_message
 
    if 'belt' in message.lower():
        relevant_system_message += """ The store does not sell belts; if you are asked for belts, 
                                       be sure to point out other items on sale."""

    messages = [
                {"role": "system", "content": relevant_system_message}] + history + \
                [{"role": "user", "content": message}
            ]
    
    stream = anthropic_client.chat.completions.create(model=claude_model, messages=messages, stream=True)
    
    response = ""
    
    for chunk in stream:
        response += chunk.choices[0].delta.content or ''
        yield response

gr.ChatInterface(fn=chat).launch(inbrowser=True)
