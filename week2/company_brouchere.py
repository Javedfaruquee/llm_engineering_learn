import os
import requests
import gradio as gr
from IPython.display import Markdown, display
from dotenv import load_dotenv
from openai import OpenAI
from scrapers import fetch_website_contents

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

# Define this variable for system message
system_message = """
You are an assistant that analyzes the contents of a company website landing page
and creates a short brochure about the company for prospective customers, investors and recruits.
Add a few sentences about the company, its products for various industries.
Respond in markdown without code blocks.
"""

def stream_gpt(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = openai.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def stream_claude(prompt):
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
      ]
    stream = anthropic_client.chat.completions.create(
        model='claude-sonnet-4-5-20250929',
        messages=messages,
        stream=True
    )
    result = ""
    for chunk in stream:
        result += chunk.choices[0].delta.content or ""
        yield result

def stream_model(prompt, model):
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result

def stream_brochure(company_name, url, model):
    yield ""
    prompt = f"Please generate a company brochure for {company_name}. Here is their landing page:\n"
    prompt += fetch_website_contents(url)
    if model=="GPT":
        result = stream_gpt(prompt)
    elif model=="Claude":
        result = stream_claude(prompt)
    else:
        raise ValueError("Unknown model")
    yield from result

name_input = gr.Textbox(label="Company name:")
url_input = gr.Textbox(label="Landing page URL including http:// or https://")
model_selector = gr.Dropdown(["GPT", "Claude"], label="Select model", value="GPT")
message_output = gr.Markdown(label="Response:")

view = gr.Interface(
    fn=stream_brochure,
    title="Brochure Generator", 
    inputs=[name_input, url_input, model_selector], 
    outputs=[message_output], 
    examples=[
            ["Hugging Face", "https://huggingface.co", "GPT"],
            ["Edward Donner", "https://edwarddonner.com", "Claude"]
        ], 
    flagging_mode="never"
    )
view.launch(inbrowser=True)