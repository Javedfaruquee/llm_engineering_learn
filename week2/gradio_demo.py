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

# Define this variable and then pass js=force_dark_mode when creating the Interface

force_dark_mode = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

# Let's wrap a call to GPT-4.1-mini in a simple function

system_message = "You are a helpful assistant"

def message_gpt(prompt):
    messages = [
                {"role": "system", "content": system_message}, 
                {"role": "user", "content": prompt}
            ]
    response = openai.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content

def shout(text):
    print(f"Shout has been called with input {text}")
    return text.upper()

# Let's create a call that streams back results
# If you'd like a refresher on Generators (the "yield" keyword),
# Please take a look at the Intermediate Python guide in the guides folder

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

print(message_gpt("What is today's date?"))


#gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True)

#gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never").launch(inbrowser=True, auth=("ed", "bananas"))
#gr.Interface(fn=shout, inputs="textbox", outputs="textbox", flagging_mode="never", js=force_dark_mode).launch(inbrowser=True,)

# And now - changing the function from "shout" to "message_gpt"
message_input = gr.Textbox(label="Your message:", info="Enter a message for LLM", lines=7)
message_selector = gr.Dropdown(["GPT", "Claude"], label="Select Model:", value="GPT")
message_output = gr.Markdown(label="Response:")

gr.Interface(fn=stream_model, 
            title="LLMs Chat Demo", 
            inputs=[message_input, message_selector], 
            outputs=message_output, 
            examples=[
                      ["Explain the Transformer architecture to a layperson"],
                      ["Explain the Transformer architecture to an aspiring AI engineer"]   
                     ], 
            flagging_mode="never"
            ).launch(inbrowser=True)
