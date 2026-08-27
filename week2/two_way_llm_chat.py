import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from IPython.display import Markdown, display
from langchain_openai import ChatOpenAI
from litellm import completion


load_dotenv(override=True)

# Load API keys from environment variables for OpenAI, Anthropic, and Google Gemini
api_key = os.getenv('OPENAI_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY') 
google_api_key = os.getenv('GOOGLE_API_KEY')

# For Gemini, DeepSeek and Groq, we can use the OpenAI python client
# Because Google and DeepSeek have endpoints compatible with OpenAI
# And OpenAI allows you to change the base_url
anthropic_url = "https://api.anthropic.com/v1/"
gemini_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
# deepseek_url = "https://api.deepseek.com"
# groq_url = "https://api.groq.com/openai/v1"
# grok_url = "https://api.x.ai/v1"
# openrouter_url = "https://openrouter.ai/api/v1"
# ollama_url = "http://localhost:11434/v1"

# Initialize clients for OpenAI, Anthropic, and Google Gemini
openai = OpenAI(api_key=api_key)
anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
gemini = OpenAI(api_key=google_api_key, base_url=gemini_url)
# deepseek = OpenAI(api_key=deepseek_api_key, base_url=deepseek_url)
# groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
# grok = OpenAI(api_key=grok_api_key, base_url=grok_url)
# openrouter = OpenAI(base_url=openrouter_url, api_key=openrouter_api_key)
# ollama = OpenAI(api_key="ollama", base_url=ollama_url)

# Let's make a conversation between GPT-4.1-mini and Claude-haiku-4.5
# We're using cheap versions of models so the costs will be minimal
# models: gpt-4.1-mini, claude-haiku-4.5
gpt_model = "gpt-4.1-mini"
claude_model = "claude-haiku-4-5"
gemini_model = "gemini-3.5-flash-lite"

# Define system messages for both models to set their behavior
gpt_system = "You are a chatbot who is very argumentative; \
you disagree with anything in the conversation and you challenge everything, in a snarky way. \
Keep your responses short and witty."

claude_system = "You are a very polite, courteous chatbot. You try to agree with \
everything the other person says, or find common ground. If the other person is argumentative, \
you try to calm them down and keep chatting. \
Keep your responses short and polite."


# Define initial messages for both models
gpt_messages = ["Hi there"]
claude_messages = ["Hi"]

# Define functions to call GPT and Claude models
def call_gpt():
    messages = [{"role": "system", "content": gpt_system}]
    for gpt, claude in zip(gpt_messages, claude_messages):
        messages.append({"role": "assistant", "content": gpt})
        messages.append({"role": "user", "content": claude})
    response = openai.chat.completions.create(model=gpt_model, messages=messages)
    return response.choices[0].message.content

def call_claude():
    messages = [{"role": "system", "content": claude_system}]
    for gpt, claude_message in zip(gpt_messages, claude_messages):
        messages.append({"role": "user", "content": gpt})
        messages.append({"role": "assistant", "content": claude_message})
    messages.append({"role": "user", "content": gpt_messages[-1]})
    response = anthropic.chat.completions.create(model=claude_model, messages=messages)
    return response.choices[0].message.content

# Start the conversation 
print((f"### GPT:\n{gpt_messages[0]}\n"))
print((f"### Claude:\n{claude_messages[0]}\n"))

# Run the conversation for 5 turns
for i in range(5):
    gpt_next = call_gpt()
    print((f"### GPT:\n{gpt_next}\n"))
    gpt_messages.append(gpt_next)
    
    claude_next = call_claude()
    print((f"### Claude:\n{claude_next}\n"))
    claude_messages.append(claude_next)
