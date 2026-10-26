import os
import glob
from dotenv import load_dotenv
from pathlib import Path
import gradio as gr
from openai import OpenAI

# Setting up

load_dotenv(override=True)
openai_api_key = os.getenv('OPENAI_API_KEY')
if openai_api_key:
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set")

MODEL = "gpt-4.1-nano"
SEPARATOR = "================================================================="
openai = OpenAI()

knowledge = {}

KNOWLEDGE_BASE = Path(__file__).parent / "knowledge-base"
filenames = glob.glob(str(KNOWLEDGE_BASE / "employees" / "*"))

for filename in filenames:
    print(f"Loading knowledge from {os.path.basename(filename)}")
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

print(SEPARATOR)
#print(f"Loaded knowledge base with {len(knowledge)} entries. \n")
#print(SEPARATOR)
print(f"Knowledge base entries with Employees: {list(knowledge.keys())} \n")
#print(SEPARATOR)
#print(f"Knowledge for Lancaster: {knowledge['lancaster']} \n")
print(SEPARATOR)


filenames = glob.glob(str(KNOWLEDGE_BASE / "products" / "*"))

for filename in filenames:
    print(f"Loading knowledge from {os.path.basename(filename)}")
    name = Path(filename).stem.split(' ')[-1]
    with open(filename, "r", encoding="utf-8") as f:
        knowledge[name.lower()] = f.read()

print(SEPARATOR)
print(f"Knowledge base entries with Employees and Products: {list(knowledge.keys())} \n")
print(SEPARATOR)

SYSTEM_PREFIX = """
You represent Insurellm, the Insurance Tech company.
You are an expert in answering questions about Insurellm; its employees and its products.
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.
Relevant context:
"""

def get_relevant_context_simple(message):
    text = ''.join(ch for ch in message if ch.isalpha() or ch.isspace())
    words = text.lower().split()
    relevant_context = []
    for word in words:
        if word in knowledge:
            relevant_context.append(knowledge[word])
    return relevant_context


def additional_context(message):
    relevant_context = get_relevant_context_simple(message)
    if not relevant_context:
        result = "There is no additional context relevant to the user's question."
    else:
        result = "The following additional context might be relevant in answering the user's question:\n\n"
        result += "\n\n".join(relevant_context)
    return result

def chat(message, history):
    system_message = SYSTEM_PREFIX + additional_context(message)
    messages = [{"role": "system", "content": system_message}] + history + [{"role": "user", "content": message}]
    response = openai.chat.completions.create(model=MODEL, messages=messages)
    return response.choices[0].message.content

print(get_relevant_context_simple("Who is lancaster?"))
print(SEPARATOR)
print(get_relevant_context_simple("Who is Lancaster and what is carllm?"))
print(SEPARATOR)
print(SEPARATOR)
print(additional_context("Who is Alex Lancaster?"))

view = gr.ChatInterface(chat).launch(inbrowser=True)