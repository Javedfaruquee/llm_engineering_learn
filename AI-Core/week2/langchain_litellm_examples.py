import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Markdown, display
from langchain_openai import ChatOpenAI
from litellm import completion

tell_a_joke = [
    {"role": "user", "content": "Tell a joke for a student on the journey to becoming an expert in LLM Engineering"},
]

#From Langchain
llm = ChatOpenAI(model="gpt-5-mini")
response = llm.invoke(tell_a_joke)
print((response.content))

#From LiteLLM
response_litellm = completion(model="openai/gpt-4.1", messages=tell_a_joke)
reply_litellm = response_litellm.choices[0].message.content
print("\n" + (reply_litellm))

#With Hamlet file and LiteLLM
with open("week2/hamlet.txt", "r", encoding="utf-8") as f:
    hamlet_text = f.read()

location = hamlet_text.find("Speak, man")
#print(f"Location of 'Speak, man': {location}")
#print(hamlet_text[location:location+100])

question = [{"role": "user", "content": "In Hamlet, when Laertes asks 'Where is my father?' what is the reply?"}]
response = completion(model="gemini/gemini-3.1-flash-lite", messages=question)

print("\n" + (response.choices[0].message.content))

print(f"Input tokens gemini : {response.usage.prompt_tokens}")
print(f"Output tokens gemini: {response.usage.completion_tokens}")
print(f"Cached tokens gemini: {response.usage.prompt_tokens_details.cached_tokens}")
print(f"Total cost gemini: {response._hidden_params["response_cost"]*100:.4f} cents")
