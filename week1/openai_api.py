import requests
import os
import json

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
 
openai = OpenAI()

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

payload = {
    "model": "gpt-5-nano",
    "messages": [
        {"role": "user", "content": "Tell me a fun fact"}]
}

response = requests.post(
    "https://api.openai.com/v1/chat/completions",
    headers=headers,
    json=payload
)

print(json.dumps(response.json(), indent=2))
print("\n" + response.json()["choices"][0]["message"]["content"])

response1 = openai.chat.completions.create(
     model="gpt-5-nano", 
     messages=[{"role": "user", 
                "content": "Tell me a fun fact"}
              ]
   )

print("\n" + response1.choices[0].message.content)


