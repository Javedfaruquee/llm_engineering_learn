import requests
import os
import json

from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

dotenv_path = find_dotenv(usecwd=False)
load_dotenv(dotenv_path=dotenv_path, override=True)

google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not google_api_key:
    raise RuntimeError(
        f"GOOGLE_API_KEY was not found. Checked .env at: {dotenv_path or 'no .env file found'}"
    )

gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

response_gemini = gemini.chat.completions.create(
                  model="gemini-3.5-flash-lite",
                  messages=[
                            {"role": "user", 
                            "content": "Tell me a fun fact"}
                            ]
            )

print("\n" + "Response from Google is: " + "\n\n" + response_gemini.choices[0].message.content)