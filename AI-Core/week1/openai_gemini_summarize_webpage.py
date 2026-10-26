import os
import sys

from openai import OpenAI
from dotenv import find_dotenv, load_dotenv

from scraper import fetch_website_contents

# Windows consoles default to cp1252, which cannot print accented names or dashes.
sys.stdout.reconfigure(encoding="utf-8")

dotenv_path = find_dotenv(usecwd=False)
load_dotenv(dotenv_path=dotenv_path, override=True)

google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not google_api_key:
    raise RuntimeError(
        f"GOOGLE_API_KEY was not found. Checked .env at: {dotenv_path or 'no .env file found'}"
    )

gemini = OpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)

SYSTEM_PROMPT = (
    "You are an assistant that analyses the contents of a website and provides a short "
    "summary, ignoring navigation-related text. Respond in markdown."
)

MAX_CHARS = 20000


def user_prompt_for(website):
    return (
        f"You are looking at a website titled: {website.title}\n\n"
        "The contents of this website is as follows; please provide a short summary in "
        "markdown. If it includes news or announcements, summarise those too.\n\n"
        f"{website.text[:MAX_CHARS]}"
    )


def messages_for(website):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt_for(website)},
    ]


url = input("Enter a webpage URL to summarize: ").strip()
if not url:
    raise ValueError("A webpage URL is required.")

website = fetch_website_contents(url)

response_gemini = gemini.chat.completions.create(
    model="gemini-3.5-flash-lite",
    messages=messages_for(website),
)

print("\nSummary:\n\n" + response_gemini.choices[0].message.content)
