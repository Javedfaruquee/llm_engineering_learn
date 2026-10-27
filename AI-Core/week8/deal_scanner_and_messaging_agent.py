"""Week 8, Day 3: find deals in RSS feeds (ScannerAgent) and send push alerts (MessagingAgent).

Steps:
    1. Scrape the latest deals from several deal-site RSS feeds.
    2. Ask GPT-6 Luna, with structured outputs, to pick the 5 deals with the most detailed
       description and a clear price, and to summarise each product.
    3. Do the same through the ScannerAgent, which packages steps 1-2.
    4. Send a test push notification, then a deal alert written by an LLM, through the
       MessagingAgent and Pushover.

Pushover setup (free): sign up at https://pushover.net, click "Create an Application/API
Token", and install the app on your phone. Then add to .env:
    PUSHOVER_USER=u...    (the user key on your Pushover home screen)
    PUSHOVER_TOKEN=a...   (the API token of the application you created)

Needs OPENAI_API_KEY, plus ANTHROPIC_API_KEY for the MessagingAgent's alert text.
"""

import logging
import os

from dotenv import load_dotenv

from agents.deals import DealSelection, ScrapedDeal
from agents.messaging_agent import MessagingAgent
from agents.scanner_agent import ScannerAgent


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def fetch_deals():
    """Scrape the current deals from the RSS feeds and show one as an example.

    Returns:
        A list of ScrapedDeal, each with a title, details, features and URL.
    """
    deals = ScrapedDeal.fetch(show_progress=True)
    print(f"Fetched {len(deals)} deals")
    if deals:
        print(deals[min(10, len(deals) - 1)].describe())
    return deals


def select_deals(scanner, deals):
    """Ask the scanner's model to choose and summarise the 5 best-described deals.

    This is what ScannerAgent.scan() does internally, step by step. Structured outputs
    (`parse` with response_format=DealSelection) make the reply a typed object instead
    of free text. The prompts warn about "$XXX off" wording, which is a discount, not
    the product's price.

    Args:
        scanner: A ScannerAgent, used for its prompts, model and OpenAI client.
        deals: ScrapedDeals from fetch_deals().

    Returns:
        A DealSelection with up to 5 deals (description, price, URL).
    """
    user_prompt = scanner.make_user_prompt(deals)
    print(user_prompt[:2000])                       # how the prompt begins

    messages = [
        {"role": "system", "content": ScannerAgent.SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    response = scanner.openai.chat.completions.parse(
        model=ScannerAgent.MODEL, messages=messages, response_format=DealSelection, reasoning_effort="none"
    )
    return response.choices[0].message.parsed


def print_selection(selection):
    """Print each selected deal's summary, price and URL."""
    for deal in selection.deals:
        print(deal.product_description)
        print(deal.price)
        print(deal.url)
        print()


def check_pushover_keys():
    """Report whether the Pushover keys are set, showing only their first character."""
    for name in ("PUSHOVER_USER", "PUSHOVER_TOKEN"):
        value = os.getenv(name)
        print(f"{name} found and starts with {value[0]}" if value else f"{name} not found")


def send_notifications():
    """Send a plain test push, then an LLM-written alert about an example deal."""
    agent = MessagingAgent()
    agent.push("SUCH A MASSIVE DEAL!!")
    agent.notify("A special deal on Samsung 60 inch LED TV going at a great bargain", 300, 1000, "www.samsung.com")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Scan for deals by hand and with the agent, then test notifications."""
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")    # show the agents' logs

    scanner = ScannerAgent()
    print_selection(select_deals(scanner, fetch_deals()))

    # The same again, in one call through the agent
    print(scanner.scan())

    check_pushover_keys()
    send_notifications()


if __name__ == "__main__":
    main()
