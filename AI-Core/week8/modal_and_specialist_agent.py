"""Week 8, Day 1: run models in the cloud with Modal, and meet the SpecialistAgent.

The Price is Right - week 8 order of play:
    Day 1: Modal.com and SpecialistAgent            <- this script
    Day 2: RAG, FrontierAgent, EnsembleAgent
    Day 3: ScannerAgent, MessagingAgent
    Day 4: AutonomousPlanningAgent and DealAgentFramework
    Day 5: The Price Is Right finale (user interface)

Steps:
    1. Check the console can print UTF-8 (Modal prints emojis).
    2. Hello world on Modal: the same function run locally, in the cloud, and in an EU region.
    3. Run Llama 3.2 on a cloud GPU.
    4. Price a product with the fine-tuned model as an ephemeral app (exists only while running).
    5. Rewrite the product text with the Preprocessor, as the training data was rewritten.
    6. Deploy the pricer as a persistent service (a function, then a class that stays loaded).
    7. Price through the SpecialistAgent, which wraps the deployed service.

Setup (once):
  - Sign up at https://modal.com, then Settings > API Tokens > New Token and run the
    command it shows (`modal token set --token-id ak-... --token-secret as-...`), or put
    MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in .env. On Windows, if Modal can't find the
    token, move the generated .modal.toml into this folder.
  - In the Modal dashboard, create a Secret named "huggingface-secret" with key HF_TOKEN
    and your HuggingFace token as the value. The service files refer to it by that name.
  - The Preprocessor uses Llama 3.2 through Ollama by default; to use Groq instead, add
    PRICER_PREPROCESSOR_MODEL=groq/openai/gpt-oss-20b to .env.

Modal bills GPU time: the first deployment builds an image and can take ~10 minutes.
"""

import locale
import logging
import os
import subprocess
import sys

import modal
from dotenv import load_dotenv

from agents.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PRODUCT = "Quadcast HyperX condenser mic, connects via usb-c to your computer for crystal clear audio"
LLAMA_PROMPT = "Never gonna give you up, never gonna"      # or "Hey Jude, don't make it"

# Preprocessor models to compare; the last one's rewrite is what gets priced
PREPROCESSOR_MODELS = [None, "groq/openai/gpt-oss-20b"]     # None = the default model

# Deployed service names, as defined in pricer_service.py / pricer_service2.py
SERVICE_APP = "pricer-service"

# How long the deployed Pricer stays warm after a call, in seconds. None leaves Modal's
# default (2 minutes). 1200 keeps it warm for 20 minutes - faster replies, more credits.
KEEP_WARM_SECONDS = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_on_modal(app, function, *args, local=False):
    """Run a Modal function inside a temporary ("ephemeral") run of its app.

    Args:
        app: The modal.App the function belongs to.
        function: The decorated Modal function.
        *args: Arguments for the function.
        local: True runs it on this machine instead of in the cloud, for comparison.

    Returns:
        The function's return value.
    """
    with modal.enable_output():             # show Modal's build and run progress
        with app.run():
            return function.local(*args) if local else function.remote(*args)


def deploy(module):
    """Deploy a Modal app permanently, like `modal deploy -m <module>` at a terminal.

    A deployed app keeps running after this script ends and can be called by name
    from any Python program, which is how the agents use it.
    """
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}   # Modal prints emojis
    subprocess.run([sys.executable, "-m", "modal", "deploy", "-m", module], check=True, env=environment)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def check_encoding():
    """Make sure output can contain emojis and other special characters.

    Should print UTF-8; also sets PYTHONIOENCODING for any child processes.
    """
    print(f"Preferred encoding: {locale.getpreferredencoding()}")
    os.environ["PYTHONIOENCODING"] = "utf-8"


def say_hello():
    """Run hello.py's function locally, in Modal's cloud, and in Modal's EU region.

    Each reports the city its code ran in, so you can see where it executed. Choosing
    a region costs marginally more credits.
    """
    from hello import app, hello, hello_europe

    print(run_on_modal(app, hello, local=True))
    print(run_on_modal(app, hello))
    print(run_on_modal(app, hello_europe))


def generate_with_llama():
    """Ask Llama 3.2 3B, running on a cloud T4 GPU, to continue LLAMA_PROMPT."""
    # This import may warn about adding local Python modules to the Image; that's safe to ignore
    from llama import app, generate

    print(run_on_modal(app, generate, LLAMA_PROMPT))


def price_ephemeral(description):
    """Price a product with the fine-tuned model, loaded fresh in a temporary app.

    Slow: every call starts a container and loads the model from scratch.
    """
    from pricer_ephemeral import app, price

    result = run_on_modal(app, price, description)
    print(f"Ephemeral pricer: ${result:,.2f} for {description[:60]!r}")
    return result


def preprocess(description):
    """Rewrite the product text with each model in PREPROCESSOR_MODELS.

    The fine-tuned model was trained on LLM-rewritten summaries, so it prices those
    more accurately than raw text. Models that aren't available (e.g. Ollama not
    running) are reported and skipped.

    Returns:
        The last successful rewrite, or the original text if none worked.
    """
    text = description
    for model in PREPROCESSOR_MODELS:
        preprocessor = Preprocessor(model_name=model) if model else Preprocessor()
        try:
            text = preprocessor.preprocess(description)
        except Exception as error:
            print(f"Preprocessor {preprocessor.model_name} failed: {error}")
            continue
        print(f"--- {preprocessor.model_name} ---\n{text}\n")
    return text


def price_with_deployed_services(text):
    """Deploy the pricer as a service, then call it by name - first as a function, then a class.

    pricer_service.py: a deployed function; each call still loads the model (slow).
    pricer_service2.py: a deployed class that loads the model once per container and
    caches the weights on a Modal volume, so warm calls take ~2 seconds (a cold start
    ~30 seconds). Both deploy to the same app name, so the second replaces the first.
    """
    deploy("pricer_service")
    pricer = modal.Function.from_name(SERVICE_APP, "price")
    print(f"Deployed function: ${pricer.remote(text):,.2f}")     # watch it at https://modal.com

    deploy("pricer_service2")
    Pricer = modal.Cls.from_name(SERVICE_APP, "Pricer")
    pricer = Pricer()
    print(f"Deployed class, first call: ${pricer.price.remote(text):,.2f}")
    print(f"Deployed class, warm call: ${pricer.price.remote(text):,.2f}")

    if KEEP_WARM_SECONDS:
        pricer.update_autoscaler(scaledown_window=KEEP_WARM_SECONDS)
        print(f"Pricer will stay warm for {KEEP_WARM_SECONDS} seconds after each call")


def price_with_specialist_agent():
    """Price a product through SpecialistAgent, the agent wrapper around the deployed Pricer."""
    from agents.specialist_agent import SpecialistAgent

    agent = SpecialistAgent()
    print(f"Specialist Agent: ${agent.price('iPhone 10'):,.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the day's steps in order."""
    load_dotenv(override=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")    # show the agents' logs

    check_encoding()
    say_hello()
    generate_with_llama()
    price_ephemeral(PRODUCT)
    text = preprocess(PRODUCT)
    price_ephemeral(text)
    price_with_deployed_services(text)
    price_with_specialist_agent()


if __name__ == "__main__":
    main()
