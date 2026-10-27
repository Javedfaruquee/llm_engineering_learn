"""Run the base Llama 3.2 3B model on a Modal cloud GPU to continue a prompt.

Used by day1_modal_and_specialist_agent.py. Needs a Modal Secret named "huggingface-secret"
holding HF_TOKEN, since Llama is a gated model.
"""

import modal
from modal import Image

# Setup: the app, its container image, and the HuggingFace token from Modal's secrets
app = modal.App("llama")
image = Image.debian_slim().pip_install("torch", "transformers", "accelerate")
secrets = [modal.Secret.from_name("huggingface-secret")]
GPU = "T4"
MODEL_NAME = "meta-llama/Llama-3.2-3B"


@app.function(image=image, secrets=secrets, gpu=GPU, timeout=1800)
def generate(prompt: str) -> str:
    """Return the prompt followed by up to 5 tokens that Llama generates after it."""
    # Imported inside the function: these libraries exist in the container, not necessarily locally
    from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

    set_seed(42)                            # repeatable sampling
    inputs = tokenizer.encode(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0])
