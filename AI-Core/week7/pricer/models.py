"""Shared helpers for loading a quantised Llama model and using it as a pricer.

Used by evaluate_basemodel.py, evaluate_tainedmodel.py, qlora.py and training_model.py,
which all log in to HuggingFace, load the same base model with bitsandbytes
quantisation, and (for the evaluations) turn the model into a predict function.

Works on Google Colab (secrets from the Colab Secrets panel) and locally (secrets from
environment variables or a .env file). A CUDA GPU is required.
"""

import gc
import os

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ---------------------------------------------------------------------------
# Secrets and login
# ---------------------------------------------------------------------------

def get_secret(name):
    """Return a secret such as HF_TOKEN.

    On Colab it is read from the Secrets panel (key icon in the left sidebar); elsewhere
    from the environment, after loading a .env file if python-dotenv is installed.
    """
    try:
        from google.colab import userdata
        return userdata.get(name)
    except ImportError:
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        return os.environ[name]


def hf_login():
    """Log in to HuggingFace, needed for the gated Llama model and private repos.

    If you see a "Note" about the environment variable being set, it can be ignored.
    """
    login(get_secret("HF_TOKEN"), add_to_git_credential=True)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

def supports_bf16():
    """True if the GPU computes natively in bfloat16 (compute capability 8+: A100, L4, RTX 30xx+).

    Older GPUs such as Colab's T4 need float16 instead.
    """
    return torch.cuda.get_device_capability()[0] >= 8


def compute_dtype():
    """The dtype used for calculations on quantised weights: bfloat16 when supported, else float16."""
    return torch.bfloat16 if supports_bf16() else torch.float16


def quant_config(four_bit=True):
    """bitsandbytes quantisation settings for loading the base model.

    4-bit (QLoRA): NF4 weights ("normal float", suited to normally distributed weights)
    with double quantisation, which also compresses the per-block scaling constants.
    A 3B model shrinks from about 12.9 GB in float32 to about 2.2 GB.
    8-bit: about half the size of float16, with less quality loss than 4-bit.

    Args:
        four_bit: True for 4-bit, False for 8-bit.

    Returns:
        A BitsAndBytesConfig to pass to from_pretrained().
    """
    if four_bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype(),
            bnb_4bit_quant_type="nf4",
        )
    return BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=compute_dtype())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name):
    """Load the tokenizer, using the end-of-sequence token for padding.

    Llama has no padding token of its own. Padding on the right keeps each prompt's
    tokens at the start of the sequence, which is what training expects.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(model_name, quantization=None, tokenizer=None):
    """Load a causal language model across the available GPU(s).

    Args:
        model_name: HuggingFace model id, e.g. "meta-llama/Llama-3.2-3B".
        quantization: A BitsAndBytesConfig, or None for the model's full precision.
        tokenizer: If given, generation uses its padding token (avoids a warning).

    Returns:
        The model, placed automatically on GPU (and CPU if it doesn't fit).
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization,
        device_map="auto",
    )
    if tokenizer is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def print_memory(model, label="Memory footprint"):
    """Print how much memory the model's weights take, in GB."""
    print(f"{label}: {model.get_memory_footprint() / 1e9:,.2f} GB")


def clear_gpu_memory():
    """Release GPU memory held by models that are no longer referenced.

    Delete your own references first (`del model`), then call this.
    """
    gc.collect()
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def make_predictor(model, tokenizer, max_new_tokens=8):
    """Build a predict function for the evaluator from a model and its tokenizer.

    The dataset prompts end with "Price is $", so the model only has to continue with
    the number; 8 new tokens is enough for a price like "1234.00".

    Args:
        model: Base or fine-tuned causal language model.
        tokenizer: Its tokenizer.
        max_new_tokens: Upper limit on generated tokens.

    Returns:
        model_predict(item) -> the generated text that follows the item's prompt.
    """
    def model_predict(item):
        """Generate the continuation of item["prompt"], i.e. the predicted price."""
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Keep only the new tokens, not the echoed prompt
        prompt_length = inputs["input_ids"].shape[1]
        return tokenizer.decode(output_ids[0, prompt_length:])

    return model_predict
