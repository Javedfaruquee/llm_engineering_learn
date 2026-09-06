import gc
import io
import os
import sys
import zipfile
import tokenizers
import numpy as np
import soundfile as sf
import torch
from diffusers import AutoPipelineForText2Image
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as transformers_hf_logging,
    pipeline,
)


# Model names for the demos. Llama is gated and needs an accepted licence plus HF_TOKEN;
# the rest are public.
#
# Note the -Instruct suffix on Llama. The plain "meta-llama/Meta-Llama-3.1-8B" is the
# base model: pretrained on raw text, never taught turn-taking, so it ships no
# chat_template and apply_chat_template() below raises. Same vocabulary either way, so
# the encode/decode demos are unaffected by using the instruct variant throughout.
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI4 = "microsoft/Phi-4-mini-instruct"
DEEPSEEK = "deepseek-ai/DeepSeek-V3.1"
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Drop the per-model LOAD REPORTs. Every one of them so far has been UNEXPECTED
# keys - a BERT pooler head that token classification never builds, a positional
# encoding buffer that gets rebuilt at init - which means extra weights discarded,
# not weights left random. Real errors, and any MISSING-key report, still print.
transformers_hf_logging.set_verbosity_error()

# Devanagari needs UTF-8 on the way out. An interactive Windows console already uses
# it, but redirecting this script to a file falls back to the ANSI code page, where
# the Hindi translation would die with UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Read the API keys from the .env file
load_dotenv(override=True)

# Get the Hugging Face token from the environment variable. Every model here is
# public, so an unset token is fine - but login(None) falls through to the interactive
# browser flow, so only call it when we actually have one.
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token, add_to_git_credential=True)

# ROCm builds expose the AMD GPU through the torch.cuda API (HIP presents as CUDA),
# so "cuda" here covers both the NVIDIA and the Radeon 8060S case.
# CPU is the fallback, and it needs float32: fp16 math is unsupported there.
if torch.cuda.is_available():
    device, dtype = "cuda", torch.float16
    backend = f"ROCm {torch.version.hip}" if torch.version.hip else f"CUDA {torch.version.cuda}"
    print(f"Running on {torch.cuda.get_device_name(0)} via {backend}")
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
    print("Running on Apple MPS")
else:
    device, dtype = "cpu", torch.float32
    print("No GPU backend found - running on CPU, expect 1-3 minutes")


def report_vram(label):
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"{label}: {(total - free) / 1024**3:.1f} GB used of {total / 1024**3:.1f} GB")


def run(label, demo):
    """Run one demo, then hand its VRAM back.

    Each demo holds its model in a local, so the reference is gone by the time the
    call returns - which is what empty_cache() needs, since it only releases blocks
    that nothing still points at. Ten models stacked in one process would otherwise
    sit on ~15 GB by the end.
    """
    print(f"\n=== {label} ===")
    demo()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    report_vram("after cleanup")

# Tokenizer demo LLAMA is gated, so it needs an accepted licence plus HF_TOKEN; the rest are public.
tokenizer = AutoTokenizer.from_pretrained(LLAMA, trust_remote_code=True)
text = "I am excited to show Tokenizers in action to my LLM engineers"
tokens = tokenizer.encode(text)
print(f"LLAMA tokenized text: {text}" + "\n" + f"Tokens: {tokens}" + "\n")
character_count = len(text)
word_count = len(text.split(' '))
token_count = len(tokens)
print(f"There are {character_count} characters, {word_count} words and {token_count} tokens"  + "\n")

tokenizer.decode(tokens)
print(f"Decoded tokens: {tokenizer.decode(tokens)}" + "\n")

tokenizer.batch_decode(tokens)
print(f"Batch decoded tokens: {tokenizer.batch_decode(tokens)}" + "\n")
# tokenizer.vocab
tokenizer.get_added_vocab()

len(tokenizer.vocab)
print(f"Vocabulary size: {len(tokenizer.vocab)}" + "\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Tell a light-hearted joke for a room of Data Scientists"}
  ]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(prompt)
print("\n")
print(f"Prompt: {prompt}" + "\n")
print("\n")
print("====================================================================================")
# The Phi 4 tokenizer is a different tokenizer than the Llama tokenizer, so we need to load it separately.
phi4_tokenizer = AutoTokenizer.from_pretrained(PHI4)

text = "I am curiously excited to show Hugging Face Tokenizers in action to my LLM engineers"
print("PHI4:")
tokens = tokenizer.encode(text)
print("\n")
print(tokens)
print("\n" + "TOKENIZER BATCH DECODE:")
print(tokenizer.batch_decode(tokens))
print("\nPhi 4 TOKINIZER ENCODE:")
tokens = phi4_tokenizer.encode(text)
print(tokens)
print("\n")
print("\n" + "PHI4 TOKENIZER BATCH DECODE:")
print(phi4_tokenizer.batch_decode(tokens))
print("====================================================================================")

# Llama and Phi 4 have different tokenization, so let's see how they handle a chat prompt.
print(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("\nPhi:")
print(phi4_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("\nDeepSeek:")
# DeepSeek defines its tokenizer class in the repo rather than in transformers, so this
# one needs trust_remote_code to let that code run.
deepseek_tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK, trust_remote_code=True)
print(deepseek_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
print("\n")
print("====================================================================================")

# Qwen Coder is a code generation model, so let's see how it tokenizes some Python code.
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_CODER)
code = """
def hello_world(person):
  print("Hello", person)
"""
tokens = qwen_tokenizer.encode(code)
for token in tokens:
  print(f"{token}={qwen_tokenizer.decode(token)}")
print("====================================================================================")
print("\n")
# Generation - the first thing in this file that actually uses the GPU. Everything
# above is string-to-integer bookkeeping, which is pure CPU work: a tokenizer has no
# matrix maths to offload. Only once the ids reach a model is there anything to run.

def report_vram(label):
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"{label}: {(total - free) / 1024**3:.1f} GB used of {total / 1024**3:.1f} GB")


def generate_with_phi4():
    model = None
    try:
        # device_map places each shard as it loads, so the weights never take a detour
        # through CPU RAM the way .to(device) after a full load would.
        model = AutoModelForCausalLM.from_pretrained(PHI4, dtype=dtype, device_map=device)
        report_vram("After loading Phi-4-mini")

        # Same messages as the chat-template demos above, but tokenize=True this time:
        # the model wants the ids, not the marked-up string. return_dict also hands
        # back the attention_mask that generate() wants.
        inputs = phi4_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(device)
        print("\n")
        print(f"Prompt is {inputs['input_ids'].shape[-1]} tokens on {inputs['input_ids'].device}")

        outputs = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        # Slice the prompt off the front so only the newly generated tokens get decoded.
        reply = phi4_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )
        print(f"Phi-4-mini says: {reply}")
        print("====================================================================================")
    finally:
        # Release the weights. Dropping the reference has to come first - empty_cache()
        # only returns blocks that nothing still points at.
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        report_vram("After cleanup")


print("\n=== Generation on the GPU ===")
generate_with_phi4()