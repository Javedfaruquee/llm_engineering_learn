import gc
import os
import sys
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    logging as transformers_hf_logging,
    TextStreamer,
    BitsAndBytesConfig,
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
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
QWEN_CODER = "Qwen/Qwen2.5-Coder-7B-Instruct"
QWEN = "Qwen/Qwen3-4B-Instruct-2507"
GEMMA = "google/gemma-3-270m-it"


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
#
# bfloat16 over float16 on the GPU: these checkpoints were trained in bf16, and fp16's
# narrower exponent range degrades them. Measured on DeepSeek-R1-Distill 1.5B, fp16
# looped inside <think> until the token limit on every run; bf16 finished.
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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

# Tokenizer demo LLAMA is gated, so it needs an accepted licence plus HF_TOKEN; the rest are public.
messages = [
    {"role": "user", "content": "Tell a joke for a room of Data Scientists"}
  ]

# Quantization Config - this allows us to load the model into memory and use less memory

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# The model

def generate(model, messages, quant=True, max_new_tokens=80, **generate_kwargs):
    # model            Hub repo id, e.g. LLAMA. Comes in as a string and is rebound below
    #                  to the loaded model object; the tokenizer is fetched from the same
    #                  repo, since each family has its own vocabulary and chat template.
    #                    LLAMA, PHI4, DEEPSEEK, QWEN, GEMMA     the constants above
    #                    "Qwen/Qwen2.5-0.5B-Instruct"           any chat model on the Hub
    #                    r"C:\models\my-finetune"               a local save_pretrained() folder
    #                  It must be an instruct/chat variant: a base model has no chat
    #                  template, so apply_chat_template() raises.
    # messages         The conversation in chat format: a list of {"role", "content"}
    #                  dicts. apply_chat_template() turns it into the model's own prompt
    #                  markup (<|start_header_id|>, <｜User｜>, ...), so the same list works
    #                  for every model.
    #                    [{"role": "user", "content": "Tell a joke"}]            single question
    #                    [{"role": "system", "content": "Answer in one line."},  system prompt
    #                     {"role": "user", "content": "Tell a joke"}]            sets the behaviour
    #                    [{"role": "user", "content": "Tell a joke"},            multi-turn: earlier
    #                     {"role": "assistant", "content": "Why did ..."},       replies go back in, the
    #                     {"role": "user", "content": "Explain it"}]             model has no memory
    #                  The last message should be from "user". DeepSeek-R1 advises against a
    #                  system message: put the instructions in the user turn instead.
    # quant            True loads the weights in 4-bit NF4 through quant_config (needs
    #                  bitsandbytes, so .venv only). False loads them as-is in dtype:
    #                  bf16 on the GPU, float32 on CPU.
    #                    quant=True    Llama 8B: 5.6 GB instead of ~16 GB in 16-bit
    #                    quant=False   DeepSeek 1.5B: 3.6 GB in bf16, 7.1 GB in float32
    # max_new_tokens   Upper bound on tokens generated, not counting the prompt. It is a
    #                  ceiling, not a target: generation stops earlier when the model emits
    #                  its end-of-sequence token. Too low and the reply is cut off mid-sentence.
    #                    80      a one-liner; the Llama joke stopped by itself well under this
    #                    300     a few paragraphs
    #                    1500    a reasoning model, whose <think> block alone takes 300-600
    # generate_kwargs  Anything else is handed to model.generate() untouched, so decoding
    #                  can be tuned per call without touching this function.
    #                    do_sample=False            greedy: always the top token, repeatable
    #                    do_sample=True,            sample instead; temperature scales the
    #                      temperature=0.7          randomness (0.2 focused, 1.0+ wild)
    #                    top_p=0.9                  sample only from the tokens making up the
    #                                               top 90% of probability
    #                    top_k=50                   ... or only from the 50 likeliest tokens
    #                    repetition_penalty=1.1     down-weight tokens already used; 1.0 is
    #                                               off, above ~1.3 the text turns odd
    #                    no_repeat_ngram_size=3     never repeat any 3-token sequence
    #                    num_beams=4                beam search: keep 4 candidates, pick the
    #                                               best overall (cannot be streamed)
    #                  temperature, top_p and top_k only act when do_sample=True.
    #
    # Bound up front so the finally can del them even if the tokenizer load raises;
    # otherwise an UnboundLocalError there would mask the real error.
    tokenizer = inputs = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token  # Llama doesn't have a pad token, so use eos instead
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        print(f"Input tokens: {inputs['input_ids'].shape[1]}")
        print(f"Input IDs:\n{inputs['input_ids']}")
        # quant is the on/off flag; quant=False loads full precision, for comparing footprints.
        if quant:
            model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", quantization_config=quant_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", dtype=dtype)
        print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(model)
        memory = model.get_memory_footprint() / 1e6
        print(f"Memory footprint: {memory:,.1f} MB")
        report_vram("After load")
        # return_dict=True makes inputs a BatchEncoding (input_ids + attention_mask), not a
        # tensor, so unpack it. Passing the mask also matters here: pad == eos, so without
        # it generate() can't tell padding from a real end-of-sequence.
        inputs = inputs.to(model.device)
        
        # Stream tokens as they arrive. generate() only returns once every token is done,
        # which on CPU is minutes of silence that looks like a hang - worst with a
        # reasoning model like DeepSeek-R1, which spends hundreds of tokens in <think>.
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        # **inputs           Unpacks to input_ids=... and attention_mask=... . input_ids is
        #                    the prompt as token IDs; the mask marks which positions are
        #                    real tokens (1) rather than padding (0).
        #                      input_ids      = tensor([[128000, 128006, 9125, ...]])  shape (1, 44)
        #                      attention_mask = tensor([[1, 1, 1, ...]])               same shape
        #                    Equivalent to writing both out by hand:
        #                      model.generate(input_ids=inputs["input_ids"],
        #                                     attention_mask=inputs["attention_mask"], ...)
        # max_new_tokens     The cap described above. The loop runs one forward pass per
        #                    token, feeding each new token back in, until EOS or this cap.
        #                      max_new_tokens=80     at most 80 tokens after the prompt
        #                      max_length=200        the alternative: caps prompt + reply
        #                                            together; use one or the other
        # streamer           Called with every token as it is chosen; TextStreamer decodes
        #                    and prints it. skip_prompt=True keeps it from echoing the prompt.
        #                    It only affects what is printed - outputs is the same either way.
        #                      TextStreamer(tokenizer, skip_prompt=True)     print live to stdout
        #                      TextStreamer(tokenizer, skip_prompt=True,     ... without <|eot_id|>
        #                                   skip_special_tokens=True)        and similar markers
        #                      TextIteratorStreamer(tokenizer)               yields text chunks for
        #                                                                    a Gradio UI; generate()
        #                                                                    then runs in a thread
        #                      streamer=None                                 silent until finished
        # **generate_kwargs  The caller's decoding overrides. Anything not given here falls
        #                    back to the model's own generation_config.json (DeepSeek ships
        #                    do_sample=True, temperature=0.6, top_p=0.95, so its output
        #                    differs on every run).
        #                      generate(DEEPSEEK, messages, repetition_penalty=1.1)
        #                        -> model.generate(..., repetition_penalty=1.1)
        #                      generate(LLAMA, messages, do_sample=False)
        #                        -> model.generate(..., do_sample=False), same joke every run
        #                      generate(LLAMA, messages)
        #                        -> generate_kwargs is {}, the model's defaults apply
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, streamer=streamer, **generate_kwargs)
        
        # Unlike inputs, outputs is a plain tensor of token IDs, shape (batch, prompt + new).
        # Slice off the prompt to see only what the model produced.
        prompt_len = inputs['input_ids'].shape[1]
        print(f"\nGenerated {outputs.shape[1] - prompt_len} new tokens")
        print(f"Generated tokens:\n{outputs[0, prompt_len:]}")
       
        # Only outputs here: model, inputs and tokenizer are dropped in the finally, and
        # deleting them twice would raise UnboundLocalError there.
        del outputs
    finally:
        # Drop the reference first: empty_cache() only frees blocks nothing points at.
        del model, inputs, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        report_vram("After cleanup")

#generate(LLAMA, messages)
#generate(QWEN, messages)
#generate(GEMMA, messages, quant=False)
# R1 is a reasoning model: it writes a <think>...</think> monologue before the answer, so
# it needs a far bigger budget than the others - at 100 tokens it is cut off mid-thought
# and never reaches the joke. A 1.5B model also tends to circle inside <think>; a mild
# repetition penalty breaks the loop (finished in ~600 tokens in testing).
generate(DEEPSEEK, messages, quant=False, max_new_tokens=1500, repetition_penalty=1.1)