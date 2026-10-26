import gc
import importlib.util
import os
import sys
import threading

import gradio as gr
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
    logging as transformers_hf_logging,
)

# Three ways to get text out of a chat model, then a Gradio chat on top of the streaming ones:
#   1- full response         generate(), nothing to see until the last token is done
#   2- low level streaming   our own token-by-token loop around model()
#   3- high level streaming  generate() in a thread, feeding a TextIteratorStreamer
#
# The course notebook uses Llama 3.1 8B in 4-bit. Unquantized it is 16 GB on the GPU plus
# a 5 GB shard mapped at a time, and with WSL/Docker holding its usual 35 GB of commit the
# load dies with "The paging file is too small" (os error 1455). Phi-4-mini is public and
# loads in 7.7 GB. Any other chat model can be passed as the first argument instead:
#   python various_streaming_modes.py meta-llama/Meta-Llama-3.1-8B-Instruct
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI4 = "microsoft/Phi-4-mini-instruct"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else PHI4

SYSTEM_MESSAGE = {"role": "system", "content": "You are a helpful assistant"}
MAX_NEW_TOKENS = 2000

transformers_hf_logging.set_verbosity_error()

# The replies can hold any character. An interactive Windows console already uses UTF-8,
# but redirecting this script to a file falls back to the ANSI code page, where the
# first emoji would die with UnicodeEncodeError.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Read the API keys from the .env file
load_dotenv(override=True)

# Get the Hugging Face token from the environment variable. login(None) falls through
# to the interactive browser flow, so only call it when we actually have one; without
# it the hub client still picks up a token cached by an earlier `hf auth login`.
hf_token = os.getenv('HF_TOKEN')
if hf_token:
    login(hf_token, add_to_git_credential=True)

# ROCm builds expose the AMD GPU through the torch.cuda API (HIP presents as CUDA),
# so "cuda" here covers both the NVIDIA and the Radeon 8060S case.
# CPU is the fallback, and it needs float32: fp16 math is unsupported there.
#
# bfloat16 over float16 on the GPU: these checkpoints were trained in bf16, and fp16's
# narrower exponent range degrades them.
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
    print("No GPU backend found - running on CPU, expect this to take a long while")


def report_vram(label):
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        print(f"{label}: {(total - free) / 1024**3:.1f} GB used of {total / 1024**3:.1f} GB")


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Llama doesn't have a pad token, so use eos instead

    # 4-bit NF4 needs bitsandbytes, which is in .venv but has no build for
    # .venv-rocm. Without it load the weights as-is in dtype.
    if importlib.util.find_spec("bitsandbytes"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=quant_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", dtype=dtype)
    model.eval()
    return tokenizer, model


def build_inputs(messages):
    # return_dict=True gives input_ids plus the attention mask: pad == eos on Llama, so
    # without the mask generate() can't tell padding from a real end-of-sequence.
    # .to(model.device) rather than a hardcoded "cuda": it follows wherever device_map
    # put the model, CPU included.
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    return inputs.to(model.device)


# Every generating function takes the whole conversation and leaves it untouched; the
# caller owns the history. They used to append to a global `messages` and never add the
# reply, so each call stacked another unanswered user turn onto the next one's prompt.

def generate_full(messages, max_new_tokens=MAX_NEW_TOKENS):
    inputs = build_inputs(messages)
    # Greedy in all three modes, so they can be compared: same prompt, same reply.
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # outputs holds prompt + reply. Decode only the reply.
    prompt_len = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)


# inference_mode: without it every forward pass records an autograd graph that is never
# used, and memory climbs with each token. As a decorator rather than a `with` block
# because this is a generator: grad mode is thread-local, Gradio may resume the generator
# on a different worker thread, and the decorator re-enters the mode around every resume.
@torch.inference_mode()
def generate_stream_low_level(messages, max_new_tokens=MAX_NEW_TOKENS):
    # What generate() does inside, written out: forward pass, pick a token, feed it back.
    # Yields the reply so far after every token.
    #
    # A chat model ends its turn with its own marker (<|eot_id|> on Llama, <|end|> on
    # Phi), which is not always tokenizer.eos_token_id. generation_config lists them all.
    eos = model.generation_config.eos_token_id
    stop_ids = set(eos if isinstance(eos, list) else [eos])

    next_input = build_inputs(messages)["input_ids"]
    past_key_values = None
    generated = []
    for _ in range(max_new_tokens):
        # The KV cache is what keeps this linear. The first pass runs the whole prompt;
        # after that only the one new token goes in, and attention reads the keys and
        # values of everything before it from the cache. Feeding the full, growing
        # sequence back in each time recomputes all of it for every single token.
        outputs = model(input_ids=next_input, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values

        # logits is [batch, sequence, vocab]. [:, -1] is the prediction for the position
        # after the last token; argmax picks the likeliest one (greedy decoding), and
        # keepdim leaves it as [batch, 1], ready to be the next input.
        next_input = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
        token_id = next_input.item()
        if token_id in stop_ids:
            break
        generated.append(token_id)

        # Decode everything generated so far, not the one new token: an emoji or a
        # Devanagari letter spans several tokens, and each of them alone decodes to "�".
        yield tokenizer.decode(generated, skip_special_tokens=True)


def generate_stream_high_level(messages, max_new_tokens=MAX_NEW_TOKENS):
    # Same result, but generate() does the loop. TextStreamer would print to stdout,
    # which is no use to a Gradio app; TextIteratorStreamer hands the text to whoever
    # iterates it. generate() blocks until it is done, hence the thread.
    inputs = build_inputs(messages)

    # skip_special_tokens goes in directly - the streamer passes its extra kwargs on to
    # tokenizer.decode(). Wrapped as decode_kwargs={...} it is silently ignored, and
    # <|eot_id|> then has to be stripped from the text by hand.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    def run():
        try:
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)
        except BaseException:
            # generate() only closes the streamer on success. If it dies (out of memory,
            # say) the loop below would wait on the queue forever.
            streamer.end()
            raise

    thread = threading.Thread(target=run)
    thread.start()

    reply = ""
    for text_chunk in streamer:
        reply += text_chunk
        yield reply
    thread.join()


def print_stream(stream):
    # The generators yield the reply so far; print only what is new.
    # flush=True because print output is buffered by default, and would otherwise
    # arrive in bursts instead of token by token.
    printed = 0
    for reply in stream:
        # Hold back a trailing "�": it is a character still waiting for its other tokens.
        if not reply.endswith("�"):
            print(reply[printed:], end="", flush=True)
            printed = len(reply)
    print()


def message_text(content):
    # Gradio 6 hands history content over as a list of {"type": "text", "text": ...}
    # parts; older versions as a plain string.
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict))


def chat(user_input, history):
    # Gradio keeps the history per browser session and passes it in, so two people
    # chatting at once no longer write into one shared global conversation.
    messages = [SYSTEM_MESSAGE]
    messages += [{"role": turn["role"], "content": message_text(turn["content"])} for turn in history]
    messages.append({"role": "user", "content": user_input})
    # Swap in generate_stream_low_level to watch the hand-written loop drive the UI.
    yield from generate_stream_high_level(messages)


# Bound up front so the finally can del them even if the load raises.
tokenizer = model = None
try:
    tokenizer, model = load_model(MODEL_NAME)
    report_vram(f"After loading {MODEL_NAME}")

    demo_messages = [
        SYSTEM_MESSAGE,
        {"role": "user", "content": "What is the meaning of life? Answer in markdown and in 5 lines maximum."},
    ]

    print("\n--- 1- full response ---")
    print(generate_full(demo_messages))

    print("\n--- 2- low level streaming ---")
    print_stream(generate_stream_low_level(demo_messages))

    print("\n--- 3- high level streaming ---")
    print_stream(generate_stream_high_level(demo_messages))

    # launch() blocks until Ctrl+C. Gradio runs one chat() at a time by default
    # (concurrency_limit=1), which is what a single GPU wants.
    demo = gr.ChatInterface(chat, title="Chat with AI (Streaming Enabled)")
    demo.launch()
finally:
    # Drop the references first: empty_cache() only frees blocks nothing points at.
    del model, tokenizer
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    report_vram("After cleanup")
