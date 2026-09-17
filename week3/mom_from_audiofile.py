import gc
import importlib.util
import os
import sys
from pathlib import Path

import soundfile as sf
import soxr
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer,
    logging as transformers_hf_logging,
    pipeline,
)

# Minutes of a meeting from an audio file: Whisper transcribes, a chat model writes the
# minutes.
#
# The course notebook uses Llama 3.2 3B. It is gated, and its licence is separate from
# 3.1's and has not been accepted on this account (GatedRepoError). The 3.1 8B that is
# in the cache runs, but only with ~22 GB of free commit: unquantized it is 16 GB on the
# GPU plus a 5 GB shard mapped at a time, and with WSL/Docker holding its usual 35 GB the
# load dies with "The paging file is too small" (os error 1455). Phi-4-mini is public,
# close to the notebook's 3B in size, and loads in 7.7 GB.
WHISPER = "openai/whisper-medium.en"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Meta-Llama-3.1-8B-Instruct"
PHI4 = "microsoft/Phi-4-mini-instruct"
MINUTES_MODEL = PHI4
OPENAI_AUDIO_MODEL = "gpt-4o-mini-transcribe"

# Also transcribe with OpenAI, to compare against Whisper. Off by default: it uploads
# the audio and is billed per minute, and it needs `pip install openai` plus
# OPENAI_API_KEY in .env. The minutes are always written from the Whisper transcript.
COMPARE_WITH_OPENAI = False

# The Denver council extract from the course:
# https://drive.google.com/file/d/1N_kpSojRR5RYzupz6nqM8hMSoEF_R7pU/view?usp=sharing
# Any other recording can be passed as the first argument instead.
DEFAULT_AUDIO = Path(__file__).parent / "denver_extract.mp3"

# The sample rate Whisper was trained on. The pipeline resamples to it if the audio
# is at a different rate, but the resampling is slow and the model is more accurate
# when the audio is at the rate it was trained on.
WHISPER_SAMPLE_RATE = 16_000

# Suppress the "Using pad_token, but it is not set yet." warning from the tokenizer.
transformers_hf_logging.set_verbosity_error()

# The transcript and the minutes can hold any character. An interactive Windows console
# already uses UTF-8, but redirecting this script to a file falls back to the ANSI code
# page, where the first curly quote would die with UnicodeEncodeError.
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


def load_audio(audio_path):
    # Handing the pipeline a filename makes it shell out to ffmpeg, and handing it audio
    # at the wrong rate makes it resample through torchaudio. Neither is installed here
    # (and torchaudio must not be: pip resolves a build against a different ROCm than
    # torch's). libsndfile decodes mp3 by itself, and soxr resamples with a proper
    # low-pass filter - plain decimation of 44.1 kHz would alias into the speech band.
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # Whisper is mono
    if sample_rate != WHISPER_SAMPLE_RATE:
        audio = soxr.resample(audio, sample_rate, WHISPER_SAMPLE_RATE)
    print(f"Loaded {audio_path.name}: {len(audio) / WHISPER_SAMPLE_RATE / 60:.1f} minutes")
    return audio


def transcribe(audio_path):
    # Bound up front so the finally can del it even if the load raises.
    pipe = None
    try:
        pipe = pipeline("automatic-speech-recognition", model=WHISPER, dtype=dtype, device=device)
        report_vram("After loading Whisper")
        # Whisper sees 30 seconds at a time. chunk_length_s cuts the recording into
        # overlapping 30s windows that run through the model as one batch, instead of
        # return_timestamps=True walking the file sequentially: same text, several
        # times faster on a GPU.
        result = pipe(
            {"raw": load_audio(audio_path), "sampling_rate": WHISPER_SAMPLE_RATE},
            chunk_length_s=30,
            batch_size=8,
        )
        return result["text"].strip()
    finally:
        # Whisper has to be gone before the minutes model loads. Drop the reference first:
        # empty_cache() only frees blocks nothing points at.
        del pipe
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        report_vram("After Whisper cleanup")


def transcribe_with_openai(audio_path):
    from openai import OpenAI

    # The constructor reads OPENAI_API_KEY from the environment.
    with open(audio_path, "rb") as audio_file:
        return OpenAI().audio.transcriptions.create(
            model=OPENAI_AUDIO_MODEL, file=audio_file, response_format="text"
        ).strip()


system_message = """
You produce minutes of meetings from transcripts, with summary, key discussion points,
takeaways and action items with owners, in markdown format without code blocks.
"""


def write_minutes(transcription, max_new_tokens=2000):
    user_prompt = f"""
Below is an extract transcript of a Denver council meeting.
Please write minutes in markdown without code blocks, including:
- a summary with attendees, location and date
- discussion points
- takeaways
- action items with owners

Transcription:
{transcription}
"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
      ]

    tokenizer = inputs = model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(MINUTES_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Llama doesn't have a pad token, so use eos instead
        # add_generation_prompt=True appends the empty assistant header. Without it the
        # prompt ends on the user turn and the model may carry on writing as the user.
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        print(f"Prompt tokens: {inputs['input_ids'].shape[1]}")

        # 4-bit NF4 needs bitsandbytes, which is in .venv but has no build for
        # .venv-rocm. Without it load the weights as-is in dtype.
        if importlib.util.find_spec("bitsandbytes"):
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4"
            )
            model = AutoModelForCausalLM.from_pretrained(MINUTES_MODEL, device_map="auto", quantization_config=quant_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(MINUTES_MODEL, device_map="auto", dtype=dtype)
        report_vram(f"After loading {MINUTES_MODEL}")

        # Send the inputs where the model actually landed rather than a hardcoded "cuda",
        # and pass the attention mask too: pad == eos, so without it generate() can't
        # tell padding from a real end-of-sequence.
        inputs = inputs.to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        # Greedy: minutes should report the meeting, and be the same on every run.
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, streamer=streamer)

        # outputs holds prompt + reply. Decode only the reply, or the minutes start with
        # the whole transcript and the chat markup.
        prompt_len = inputs['input_ids'].shape[1]
        return tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True).strip()
    finally:
        del model, inputs, tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        report_vram("After minutes model cleanup")


audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_AUDIO
if not audio_path.exists():
    sys.exit(f"Audio file not found: {audio_path}")

# This is a script, not a notebook: display(Markdown(...)) has nothing to render into.
# Write the results next to the audio instead.
transcript_path = audio_path.with_name(f"{audio_path.stem}_transcript.txt")
minutes_path = audio_path.with_name(f"{audio_path.stem}_minutes.md")

# Transcribing is the slow half, and its result doesn't change. Keep it on disk so that
# reworking the prompt doesn't pay for it again; delete the file to force a fresh run.
if transcript_path.exists():
    print(f"Reusing {transcript_path.name}")
    transcription = transcript_path.read_text(encoding="utf-8")
else:
    transcription = transcribe(audio_path)
    transcript_path.write_text(transcription, encoding="utf-8")
print(f"\n--- Whisper transcript ---\n{transcription}\n")

if COMPARE_WITH_OPENAI:
    print(f"--- {OPENAI_AUDIO_MODEL} transcript ---\n{transcribe_with_openai(audio_path)}\n")

print("--- Minutes ---")
minutes = write_minutes(transcription)
minutes_path.write_text(minutes, encoding="utf-8")
print(f"\nSaved to {minutes_path}")
