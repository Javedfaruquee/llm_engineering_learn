import gc
import io
import os
import sys
import zipfile

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
    logging as transformers_hf_logging,
    pipeline,
)

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


# my_pipeline = pipeline(task, model=..., device=..., dtype=...)
# Sentiment Analysis

def sentiment_analysis():
    analyser = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device,
        dtype=dtype,
    )
    print(analyser("I should be more excited to be on the way to LLM mastery!!"))


# Named Entity Recognition

def named_entity_recognition():
    # aggregation_strategy="simple" stitches word pieces back together, so "Ed Donner"
    # arrives as one PER span instead of four B-/I- tagged fragments.
    ner = pipeline(
        "ner",
        model="dbmdz/bert-large-cased-finetuned-conll03-english",
        aggregation_strategy="simple",
        device=device,
        dtype=dtype,
    )
    result = ner(
        "AI Engineers are learning about the amazing pipelines from HuggingFace "
        "in Google Colab from Ed Donner"
    )
    for entity in result:
        print(f"Entity: {entity['word']}, Label: {entity['entity_group']}, "
              f"Score: {entity['score']:.2f}")


# Question Answering with Context
#
# transformers 5 removed the question-answering, summarization and translation
# pipelines. The models are all still here, so the next three demos call them
# directly - which is what the pipeline was wrapping anyway.

def question_answering():
    question = "What are Hugging Face pipelines?"
    context = "Pipelines are a high level API for inference of LLMs with common tasks"

    name = "distilbert/distilbert-base-cased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForQuestionAnswering.from_pretrained(name, dtype=dtype).to(device)

    inputs = tokenizer(question, context, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    # The model scores every token as a possible start and end of the answer span;
    # the answer is the slice between the two best-scoring positions.
    start = outputs.start_logits.argmax()
    end = outputs.end_logits.argmax() + 1
    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    print(f"Question: {question}, Answer: {answer}")


# Text Summarization

def summarization():
    text = """
The Hugging Face transformers library is an incredibly versatile and powerful tool for natural language processing (NLP).
It allows users to perform a wide range of tasks such as text classification, named entity recognition, and question answering, among others.
It's an extremely popular library that's widely used by the open-source data science community.
It lowers the barrier to entry into the field by providing Data Scientists with a productive, convenient way to work with transformer models.
"""
    name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=dtype).to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    summary_ids = model.generate(**inputs, max_length=50, min_length=25, do_sample=False)
    print(f"Summary: {tokenizer.decode(summary_ids[0], skip_special_tokens=True)}")


# Translation
# All translation models are here: https://huggingface.co/models?pipeline_tag=translation&sort=trending

def translate(name, language, target_code=None):
    """Translate one sentence into `language`.

    Two model families, two calling conventions. Marian (opus-mt) ships one model per
    language pair, so the pair is baked into the checkpoint and there is nothing to
    specify. NLLB is a single model covering 200 languages, so it has to be told both
    ends: the source goes on the tokenizer, and the target is forced as the first
    token the decoder emits - that is what target_code is.
    """
    sentence = ("The Data Scientists were truly amazed by the power and simplicity "
                "of the HuggingFace pipeline API.")
    tokenizer = AutoTokenizer.from_pretrained(
        name, **({"src_lang": "eng_Latn"} if target_code else {})
    )
    # These are fp32-trained models whose generate() can overflow to NaN in fp16, so
    # they stay at full precision. Marian is ~300 MB and NLLB ~2.4 GB, so it is cheap.
    model = AutoModelForSeq2SeqLM.from_pretrained(name, dtype=torch.float32).to(device)

    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    forced = (
        {"forced_bos_token_id": tokenizer.convert_tokens_to_ids(target_code)}
        if target_code else {}
    )
    tokens = model.generate(**inputs, max_new_tokens=60, **forced)
    print(f"{language}: {tokenizer.decode(tokens[0], skip_special_tokens=True)}")


# Classification

def zero_shot_classification():
    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        dtype=dtype,
    )
    result = classifier(
        "Hugging Face's Transformers library is amazing!",
        candidate_labels=["technology", "sports", "politics"],
    )
    print(f"Classification: {result['labels'][0]} with score {result['scores'][0]:.2f}")


# Text Generation

def text_generation():
    # Naming gpt2 explicitly: the task default is now SmolLM3-3B, a 6 GB download
    # that would dwarf everything else in this script.
    generator = pipeline(
        "text-generation", model="openai-community/gpt2", device=device, dtype=dtype
    )
    result = generator(
        "If there's one thing I want you to remember about using HuggingFace pipelines, it's",
        max_new_tokens=60,
        truncation=True,
    )
    print(f"Generated Text: {result[0]['generated_text']}")


# Image Generation - remember this?! Now you know what's going on
# Pipelines can be used for diffusion models as well as transformers

IMAGE_PATH = "pipelines_popart.png"

def image_generation():
    # variant="fp16" just picks the smaller weight files (~7GB instead of ~14GB);
    # dtype still controls the precision they are loaded at.
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", dtype=dtype, variant="fp16"
    ).to(device)
    prompt = "A class of students learning AI engineering in a vibrant pop-art style"
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    image.save(IMAGE_PATH)
    print(f"Saved to {IMAGE_PATH}")


# Audio Generation

AUDIO_PATH = "pipelines_speech.wav"


def text_to_speech():
    synthesiser = pipeline(
        "text-to-speech", "microsoft/speecht5_tts", device=device, dtype=dtype
    )

    # The x-vector picks the voice. datasets 5.x dropped support for script-based
    # repos, and cmu-arctic-xvectors is one, so pull its archive from the hub and read
    # the .npy directly - sorted the same way the old loader script ordered its rows,
    # which keeps index 7306 pointing at the same slt voice.
    archive = hf_hub_download(
        "Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset"
    )
    with zipfile.ZipFile(archive) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".npy"))
        xvector = np.load(io.BytesIO(z.read(names[7306])))

    # Match the model's precision, or the matmul against fp16 weights fails on GPU.
    speaker_embedding = torch.tensor(xvector).unsqueeze(0).to(dtype)

    speech = synthesiser(
        "Hi to an artificial intelligence engineer, on the way to mastery!",
        forward_params={"speaker_embeddings": speaker_embedding},
    )

    # This is a script, not a notebook: IPython's Audio() would just build a widget
    # that nothing renders. Write a wav file instead.
    sf.write(AUDIO_PATH, speech["audio"], speech["sampling_rate"])
    print(f"Saved to {AUDIO_PATH} ({len(speech['audio']) / speech['sampling_rate']:.1f}s)")


run("Sentiment Analysis", sentiment_analysis)
run("Named Entity Recognition", named_entity_recognition)
run("Question Answering", question_answering)
run("Summarization", summarization)
run("Translation (en to de)", lambda: translate("Helsinki-NLP/opus-mt-en-de", "German"))
run("Translation (en to es)", lambda: translate("Helsinki-NLP/opus-mt-en-es", "Spanish"))
run("Translation (en to hi)", lambda: translate("facebook/nllb-200-distilled-600M", "Hindi", "hin_Deva"))
run("Zero-shot Classification", zero_shot_classification)
run("Text Generation", text_generation)
run("Image Generation", image_generation)
run("Text to Speech", text_to_speech)
