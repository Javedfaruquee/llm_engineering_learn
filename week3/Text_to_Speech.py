import os
import gc
import io
import zipfile
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login, hf_hub_download
from transformers import pipeline
import soundfile as sf

# Read the API keys from the .env file
load_dotenv(override=True)

# Get the Hugging Face token from the environment variable. Both models here are
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


OUTPUT_PATH = "speech.wav"

synthesiser = None
try:
    # No auth kwarg here: the hub client picks up the login above on its own, and
    # transformers 5 dropped use_auth_token anyway.
    synthesiser = pipeline(
        "text-to-speech", "microsoft/speecht5_tts", device=device, dtype=dtype
    )
    report_vram("After loading")

    # The x-vector picks the voice. datasets 5.x dropped support for script-based
    # repos, and cmu-arctic-xvectors is one, so pull its archive from the hub and read
    # the .npy directly - sorted the same way the old loader script ordered its rows,
    # which keeps index 7306 pointing at the same slt voice.
    archive = hf_hub_download("Matthijs/cmu-arctic-xvectors", "spkrec-xvect.zip", repo_type="dataset")
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
    sf.write(OUTPUT_PATH, speech["audio"], speech["sampling_rate"])
    print(f"Saved to {OUTPUT_PATH} ({len(speech['audio']) / speech['sampling_rate']:.1f}s)")
finally:
    # Release the weights. Dropping the reference has to come first - empty_cache()
    # only returns blocks that nothing still points at.
    del synthesiser
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    report_vram("After cleanup")
