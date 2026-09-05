import os
import gc
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from diffusers import AutoPipelineForText2Image

# Read the API keys from the .env file
load_dotenv(override=True)


hf_token = os.getenv('HF_TOKEN')
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


pipe = None
try:
    # variant="fp16" just picks the smaller weight files (~7GB instead of ~14GB);
    # torch_dtype still controls the precision they are loaded at.
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=dtype, variant="fp16"
    )
    pipe.to(device)
    report_vram("After loading")

    prompt = "A class of students learning AI engineering in a vibrant pop-art style"
    image = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    image.save("students_popart.png")
    image.show()
    print("Saved to students_popart.png")
finally:
    # Release the weights. Dropping the reference has to come first - empty_cache()
    # only returns blocks that nothing still points at.
    del pipe
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    report_vram("After cleanup")
