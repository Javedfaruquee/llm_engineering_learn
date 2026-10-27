"""The fine-tuned pricer as a deployable Modal class that keeps the model loaded.

Deploy with `modal deploy -m pricer_service2`; then call it with
`modal.Cls.from_name("pricer-service", "Pricer")().price.remote(description)`.
This is the version the SpecialistAgent uses.

Faster than pricer_service.py because:
  - the model is loaded once when a container starts (@modal.enter), not on every call,
  - downloaded model weights are cached on a Modal Volume, so new containers skip the download.
A warm container answers in ~2 seconds; waking a cold one takes ~30 seconds.
"""

import modal
from modal import Volume, Image
# Setup - define our infrastructure with code!

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

# This collects the secret from Modal.
# Depending on your Modal configuration, you may need to replace "huggingface-secret" with "hf-secret"
secrets = [modal.Secret.from_name("huggingface-secret")]

# Constants: which fine-tuned adapter to load (the Hub repo and a pinned commit)
GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner"  # your HF name here! Or use mine if you just want to reproduce my results.
RUN_NAME = "2025-11-28_18.47.07"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"                        # where the volume is mounted inside the container

# Change this to 1 if you want Modal to be always running, otherwise it will go cold after 2 mins
# (an always-on container uses credits continuously)
MIN_CONTAINERS = 0

# Same prompt format as in training; the model completes the number after "Price is $"
PREFIX = "Price is $"
QUESTION = "What does this cost to the nearest dollar?"

# Persistent storage for the HuggingFace download cache, shared by all containers
hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)


@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Pricer:
    """Holds the fine-tuned model in GPU memory for as long as its container is alive."""

    @modal.enter()
    def setup(self):
        """Load the 4-bit base model, tokenizer and LoRA adapter once, when the container starts."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Quant Config: 4-bit NF4, as used for training (float16 maths - a T4 has no bfloat16)
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )

    @modal.method()
    def price(self, description: str) -> float:
        """Return the fine-tuned model's price estimate for a product description."""
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION}\n\n{description}\n\n{PREFIX}"

        # Generate a few tokens and pull the first number out of what follows "Price is $"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(inputs, max_new_tokens=5)
        result = self.tokenizer.decode(outputs[0])
        contents = result.split("Price is $")[1]
        contents = contents.replace(",", "")
        match = re.search(r"[-+]?\d*\.\d+|\d+", contents)
        return float(match.group()) if match else 0
