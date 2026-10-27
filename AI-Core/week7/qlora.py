"""Explore QLoRA: how quantisation shrinks Llama 3.2 3B, and how small a LoRA adapter is.

Part 1 loads the base model three times (full precision, 8-bit, 4-bit) and prints the
memory each takes, then applies a fine-tuned LoRA adapter to the 4-bit model and prints
its architecture so you can see the lora_A / lora_B matrices.

Part 2 counts the adapter's parameters from the layer shapes, for the two settings used
by training_model.py (lite: r=32 on attention layers; full: r=256 on attention and MLP).

Runs on a CUDA GPU (e.g. Google Colab T4 or better). Needs HF_TOKEN with access to
meta-llama/Llama-3.2-3B, in Colab Secrets or the environment.
"""

from peft import PeftModel
from transformers import BitsAndBytesConfig

from pricer.models import clear_gpu_memory, hf_login, load_model, print_memory, quant_config


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-3B"
FINETUNED_MODEL = "ed-donner/price-2025-11-30_15.10.55-lite"     # a lite run: r=32, attention only

# Llama 3.2 3B shapes: (input size, output size) of each layer LoRA can adapt.
# k/v are smaller because 8 key/value heads are shared by 24 query heads.
LAYER_COUNT = 28
MODULE_SHAPES = {
    # Attention
    "q_proj": (3072, 3072),
    "k_proj": (3072, 1024),
    "v_proj": (3072, 1024),
    "o_proj": (3072, 3072),
    # MLP (feed-forward)
    "gate_proj": (3072, 8192),
    "up_proj": (3072, 8192),
    "down_proj": (8192, 3072),
}
ATTENTION_LAYERS = ["q_proj", "k_proj", "v_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]


# ---------------------------------------------------------------------------
# Part 1: memory at different precisions
# ---------------------------------------------------------------------------

def compare_precisions():
    """Load the base model in full precision, 8-bit and 4-bit, printing each one's size.

    Each model is freed before the next is loaded, so they don't have to fit in GPU
    memory together; the 4-bit model is kept and returned.

    Returns:
        The 4-bit base model.
    """
    model = load_model(BASE_MODEL)
    print_memory(model, "Full precision")
    del model
    clear_gpu_memory()

    model = load_model(BASE_MODEL, BitsAndBytesConfig(load_in_8bit=True))
    print_memory(model, "8-bit")
    del model
    clear_gpu_memory()

    model = load_model(BASE_MODEL, quant_config(four_bit=True))
    print_memory(model, "4-bit")
    return model


def show_adapter(base_model):
    """Apply the fine-tuned LoRA adapter and print the model's memory and architecture.

    In the printed architecture, each adapted layer gains a lora_A (input size x r) and a
    lora_B (r x output size) matrix - these are the shapes used in Part 2.
    """
    fine_tuned_model = PeftModel.from_pretrained(base_model, FINETUNED_MODEL)
    print_memory(fine_tuned_model, "4-bit + LoRA adapter")
    print(fine_tuned_model)


# ---------------------------------------------------------------------------
# Part 2: LoRA adapter size
# ---------------------------------------------------------------------------

def lora_parameter_count(r, modules):
    """Number of LoRA weights for rank r on the given modules across all layers.

    Each adapted layer gets lora_A (r x input) and lora_B (output x r), and its weights
    are adapted by adding alpha * B @ A - so it costs r * (input + output) parameters
    instead of input * output for the full matrix.

    Args:
        r: LoRA rank, the inner dimension of the two small matrices.
        modules: Module names from MODULE_SHAPES to adapt.
    """
    per_layer = sum(r * (n_in + n_out) for n_in, n_out in (MODULE_SHAPES[m] for m in modules))
    return per_layer * LAYER_COUNT


def print_lora_size(r, modules):
    """Print the adapter's parameter count and its size in MB, at 4 bytes (float32) per weight."""
    params = lora_parameter_count(r, modules)
    size_mb = params * 4 / 1_000_000
    print(f"r={r}, {', '.join(modules)}")
    print(f"Total number of params: {params:,} and size {size_mb:,.1f}MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Compare model sizes, inspect the adapter, then count adapter parameters."""
    hf_login()

    base_model = compare_precisions()
    show_adapter(base_model)

    print_lora_size(32, ATTENTION_LAYERS)                   # lite training settings
    print_lora_size(256, ATTENTION_LAYERS + MLP_LAYERS)     # full training settings


if __name__ == "__main__":
    main()
