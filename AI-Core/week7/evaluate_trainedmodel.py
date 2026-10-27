"""Evaluate a fine-tuned Llama 3.2 3B (base model + LoRA adapter) as a price predictor.

Loads the quantised base model, applies the LoRA adapter that training_model.py pushed
to the HuggingFace Hub, and scores it on the test set, for comparison with
evaluate_basemodel.py.

Runs on a CUDA GPU (e.g. Google Colab T4 or better). Needs:
  - HF_TOKEN with access to meta-llama/Llama-3.2-3B and to the adapter repo,
  - util.py from the course repository, which provides evaluate():
        wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py
"""

from datasets import load_dataset
from peft import PeftModel
from transformers import set_seed

from pricer.models import hf_login, load_model, load_tokenizer, make_predictor, print_memory, quant_config
from util import evaluate


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner"                       # your HF name here!

LITE_MODE = False                           # which training run to evaluate (lite or full)

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# The training run to load: RUN_NAME is the timestamp training_model.py gave it.
# REVISION pins a specific commit (checkpoint) of the adapter repo; None means the latest.
if LITE_MODE:
    RUN_NAME = "2025-11-30_15.10.55-lite"
    REVISION = None
else:
    RUN_NAME = "2025-11-28_18.47.07"
    REVISION = "b19c8bfea3b6ff62237fbb0a8da9779fc12cefbd"

PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

QUANT_4_BIT = True                          # must match how the model was loaded for training
SEED = 42


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load the base model plus the fine-tuned adapter and evaluate on the test set."""
    hf_login()
    test = load_dataset(DATASET_NAME)["test"]
    print(test[0])

    tokenizer = load_tokenizer(BASE_MODEL)
    base_model = load_model(BASE_MODEL, quant_config(QUANT_4_BIT), tokenizer)

    # The adapter adds only the small LoRA matrices on top of the frozen base model
    fine_tuned_model = PeftModel.from_pretrained(base_model, HUB_MODEL_NAME, revision=REVISION)
    print_memory(fine_tuned_model)
    print(fine_tuned_model)

    set_seed(SEED)
    evaluate(make_predictor(fine_tuned_model, tokenizer), test)


if __name__ == "__main__":
    main()
