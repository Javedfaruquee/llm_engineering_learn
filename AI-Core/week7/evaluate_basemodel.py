"""Evaluate the base Llama 3.2 3B model, before any fine-tuning, as a price predictor.

This is the starting point that fine-tuning (training_model.py) has to improve on. The
base model has never been trained for this task: it simply continues each test prompt,
which ends with "Price is $", and the evaluator reads the number it writes.

Runs on a CUDA GPU (e.g. Google Colab T4 or better). Needs:
  - HF_TOKEN with access to meta-llama/Llama-3.2-3B (Colab Secrets or environment),
  - util.py from the course repository, which provides evaluate():
        wget -q https://raw.githubusercontent.com/ed-donner/llm_engineering/main/week7/util.py -O util.py
"""

from datasets import load_dataset
from transformers import set_seed

from pricer.models import hf_login, load_model, load_tokenizer, make_predictor, print_memory, quant_config
from util import evaluate


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-3B"

LITE_MODE = True                            # True: lite dataset; the test split is the same size either way

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

QUANT_4_BIT = True                          # False loads in 8-bit: larger but closer to full quality
SEED = 42                                   # the base model samples its output, so fix the seed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load the quantised base model and evaluate it on the test set."""
    hf_login()
    test = load_dataset(DATASET_NAME)["test"]

    tokenizer = load_tokenizer(BASE_MODEL)
    base_model = load_model(BASE_MODEL, quant_config(QUANT_4_BIT), tokenizer)
    print_memory(base_model)

    # Try one item: the prompt and the base model's continuation
    model_predict = make_predictor(base_model, tokenizer)
    print(test[0])
    print(model_predict(test[0]))

    set_seed(SEED)
    evaluate(model_predict, test)


if __name__ == "__main__":
    main()
