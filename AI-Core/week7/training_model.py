"""Fine-tune Llama 3.2 3B to predict prices with QLoRA, and push the adapter to the Hub.

QLoRA = the base model is loaded quantised to 4 bits and frozen; only small LoRA adapter
matrices added to selected layers are trained. That fits a 3B model's training on a
single Colab GPU.

The dataset rows are {"prompt": "...Price is $", "completion": "123.00"}. The SFT trainer
learns to produce the completion given the prompt.

Progress is logged to Weights & Biases, and the adapter is pushed to a private
HuggingFace repo at every checkpoint, so a disconnected Colab session loses little.

Runs on a CUDA GPU (lite mode: T4 is enough; full mode: A100 recommended). Needs
HF_TOKEN (write access) and WANDB_API_KEY, in Colab Secrets or the environment.
"""

import os
from datetime import datetime

import wandb
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from pricer.models import get_secret, hf_login, load_model, load_tokenizer, print_memory, quant_config, supports_bf16


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "price"
HF_USER = "ed-donner"                       # your HF name here!

LITE_MODE = True                            # True: 20k items, quick run; False: 800k items

DATA_USER = "ed-donner"
DATASET_NAME = f"{DATA_USER}/items_prompts_lite" if LITE_MODE else f"{DATA_USER}/items_prompts_full"

# Each run gets a timestamped name, used for the W&B run and the Hub repo
RUN_NAME = f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
if LITE_MODE:
    RUN_NAME += "-lite"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
HUB_MODEL_NAME = f"{HF_USER}/{PROJECT_RUN_NAME}"

# Hyper-parameters - overall
EPOCHS = 1 if LITE_MODE else 3
BATCH_SIZE = 32 if LITE_MODE else 256
MAX_SEQUENCE_LENGTH = 128                   # tokens; prompts were trimmed to fit
GRADIENT_ACCUMULATION_STEPS = 1

# Hyper-parameters - QLoRA
QUANT_4_BIT = True
LORA_R = 32 if LITE_MODE else 256           # rank: size of the adapter matrices
LORA_ALPHA = LORA_R * 2                     # scaling of the adapter's contribution
ATTENTION_LAYERS = ["q_proj", "v_proj", "k_proj", "o_proj"]
MLP_LAYERS = ["gate_proj", "up_proj", "down_proj"]
TARGET_MODULES = ATTENTION_LAYERS if LITE_MODE else ATTENTION_LAYERS + MLP_LAYERS
LORA_DROPOUT = 0.1

# Hyper-parameters - training
LEARNING_RATE = 1e-4
WARMUP_RATIO = 0.01                         # first 1% of steps ramp the learning rate up
LR_SCHEDULER_TYPE = "cosine"
WEIGHT_DECAY = 0.001
OPTIMIZER = "paged_adamw_32bit"             # pages optimizer state to CPU on memory spikes

# Tracking
VAL_SIZE = 500 if LITE_MODE else 1000       # validation rows scored at each evaluation
LOG_STEPS = 5 if LITE_MODE else 10
SAVE_STEPS = 100 if LITE_MODE else 200      # checkpoint, push to Hub and evaluate this often
LOG_TO_WANDB = True


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_wandb():
    """Log in to Weights & Biases and start a run named RUN_NAME in project PROJECT_NAME.

    Only metrics are logged: model checkpoints and gradient histograms are switched off
    to save time and storage.
    """
    os.environ["WANDB_API_KEY"] = get_secret("WANDB_API_KEY")
    wandb.login()

    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)


def load_data():
    """Load the prompt/completion dataset.

    Returns:
        (train, val): all training rows, and the first VAL_SIZE validation rows
        (scoring the whole validation split at every checkpoint would be slow).
    """
    dataset = load_dataset(DATASET_NAME)
    return dataset["train"], dataset["val"].select(range(VAL_SIZE))


def lora_config():
    """Which layers get LoRA adapters, and how large they are.

    Lite mode adapts only the attention projections with rank 32; full mode adds the
    MLP layers and uses rank 256 for more capacity. See qlora.py for the parameter counts.
    """
    return LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_R,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )


def training_config():
    """Training, logging, checkpointing and Hub settings for the SFT trainer.

    Notable choices:
      - bf16 on GPUs that support it (A100), fp16 otherwise (T4),
      - gradients clipped at 0.3 for stability,
      - group_by_length batches prompts of similar length together, wasting less padding,
      - every SAVE_STEPS steps: save a checkpoint, push it to HUB_MODEL_NAME (private),
        and evaluate on the validation rows; the last 10 checkpoints are kept locally.
    """
    use_bf16 = supports_bf16()
    return SFTConfig(
        output_dir=PROJECT_RUN_NAME,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIMIZER,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        logging_steps=LOG_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        fp16=not use_bf16,
        bf16=use_bf16,
        max_grad_norm=0.3,
        max_steps=-1,                       # -1: run for num_train_epochs
        warmup_ratio=WARMUP_RATIO,
        group_by_length=True,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="wandb" if LOG_TO_WANDB else "none",
        run_name=RUN_NAME,
        max_length=MAX_SEQUENCE_LENGTH,
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=True,
        hub_model_id=HUB_MODEL_NAME,
        hub_private_repo=True,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Load data and the 4-bit base model, train the LoRA adapter, and push it to the Hub."""
    print(f"bf16 supported: {supports_bf16()}")             # A100 supports this; T4 does not natively

    hf_login()
    if LOG_TO_WANDB:
        setup_wandb()

    train, val = load_data()

    tokenizer = load_tokenizer(BASE_MODEL)
    base_model = load_model(BASE_MODEL, quant_config(QUANT_4_BIT), tokenizer)
    print_memory(base_model)

    # The trainer wraps the frozen base model with the LoRA adapters and trains only those
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=train,
        eval_dataset=val,
        peft_config=lora_config(),
        args=training_config(),
    )
    fine_tuning.train()

    # Final push of the trained adapter (only the LoRA weights, not the base model)
    fine_tuning.model.push_to_hub(PROJECT_RUN_NAME, private=True)
    print(f"Saved to the hub: {PROJECT_RUN_NAME}")

    if LOG_TO_WANDB:
        wandb.finish()


if __name__ == "__main__":
    main()
