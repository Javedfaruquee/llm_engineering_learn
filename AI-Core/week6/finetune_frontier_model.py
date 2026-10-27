"""Fine-tune gpt-4.1-nano on product prices with the OpenAI fine-tuning API.

The earlier scripts asked frontier models for prices zero-shot. Here a small frontier model
is shown 100 priced examples first, to see whether a little task-specific training beats
much larger models used as-is.

Pipeline:
    1. Load the summarised items from the HuggingFace Hub.
    2. Write 100 training and 50 validation examples as chat-format JSONL files.
    3. Upload them and start a fine-tuning job (1 epoch), or resume a job already started.
    4. Wait for the job to finish, printing its progress events.
    5. Evaluate the fine-tuned model on the test set with pricer.evaluator.

Needs HF_TOKEN and OPENAI_API_KEY in the environment or a .env file. Training is billed by
OpenAI per token trained; with 100 short examples and 1 epoch it costs a few cents.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login
from openai import OpenAI

from pricer.evaluator import evaluate
from pricer.items import Item


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

LITE_MODE = False                           # True: 20k training items; False: 800k

HF_USERNAME = "ed-donner"                   # change to your own HuggingFace username
DATASET = f"{HF_USERNAME}/items_lite" if LITE_MODE else f"{HF_USERNAME}/items_full"

BASE_MODEL = "gpt-4.1-nano-2025-04-14"
SEED = 42

# OpenAI suggests 50-100 examples to start with; our examples are short, so use 100
# training examples and 1 epoch
TRAIN_EXAMPLES = 100
VALIDATION_EXAMPLES = 50

JSONL_FOLDER = Path("jsonl")
TRAIN_FILE = JSONL_FOLDER / "fine_tune_train.jsonl"
VALIDATION_FILE = JSONL_FOLDER / "fine_tune_validation.jsonl"
JOB_FILE = JSONL_FOLDER / "fine_tune_job.txt"          # id of the job started by this script

POLL_SECONDS = 60                           # wait between checks on the fine-tuning job

PROMPT = "Estimate the price of this product. Respond with the price, no explanation"

# Read .env before creating the client, which needs OPENAI_API_KEY
load_dotenv(override=True)
openai = OpenAI()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items():
    """Log in to HuggingFace and load the summarised train / validation / test splits.

    Returns:
        (train, val, test) lists of Items.
    """
    # If you see a "Note" about the environment variable being set, it can be ignored
    login(os.environ["HF_TOKEN"], add_to_git_credential=True)

    train, val, test = Item.from_hub(DATASET)
    print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
    return train, val, test


# ---------------------------------------------------------------------------
# Prompts and training files
# ---------------------------------------------------------------------------

def test_messages_for(item):
    """The question for one item: the prompt followed by its summary.

    This is what the fine-tuned model is asked at inference time.
    """
    return [{"role": "user", "content": f"{PROMPT}\n\n{item.summary}"}]


def messages_for(item):
    """A complete training example: the question plus the ideal answer, e.g. "$129.99".

    Training on the price alone teaches the model to reply in exactly that format.
    """
    return test_messages_for(item) + [{"role": "assistant", "content": f"${item.price:.2f}"}]


def make_jsonl(items):
    """Convert items into OpenAI's chat fine-tuning format: one JSON object per line.

    Each line looks like {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}.

    Returns:
        The JSONL text, without a trailing newline.
    """
    return "\n".join(json.dumps({"messages": messages_for(item)}) for item in items)


def write_jsonl(items, path):
    """Write items as a JSONL training file at path, creating the folder if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(make_jsonl(items), encoding="utf-8")

--
def upload_file(path):
    """Upload a JSONL file to OpenAI for fine-tuning.

    Returns:
        The id of the uploaded file, to pass to the fine-tuning job.
    """
    with path.open("rb") as f:
        return openai.files.create(file=f, purpose="fine-tune").id


# ---------------------------------------------------------------------------
# Fine-tuning job
# ---------------------------------------------------------------------------

def start_job(train, val):
    """Start a fine-tuning job, or return the one this script started before.

    First run: writes the first TRAIN_EXAMPLES training and VALIDATION_EXAMPLES validation
    items to JSONL, uploads both files, creates the job and saves its id to JOB_FILE.
    Re-run: reuses the job id in JOB_FILE, so an interrupted run resumes instead of
    paying for a second fine-tune. Delete JOB_FILE to train again.

    Hyperparameters: 1 epoch and batch size 1 (the model sees each example once and
    updates after each one); a fixed seed makes the job repeatable. The "pricer" suffix
    appears in the fine-tuned model's name.

    Args:
        train: Training Items.
        val: Validation Items; OpenAI reports validation loss on them during training.

    Returns:
        The fine-tuning job id.
    """
    if JOB_FILE.exists():
        job_id = JOB_FILE.read_text(encoding="utf-8").strip()
        print(f"Resuming fine-tuning job {job_id}")
        return job_id

    print("Example training line:")
    print(make_jsonl(train[:1]))

    write_jsonl(train[:TRAIN_EXAMPLES], TRAIN_FILE)
    write_jsonl(val[:VALIDATION_EXAMPLES], VALIDATION_FILE)

    job = openai.fine_tuning.jobs.create(
        training_file=upload_file(TRAIN_FILE),
        validation_file=upload_file(VALIDATION_FILE),
        model=BASE_MODEL,
        seed=SEED,
        hyperparameters={"n_epochs": 1, "batch_size": 1},
        suffix="pricer",
    )
    JOB_FILE.write_text(job.id, encoding="utf-8")
    print(f"Started fine-tuning job {job.id}")
    return job.id


def wait_for_job(job_id):
    """Poll the job until it finishes, printing each new progress event once.

    Events include file validation, training steps with their loss, and completion. Jobs
    usually spend a few minutes queued before training starts.

    Args:
        job_id: The fine-tuning job id.

    Returns:
        The fine-tuned model's name, e.g. "ft:gpt-4.1-nano-2025-04-14:org::pricer:abc123".

    Raises:
        RuntimeError: if the job failed or was cancelled.
    """
    seen = set()
    while True:
        # Events come newest first; print the unseen ones in chronological order
        events = openai.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=10).data
        for event in reversed(events):
            if event.id not in seen:
                seen.add(event.id)
                print(event.message)

        job = openai.fine_tuning.jobs.retrieve(job_id)
        if job.status == "succeeded":
            return job.fine_tuned_model
        if job.status in ("failed", "cancelled"):
            raise RuntimeError(f"Fine-tuning job {job_id} {job.status}: {job.error}")
        time.sleep(POLL_SECONDS)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def make_fine_tuned_pricer(model_name):
    """Build a pricer that asks the fine-tuned model for each item's price.

    max_tokens=7 is enough for a reply like "$1234.56" and stops the model from rambling.
    The function name becomes the chart title: "GPT 4.1 Nano Fine Tuned".

    Args:
        model_name: The fine-tuned model's name from wait_for_job().

    Returns:
        The pricer function. It returns the raw reply text; the evaluator parses the number.
    """
    def gpt_4__1_nano_fine_tuned(item):
        """Ask the fine-tuned model for this item's price; returns its reply, e.g. "$129.99"."""
        response = openai.chat.completions.create(
            model=model_name,
            messages=test_messages_for(item),
            max_tokens=7,
        )
        return response.choices[0].message.content

    return gpt_4__1_nano_fine_tuned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Fine-tune the model (or resume the job), then evaluate it on the test set."""
    train, val, test = load_items()

    # Steps 2-4: train
    job_id = start_job(train, val)
    model_name = wait_for_job(job_id)
    print(f"Fine-tuned model: {model_name}")

    # Step 5: try one item, then evaluate
    pricer = make_fine_tuned_pricer(model_name)
    print(f"Predicted {pricer(test[0])!r} for an item that actually costs {test[0].price}")
    evaluate(pricer, test)


if __name__ == "__main__":
    main()
