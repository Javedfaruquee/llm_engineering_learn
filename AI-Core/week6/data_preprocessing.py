"""Rewrite every curated product's text into a short, uniform summary with an LLM.

The raw `full` text from data_curation.py is long and noisy (up to 4,000 characters of
descriptions, feature lists and JSON details). This step asks gpt-oss-20b on Groq to turn
each one into five labelled lines (Title, Category, Brand, Description, Details), which is
what the price-prediction models are trained on.

Pipeline:
    1. Load the curated items from the HuggingFace Hub (lite or full, see LITE_MODE).
    2. Summarise one item with a normal API call, to check the output format and the cost.
    3. Summarise all items with the Groq Batch API (50% cheaper, results within 24 hours),
       1,000 items per batch, polling until every batch has finished.
    4. Push the summarised items back to the Hub with the same train / validation / test split.

Needs GROQ_API_KEY, and HF_TOKEN with write access, in the environment or a .env file.
Batch files are written to ./lite or ./full and progress is saved to ./batches.pkl, so a
re-run after an interruption resumes the submitted batches instead of paying for them again.
"""

import time

from dotenv import load_dotenv

from pricer.batch import Batch, state as BATCH_STATE_FILE
from pricer.items import Item
from pricer.preprocessor import Preprocessor


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

LITE_MODE = True                            # True: 22k items (minutes, cents); False: 820k items

HF_USERNAME = "ed-donner"                   # change to your own HuggingFace username
SOURCE_DATASET = f"{HF_USERNAME}/items_raw_lite" if LITE_MODE else f"{HF_USERNAME}/items_raw_full"
FULL_DATASET = f"{HF_USERNAME}/items_full"
LITE_DATASET = f"{HF_USERNAME}/items_lite"

# Split sizes; the test split is whatever remains (1,000 lite / 10,000 full)
TRAIN_SIZE = 20_000 if LITE_MODE else 800_000
VAL_SIZE = 1_000 if LITE_MODE else 10_000

POLL_SECONDS = 300                          # wait between checks on the submitted batches
MAX_WAIT_HOURS = 25                         # Groq's 24h completion window plus a margin


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_items():
    """Load the curated items from the Hub as one list and give each a numeric id.

    The three splits are concatenated in train, validation, test order. push_datasets()
    slices them back at the same boundaries, so every item stays in its original split.

    The id is the item's position in the list. It is sent with each batch request as the
    `custom_id`, which is how a batch result is matched back to its item.

    Returns:
        All Items, with `id` set and `summary` still empty.
    """
    train, val, test = Item.from_hub(SOURCE_DATASET)
    items = train + val + test
    for index, item in enumerate(items):
        item.id = index

    print(f"Loaded {len(items):,} items from {SOURCE_DATASET}")
    print(items[0])
    return items


def try_one_item(item):
    """Summarise a single item with a normal API call and print the result and its cost.

    A cheap sanity check before submitting thousands of requests: if the summary format or
    the cost per item looks wrong, stop here and adjust pricer/preprocessor.py.

    Args:
        item: Any Item with `full` text.
    """
    preprocessor = Preprocessor()
    print(preprocessor.preprocess(item.full))
    print()
    print(f"Input tokens: {preprocessor.total_input_tokens}")
    print(f"Output tokens: {preprocessor.total_output_tokens}")
    print(f"Cost: {preprocessor.total_cost * 100:.3f} cents")


def start_batches(items):
    """Submit all items to the Groq Batch API, or resume batches already submitted.

    First run: splits the items into batches of Batch.BATCH_SIZE (1,000), writes one JSONL
    request file per batch, uploads it and starts the batch, then saves the batch ids to
    BATCH_STATE_FILE.

    Re-run: loads the saved batches instead of submitting them again, and re-applies the
    results already downloaded for finished batches (summaries live only in memory, so the
    freshly loaded items don't have them yet). Delete BATCH_STATE_FILE to start over, e.g.
    after switching LITE_MODE.

    Args:
        items: All Items from load_items(); summaries are written into them in place.
    """
    if BATCH_STATE_FILE.exists():
        Batch.load(items)
        for batch in Batch.batches:
            if batch.done:
                batch.apply_output()
        return

    Batch.create(items, LITE_MODE)
    Batch.run()
    Batch.save()


def wait_for_batches():
    """Poll the submitted batches until all have finished or MAX_WAIT_HOURS has passed.

    Each poll asks Groq for the status of every unfinished batch, then downloads and applies
    the results of those that completed (Batch.fetch). Progress is saved after each poll so
    an interrupted run can resume. Small batches often finish within minutes; Groq only
    guarantees 24 hours.

    Returns:
        True if every batch finished, False if the wait timed out.
    """
    deadline = time.monotonic() + MAX_WAIT_HOURS * 3600
    while True:
        Batch.fetch()
        Batch.save()
        if all(batch.done for batch in Batch.batches):
            return True
        if time.monotonic() > deadline:
            return False
        print(f"Checking again in {POLL_SECONDS // 60} minutes...")
        time.sleep(POLL_SECONDS)


def missing_summaries(items):
    """Return the ids of items that have no summary.

    An item can be missing one if its batch failed or expired, or if a single request in an
    otherwise finished batch errored.
    """
    return [item.id for item in items if not item.summary]


def push_datasets(items):
    """Drop the working fields and push the summarised items to the Hub.

    `full` (the long raw text) and `id` (only needed to match batch results) are cleared
    to keep the published dataset small; `summary` replaces `full` as the model input.

    Splits are cut at the same boundaries as the source dataset:
      - full mode: 800,000 / 10,000 / 10,000 -> FULL_DATASET, then the lite subset below
      - both modes: 20,000 / 1,000 / 1,000 -> LITE_DATASET (the first rows of each split)
    Pushing overwrites an existing dataset of the same name.

    Args:
        items: All Items, each with a summary.
    """
    for item in items:
        item.full = None
        item.id = None

    train = items[:TRAIN_SIZE]
    val = items[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test = items[TRAIN_SIZE + VAL_SIZE:]
    if not LITE_MODE:
        Item.push_to_hub(FULL_DATASET, train, val, test)

    # In lite mode the splits are already this size, so this pushes them whole
    Item.push_to_hub(LITE_DATASET, train[:20_000], val[:1_000], test[:1_000])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the whole preprocessing pipeline, from loading items to pushing the summaries."""
    load_dotenv(override=True)

    # Step 1: load the curated items
    items = load_items()

    # Step 2: check the prompt and cost on one item
    try_one_item(items[0])

    # Step 3: summarise everything with the Batch API
    start_batches(items)
    if not wait_for_batches():
        print(f"Gave up after {MAX_WAIT_HOURS} hours; re-run this script later to resume.")
        return

    # Don't publish a dataset with holes in it
    missing = missing_summaries(items)
    if missing:
        print(f"{len(missing):,} items have no summary, e.g. ids {missing[:20]}; not pushing.")
        return

    # Step 4: publish
    push_datasets(items)


if __name__ == "__main__":
    main()
