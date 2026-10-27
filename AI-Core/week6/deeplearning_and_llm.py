"""Price predictors from a human, a simple neural network and frontier LLMs.

Continues traditional_ml_baseline.py with three more kinds of predictor, all scored by
pricer.evaluator on the test set (errors printed, two charts per predictor):

    1. Human: a person prices the first 100 test items by hand from a CSV file.
    2. Neural network: an 8-layer fully connected network trained on a bag of words.
    3. Frontier LLMs, zero-shot: each model is shown the product summary and asked for
       a price, with no training or examples.

Needs HF_TOKEN, plus OPENAI_API_KEY, ANTHROPIC_API_KEY and GEMINI_API_KEY for the LLMs,
in the environment or a .env file. The LLM step makes 200 paid API calls per model.
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from huggingface_hub import login
from litellm import completion
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from pricer.evaluator import evaluate
from pricer.items import Item


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

LITE_MODE = False                           # True: 20k training items; False: 800k

HF_USERNAME = "ed-donner"                   # change to your own HuggingFace username
DATASET = f"{HF_USERNAME}/items_lite" if LITE_MODE else f"{HF_USERNAME}/items_full"

SEED = 42

# Human pricing: the script writes HUMAN_IN; a person fills in prices and saves HUMAN_OUT
HUMAN_IN = Path("human_in.csv")
HUMAN_OUT = Path("human_out.csv")
HUMAN_SIZE = 100

# Neural network
HASH_FEATURES = 5_000                       # length of each bag-of-words vector
EPOCHS = 2                                  # complete passes through the training data
BATCH_SIZE = 64
LEARNING_RATE = 0.001
VAL_FRACTION = 0.01                         # share of training items held back for validation

# Frontier LLMs: (function name, LiteLLM model id, extra completion() arguments).
# The function name becomes the chart title: "gpt_5__1" -> "GPT 5.1".
LLM_MODELS = [
    ("gpt_6_luna", "openai/gpt-6-luna", {"extra_body": {"reasoning_effort": "none"}}),
    ("claude_opus_4_5", "anthropic/claude-opus-4-5", {}),
    ("gemini_3__1_flash_lite", "gemini/gemini-3.1-flash-lite", {}),
    ("gpt_5__1", "gpt-5.1", {"reasoning_effort": "high", "seed": SEED}),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items():
    """Log in to HuggingFace and load the summarised train / validation / test splits.

    Returns:
        (train, val, test) lists of Items. Validation is not used here: the neural network
        holds back its own VAL_FRACTION of the training items instead.
    """
    # If you see a "Note" about the environment variable being set, it can be ignored
    login(os.environ["HF_TOKEN"], add_to_git_credential=True)

    train, val, test = Item.from_hub(DATASET)
    print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
    return train, val, test


# ---------------------------------------------------------------------------
# 1. Human baseline
# ---------------------------------------------------------------------------

def write_human_input(test):
    """Write the first HUMAN_SIZE test summaries to HUMAN_IN for a person to price.

    Each row is `summary, 0`. The person replaces each 0 with their guess, keeping the
    row order, and saves the result as HUMAN_OUT. Overwrites HUMAN_IN on every run.

    Args:
        test: Test Items; only the first HUMAN_SIZE are written.
    """
    with HUMAN_IN.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for item in test[:HUMAN_SIZE]:
            writer.writerow([item.summary, 0])


def make_human_pricer(test):
    """Read the human's guesses from HUMAN_OUT and return them as a pricer.

    Row i of HUMAN_OUT holds the guess for test[i]; the price is in the second column.

    Args:
        test: Test Items, in the same order as written by write_human_input().

    Returns:
        The pricer function, or None if HUMAN_OUT doesn't exist yet.
    """
    if not HUMAN_OUT.exists():
        return None
    with HUMAN_OUT.open("r", encoding="utf-8") as csv_file:
        guesses = [float(row[1]) for row in csv.reader(csv_file)]

    # Look items up by identity: the evaluator passes the same objects from `test`
    guess_for = {id(item): guess for item, guess in zip(test, guesses)}

    def human_pricer(item):
        """Return the human's guess for this test item (only the first HUMAN_SIZE have one)."""
        return guess_for[id(item)]

    print(f"Human predicted {human_pricer(test[0])} for an item that actually costs {test[0].price}")
    return human_pricer


# ---------------------------------------------------------------------------
# 2. Neural network
# ---------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    """Eight fully connected layers: input -> 128 -> 64 (x6) -> 1 price.

    Each layer except the last is followed by a ReLU, which lets the network learn
    non-linear combinations of words. The output is the predicted price in dollars.
    """

    def __init__(self, input_size):
        """Build the layers for input vectors of length input_size (HASH_FEATURES)."""
        super().__init__()
        sizes = [input_size, 128] + [64] * 6 + [1]
        layers = []
        for n_in, n_out in zip(sizes, sizes[1:]):
            layers += [nn.Linear(n_in, n_out), nn.ReLU()]
        self.layers = nn.Sequential(*layers[:-1])               # no ReLU after the output

    def forward(self, x):
        """Map a batch of bag-of-words vectors (batch, input_size) to prices (batch, 1)."""
        return self.layers(x)


def make_vectorizer():
    """Bag-of-words vectorizer that hashes each word into one of HASH_FEATURES slots.

    Hashing needs no vocabulary to be learned (fit does nothing), so any text can be
    vectorised the same way. binary=True records whether a word is present (1) rather
    than how often it appears, and English stop words ("the", "and", ...) are ignored.
    """
    return HashingVectorizer(n_features=HASH_FEATURES, stop_words="english", binary=True)


def to_tensor(vectorizer, texts):
    """Vectorise texts into a dense float tensor, one row per text."""
    return torch.FloatTensor(vectorizer.transform(texts).toarray())


def train_neural_network(train):
    """Train NeuralNetwork to predict price from the bag-of-words vector of a summary.

    Uses mean squared error loss and the Adam optimiser for EPOCHS passes over the data,
    in mini-batches of BATCH_SIZE. After each epoch it prints the loss on the last
    training batch and on the held-back validation items (in squared dollars).

    Memory: the training matrix is held dense - in full mode about 16 GB
    (800,000 items x 5,000 features x 4 bytes), plus a copy for the train/validation split.

    Args:
        train: Training Items.

    Returns:
        (model, vectorizer): the trained network and the vectorizer it expects.
    """
    vectorizer = make_vectorizer()
    X = to_tensor(vectorizer, [item.summary for item in train])
    y = torch.FloatTensor([float(item.price) for item in train]).unsqueeze(1)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_FRACTION, random_state=SEED)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    model = NeuralNetwork(X.shape[1])
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params:,}")

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in tqdm(train_loader):
            # The four stages of training: forward pass, loss, backward pass, update weights
            optimizer.zero_grad()
            loss = loss_function(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_function(model(X_val), y_val)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {loss.item():.3f}, Val Loss: {val_loss.item():.3f}")

    return model, vectorizer


def make_neural_network_pricer(model, vectorizer):
    """Wrap the trained network as a pricer; negative predictions are clamped to $0."""
    model.eval()

    def neural_network(item):
        """Predict the item's price from its summary with the trained network."""
        with torch.no_grad():                # inference only: no gradients needed
            return max(0, model(to_tensor(vectorizer, [item.summary]))[0].item())

    return neural_network


# ---------------------------------------------------------------------------
# 3. Frontier LLMs
# ---------------------------------------------------------------------------

def messages_for(item):
    """Chat messages asking an LLM to price one item, with no examples (zero-shot).

    The evaluator extracts the first number from the reply, so the prompt asks for the
    price alone.
    """
    message = f"Estimate the price of this product. Respond with the price, no explanation\n\n{item.summary}"
    return [{"role": "user", "content": message}]


def make_llm_pricer(name, model, options):
    """Build a pricer that asks an LLM, through LiteLLM, for each item's price.

    Args:
        name: Function name; pricer.evaluator turns it into the chart title.
        model: LiteLLM model id, e.g. "anthropic/claude-opus-4-5".
        options: Extra completion() arguments, e.g. reasoning effort or a seed.

    Returns:
        The pricer function. It returns the raw reply text; the evaluator parses the number.
    """
    def pricer(item):
        """Ask the LLM for this item's price; returns its reply text."""
        response = completion(model=model, messages=messages_for(item), **options)
        return response.choices[0].message.content

    pricer.__name__ = name
    return pricer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Evaluate the human, the neural network and each frontier LLM in turn."""
    load_dotenv(override=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)                 # repeatable network weights and batch order

    train, _, test = load_items()

    # 1. Human - skipped until someone has filled in HUMAN_OUT
    write_human_input(test)
    human_pricer = make_human_pricer(test)
    if human_pricer:
        evaluate(human_pricer, test, size=HUMAN_SIZE)
    else:
        print(f"Skipping the human baseline: price the items in {HUMAN_IN} and save them as {HUMAN_OUT}")

    # 2. Neural network
    model, vectorizer = train_neural_network(train)
    evaluate(make_neural_network_pricer(model, vectorizer), test)

    # 3. Frontier LLMs - show one prompt and answer first, then evaluate each model
    pricers = [make_llm_pricer(name, model_id, options) for name, model_id, options in LLM_MODELS]
    print(messages_for(test[0])[0]["content"])
    print(f"{pricers[0].__name__} answered {pricers[0](test[0])!r}; actual price {test[0].price}")
    for pricer in pricers:
        evaluate(pricer, test)


if __name__ == "__main__":
    main()
