"""Curate the Amazon Reviews 2023 product metadata into a price-prediction dataset.

Pipeline:
    1. Explore one category (Appliances): most expensive item, text lengths, prices.
    2. Load eight categories into Item objects (price $0.50-$999.49 with at least 600
       characters of text - see pricer/parser.py).
    3. Shuffle and remove duplicate titles and duplicate descriptions.
    4. Draw a weighted sample that favours pricier items and fewer Automotive / Tools items,
       so the final price and category mix is better balanced.
    5. Explore the sample, split it into train / validation / test and push a full and a
       lite version to the HuggingFace Hub.

Needs HF_TOKEN (with write access) in the environment or a .env file.
"""

import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm.auto import tqdm

from pricer.items import Item
from pricer.loaders import ItemLoader
from pricer.parser import parse


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SOURCE_DATASET = "McAuley-Lab/Amazon-Reviews-2023"

CATEGORIES = [
    "Automotive",
    "Electronics",
    "Office_Products",
    "Tools_and_Home_Improvement",
    "Cell_Phones_and_Accessories",
    "Toys_and_Games",
    "Appliances",
    "Musical_Instruments",
]

SEED = 42
SAMPLE_SIZE = 820_000                       # 800k train + 10k validation + 10k test

# Sampling-weight multipliers: shrink the chance of picking items from over-represented
# categories (1.0 = unchanged, 0.05 = 1/20th of the usual chance)
CATEGORY_WEIGHTS = {
    "Tools_and_Home_Improvement": 0.5,
    "Automotive": 0.05,
}

HF_USERNAME = "ed-donner"                   # change to your own HuggingFace username
FULL_DATASET = f"{HF_USERNAME}/items_raw_full"
LITE_DATASET = f"{HF_USERNAME}/items_raw_lite"


# ---------------------------------------------------------------------------
# Plot helpers - each chart is drawn several times during curation
# ---------------------------------------------------------------------------

def plot_length_histogram(items, color, max_length, bin_width):
    """Histogram of the character length of each item's full text.

    The full text is what the model will read (title, description, features and details,
    scrubbed and capped at 4,000 characters), so this shows how much input it gets.

    Args:
        items: Items to plot; each must have `full` set.
        color: Matplotlib colour for the bars.
        max_length: Upper edge of the last bin; longer texts are not drawn.
        bin_width: Width of each bin in characters.
    """
    lengths = [len(item.full) for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(f"Text length: Avg {sum(lengths) / len(lengths):,.1f} and highest {max(lengths):,}\n")
    plt.xlabel("Length (characters)")
    plt.ylabel("Count")
    plt.hist(lengths, rwidth=0.7, color=color, bins=range(0, max_length, bin_width))
    plt.show()


def plot_price_histogram(items, color):
    """Histogram of item prices in $10 buckets from $0 to $1,000.

    Used before and after sampling to check that the cheap-item skew has been reduced.

    Args:
        items: Items to plot.
        color: Matplotlib colour for the bars.
    """
    prices = [item.price for item in items]
    plt.figure(figsize=(15, 6))
    plt.title(
        f"Prices: Avg {sum(prices) / len(prices):,.1f} "
        f"lowest {min(prices):,} and highest {max(prices):,}\n"
    )
    plt.xlabel("Price ($)")
    plt.ylabel("Count")
    plt.hist(prices, rwidth=0.7, color=color, bins=range(0, 1000, 10))
    plt.show()


def plot_category_bar(category_counts):
    """Bar chart of the number of items per category, with the count above each bar.

    Args:
        category_counts: Mapping of category name -> number of items (e.g. a Counter).
    """
    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.figure(figsize=(15, 6))
    plt.bar(categories, counts, color="goldenrod")
    plt.title("How many in each category")
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")     # long category names would overlap otherwise
    for i, count in enumerate(counts):
        plt.text(i, count, f"{count:,}", ha="center", va="bottom")
    plt.show()


def plot_category_donut(category_counts):
    """Donut chart showing each category's percentage share of the items.

    Args:
        category_counts: Mapping of category name -> number of items (e.g. a Counter).
    """
    plt.figure(figsize=(12, 10))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct="%1.0f%%", startangle=90)

    # A white circle in the middle turns the pie into a donut
    plt.gca().add_artist(plt.Circle((0, 0), 0.70, fc="white"))
    plt.title("Categories")
    plt.axis("equal")                       # keep the pie circular
    plt.show()


def plot_price_scatter(x_values, items, color, x_label, title, x_limit=None):
    """Scatter plot of price against another item attribute.

    A quick visual check of whether a single simple feature predicts price; points are
    tiny (s=0.2) because there are hundreds of thousands of them.

    Args:
        x_values: One x value per item, in the same order as `items`.
        items: Items whose prices form the y axis.
        color: Matplotlib colour for the points.
        x_label: Label for the x axis.
        title: Chart title.
        x_limit: Optional upper limit for the x axis, to cut off extreme outliers.
    """
    plt.figure(figsize=(15, 8))
    plt.scatter(x_values, [item.price for item in items], s=0.2, color=color)
    plt.xlabel(x_label)
    plt.ylabel("Price")
    if x_limit:
        plt.xlim(0, x_limit)
    plt.title(title)
    plt.show()


# ---------------------------------------------------------------------------
# Curation steps
# ---------------------------------------------------------------------------

def explore_appliances():
    """Look at one raw category before committing to the full eight-category pipeline.

    Downloads the Appliances metadata (one of the smaller categories), then:
      - prints the most expensive datapoint, including unrealistic prices that parse() drops,
      - parses it into Items and reports how many datapoints survive the filters,
      - plots text lengths and prices, and prints the longest item's text.
    Nothing is returned; this step only informs the choices made in the later steps.
    """
    dataset = load_dataset(SOURCE_DATASET, "raw_meta_Appliances", split="full", trust_remote_code=True)
    print(f"Number of Appliances: {len(dataset):,}")

    # Most expensive item - many datapoints have an empty price, so skip those
    max_price, max_item = 0, None
    for datapoint in tqdm(dataset):
        try:
            price = float(datapoint["price"])
        except ValueError:
            continue
        if price > max_price:
            max_price, max_item = price, datapoint
    print(f"The most expensive item is {max_item['title']} and it costs {max_price:,.2f}")

    # parse() returns None for items outside the price range or with too little text
    items = [item for item in (parse(datapoint, "Appliances") for datapoint in tqdm(dataset)) if item]
    print(f"There are {len(items):,} items from {len(dataset):,} datapoints")

    plot_length_histogram(items, "lightblue", max_length=6000, bin_width=100)
    print(max(items, key=lambda item: len(item.full)).full)      # the longest item's text
    plot_price_histogram(items, "orange")


def load_all_categories():
    """Download and parse every category in CATEGORIES into one list of Items.

    ItemLoader parses each category in parallel with a process pool (all CPU cores but
    one), so this is the slowest step: expect several minutes per large category and a
    busy machine. Downloads are cached by HuggingFace, so re-runs are faster.

    Returns:
        All parsed Items, grouped by category in CATEGORIES order.
    """
    items = []
    for category in CATEGORIES:
        items.extend(ItemLoader(category).load())
    print(f"A grand total of {len(items):,} items")
    return items


def unique_by(items, key):
    """Keep the first item for each distinct key value, preserving order.

    `seen.add()` returns None, so `x in seen or seen.add(x)` is falsy only the first time
    a key appears - and records it in the same expression.

    Args:
        items: Items to filter.
        key: Function returning the value that must be unique (e.g. the title).

    Returns:
        A new list without the later duplicates.
    """
    seen = set()
    return [item for item in tqdm(items) if not (key(item) in seen or seen.add(key(item)))]


def deduplicate(items):
    """Shuffle, then drop items whose title or full text was already seen.

    Amazon lists the same product many times (colour / size variants, resellers). Leaving
    those in would let near-identical items land in both train and test, inflating the
    test score. Shuffling first means the copy that survives is random rather than always
    the one from the first category loaded.

    Args:
        items: All parsed Items; shuffled in place.

    Returns:
        The de-duplicated Items, in shuffled order.
    """
    random.seed(SEED)
    random.shuffle(items)
    items = unique_by(items, lambda item: item.title)
    items = unique_by(items, lambda item: item.full)
    print(f"After deduplication, we have {len(items):,} items")
    return items


def weighted_sample(items):
    """Sample SAMPLE_SIZE items without replacement, favouring higher prices.

    The raw data is dominated by cheap items and by Automotive / Tools. A model trained on
    that would learn to guess low prices for everything.

    "Weights" here are sampling weights in the statistical sense: each item's relative chance
    of being picked (they become the probabilities `p` passed to np.random.choice). They are
    not neural-network weights and not the product's physical weight. Each item's weight is:

        weight = ((price - min) / (max - min)) ** 2  *  CATEGORY_WEIGHTS.get(category, 1)

    Squaring the normalised price pushes the sample towards expensive items; the category
    factor thins out the over-represented categories. Weights are normalised to sum to 1.
    The cheapest item gets a weight of zero and can never be picked.

    Args:
        items: De-duplicated Items; must contain more than SAMPLE_SIZE items with a
            non-zero weight, or np.random.choice raises ValueError.

    Returns:
        SAMPLE_SIZE Items in random order. Seeded, so re-runs give the same sample.
    """
    np.random.seed(SEED)
    prices = np.array([item.price for item in items], dtype=float)
    categories = np.array([item.category for item in items])

    # Normalise prices to 0..1; the small epsilon avoids dividing by zero if all are equal
    normalised = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
    weights = normalised**2
    for category, factor in CATEGORY_WEIGHTS.items():
        weights[categories == category] *= factor
    weights /= weights.sum()

    indexes = np.random.choice(len(items), size=SAMPLE_SIZE, replace=False, p=weights)
    sample = [items[i] for i in indexes]

    # Shuffle again so the order carries no trace of the sampling
    random.seed(SEED)
    random.shuffle(sample)
    return sample


def explore_sample(sample):
    """Check the final sample's price and category mix and look for simple price signals.

    Plots:
      - the price histogram (should be flatter than before sampling),
      - category counts as a bar chart and as a donut,
      - price against text length and against weight - if either showed a strong pattern,
        a simple baseline model could exploit it.

    Args:
        sample: The weighted sample from weighted_sample().
    """
    plot_price_histogram(sample, "blueviolet")

    # Automotive is still the largest category, but much less dominant
    category_counts = Counter(item.category for item in sample)
    plot_category_bar(category_counts)
    plot_category_donut(category_counts)

    plot_price_scatter(
        [len(item.full) for item in sample], sample, "red",
        x_label="Size", title="Is there a simple correlation with text length?",
    )
    # item.weight is the product's physical shipping weight ("Item Weight" in the Amazon
    # details), not a sampling or model weight. parser.get_weight() converts every unit
    # to pounds; items without a weight get 0

    plot_price_scatter(
        [item.weight for item in sample], sample, "darkorange",
        x_label="Weight (pounds)", title="Is there a simple correlation with weight?", x_limit=400,
    )


def push_datasets(sample):
    """Split the sample and push it to the HuggingFace Hub as two datasets.

    Splits (the sample is already shuffled, so slicing gives random splits):
      - full: 800,000 train / 10,000 validation / 10,000 test -> FULL_DATASET
      - lite:  20,000 train /  1,000 validation /  1,000 test -> LITE_DATASET
    The lite splits are the first rows of the full splits, for quick experiments on modest
    hardware. Pushing overwrites an existing dataset of the same name and needs an HF_TOKEN
    with write access to the HF_USERNAME account.

    Args:
        sample: The SAMPLE_SIZE shuffled Items from weighted_sample().
    """
    train = sample[:800_000]
    val = sample[800_000:810_000]
    test = sample[810_000:]
    Item.push_to_hub(FULL_DATASET, train, val, test)

    Item.push_to_hub(LITE_DATASET, train[:20_000], val[:1_000], test[:1_000])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run the whole curation pipeline, from login to pushing the datasets."""
    load_dotenv(override=True)
    # If you see a "Note" about the environment variable being set, it can be ignored
    login(os.environ["HF_TOKEN"], add_to_git_credential=True)

    # Step 1: explore a single category
    explore_appliances()

    # Steps 2-3: load everything and remove duplicates, then look at the result
    items = load_all_categories()
    items = deduplicate(items)
    plot_length_histogram(items, "skyblue", max_length=4050, bin_width=50)
    plot_price_histogram(items, "blueviolet")
    plot_category_bar(Counter(item.category for item in items))

    # Steps 4-5: rebalance, check the sample and publish it
    sample = weighted_sample(items)
    explore_sample(sample)
    push_datasets(sample)


# The guard is required: ItemLoader starts worker processes, which re-import this module on Windows
if __name__ == "__main__":
    main()
