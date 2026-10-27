"""Baseline price predictors, from random guessing to classic machine learning.

These set the bar that the LLM-based and fine-tuned models later have to beat. Each
model becomes a "pricer" (a function Item -> price) and is scored by pricer.evaluator on
the first 200 test items, which prints the errors and shows two charts per model.

Models, roughly from worst to best:
    1. Random guess between $1 and $999.
    2. Constant: always the average training price.
    3. Linear regression on three hand-made features (weight, weight unknown, text length).
    4. Bag of words + linear regression: the 2,000 most common words in the summaries.
    5. Bag of words + random forest (trained on a 15,000-item subset to keep it fast).
    6. Bag of words + XGBoost gradient-boosted trees.

Input is the summarised dataset pushed by data_preprocessing.py.
"""

import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from pricer.evaluator import evaluate
from pricer.items import Item


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

LITE_MODE = False                           # True: 20k training items; False: 800k

HF_USERNAME = "ed-donner"                   # change to your own HuggingFace username
DATASET = f"{HF_USERNAME}/items_lite" if LITE_MODE else f"{HF_USERNAME}/items_full"

SEED = 42
FEATURE_COLUMNS = ["weight", "weight_unknown", "text_length"]
VOCABULARY_SIZE = 2_000                     # words kept by the bag-of-words vectorizer
RANDOM_FOREST_SUBSET = 15_000               # training items for the random forest
WORKERS = 4                                 # CPU threads for the random forest and XGBoost


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_items():
    """Load the summarised train / validation / test splits from the Hub.

    Returns:
        (train, val, test) lists of Items. Only train and test are used here; validation
        is kept for the later models.
    """
    train, val, test = Item.from_hub(DATASET)
    print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
    return train, val, test


# ---------------------------------------------------------------------------
# 1-2. Trivial baselines
# ---------------------------------------------------------------------------

def random_pricer(item):
    """Guess a random whole-dollar price from $1 to $999, ignoring the item entirely.

    The floor every real model must clear: its error is what "no information" looks like.
    """
    return random.randrange(1, 1000)


def make_constant_pricer(train):
    """Build a pricer that always predicts the average training price.

    This is the best a model can do without looking at the item, if judged by squared
    error - so a model that can't beat it has learned nothing useful.

    Args:
        train: Training Items.

    Returns:
        The pricer function.
    """
    training_average = sum(item.price for item in train) / len(train)
    print(f"Average training price: ${training_average:,.2f}")

    def constant_pricer(item):
        """Return the average training price, whatever the item."""
        return training_average

    return constant_pricer


# ---------------------------------------------------------------------------
# 3. Linear regression on hand-made features
# ---------------------------------------------------------------------------

def get_features(item):
    """Turn one item into the numeric features used by the feature-based regression.

    - weight: physical shipping weight in pounds (0 when Amazon didn't list one),
    - weight_unknown: 1 if the weight is missing, so the model can treat 0 as "unknown"
      rather than "weightless",
    - text_length: characters in the LLM summary, a rough proxy for product complexity.

    Returns:
        A dict with one entry per name in FEATURE_COLUMNS.
    """
    return {
        "weight": item.weight,
        "weight_unknown": 1 if item.weight == 0 else 0,
        "text_length": len(item.summary),
    }


def features_dataframe(items):
    """Build a DataFrame with one row of FEATURE_COLUMNS per item, in item order."""
    return pd.DataFrame([get_features(item) for item in items], columns=FEATURE_COLUMNS)


def make_linear_regression_pricer(train, test):
    """Fit a linear regression on the hand-made features and return it as a pricer.

    Prints each feature's coefficient (dollars of price per unit of that feature), plus
    the mean squared error and R² on the whole test set. R² near 0 means the features
    explain almost none of the price variation.

    Args:
        train: Training Items to fit on.
        test: Test Items for the printed metrics.

    Returns:
        The pricer function.
    """
    model = LinearRegression()
    model.fit(features_dataframe(train), [item.price for item in train])

    for feature, coefficient in zip(FEATURE_COLUMNS, model.coef_):
        print(f"{feature}: {coefficient}")
    print(f"Intercept: {model.intercept_}")

    predictions = model.predict(features_dataframe(test))
    test_prices = [item.price for item in test]
    print(f"Mean Squared Error: {mean_squared_error(test_prices, predictions)}")
    print(f"R-squared Score: {r2_score(test_prices, predictions)}")

    def linear_regression_pricer(item):
        """Predict the price from the item's weight, weight-unknown flag and text length."""
        return model.predict(features_dataframe([item]))[0]

    return linear_regression_pricer


# ---------------------------------------------------------------------------
# 4-6. Bag-of-words models
# ---------------------------------------------------------------------------

def fit_vectorizer(train):
    """Learn a bag-of-words vocabulary from the training summaries.

    Keeps the VOCABULARY_SIZE most frequent words, after removing English stop words
    ("the", "and", ...). Each summary becomes a sparse vector of word counts, which is
    what the three text models below are trained on.

    Args:
        train: Training Items.

    Returns:
        (vectorizer, X): the fitted CountVectorizer and the sparse count matrix of the
        training summaries, one row per item.
    """
    vectorizer = CountVectorizer(max_features=VOCABULARY_SIZE, stop_words="english")
    X = vectorizer.fit_transform(item.summary for item in train)

    selected_words = vectorizer.get_feature_names_out()
    print(f"Number of selected words: {len(selected_words)}")
    print("Selected words:", selected_words[1000:1020])        # a sample from the middle
    return vectorizer, X


def make_text_pricer(model, vectorizer, name):
    """Wrap a model trained on word counts as a pricer.

    The item's summary is vectorised with the same vocabulary as the training data, and
    negative predictions (possible for linear regression) are clamped to $0.

    Args:
        model: Fitted regressor with a scikit-learn style predict().
        vectorizer: The CountVectorizer the model was trained with.
        name: Function name; pricer.evaluator turns it into the chart title.

    Returns:
        The pricer function.
    """
    def pricer(item):
        """Predict the price from the word counts of the item's summary, never below $0."""
        return max(0, model.predict(vectorizer.transform([item.summary]))[0])

    pricer.__name__ = name
    return pricer


def text_models():
    """The three untrained word-count models, in the order they are trained and evaluated.

    - Linear regression: one dollar weight per word; fast, but can't combine words.
    - Random forest: 100 trees, trained on only the first RANDOM_FOREST_SUBSET items
      because it scales poorly to hundreds of thousands of rows.
    - XGBoost: 1,000 boosted trees on all items; the slowest and usually the strongest
      (expect a long run in full mode).

    Returns:
        A list of (function name, model, number of training rows or None for all).
    """
    return [
        ("natural_language_linear_regression_pricer", LinearRegression(), None),
        ("random_forest", RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=WORKERS),
         RANDOM_FOREST_SUBSET),
        ("xg_boost", xgb.XGBRegressor(n_estimators=1000, random_state=SEED, n_jobs=WORKERS, learning_rate=0.1),
         None),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Train and evaluate every baseline in turn, from simplest to most complex."""
    random.seed(SEED)                       # makes the random pricer repeatable
    np.random.seed(SEED)

    train, _, test = load_items()

    # 1-2. Trivial baselines
    evaluate(random_pricer, test)
    evaluate(make_constant_pricer(train), test)

    # 3. Hand-made features
    evaluate(make_linear_regression_pricer(train, test), test)

    # 4-6. Bag of words: train each model, then evaluate it before starting the next
    vectorizer, X = fit_vectorizer(train)
    prices = np.array([item.price for item in train], dtype=float)
    for name, model, rows in text_models():
        model.fit(X[:rows], prices[:rows])
        evaluate(make_text_pricer(model, vectorizer, name), test)


if __name__ == "__main__":
    main()
