from pricer.items import Item
import json
import re

# Filters for which products become Items
MIN_CHARS = 600                             # minimum length of the cleaned text
MIN_PRICE = 0.5                             # prices round to $1 ... $999
MAX_PRICE = 999.49

# Text length caps, to keep prompts small
MAX_TEXT_EACH = 3000                        # per part: description, features
MAX_TEXT_TOTAL = 4000                       # whole cleaned text

# Product "details" entries that carry no information about price
REMOVALS = [
    "Part Number",
    "Best Sellers Rank",
    "Batteries Included?",
    "Batteries Required?",
    "Item model number",
]


def simplify(text_list) -> str:
    """
    Return a simplified string without too much whitespace and limited to MAX_TEXT characters
    """
    return (
        str(text_list)
        .replace("\n", " ")
        .replace("\r", "")
        .replace("\t", "")
        .replace("  ", " ")
        .strip()[:MAX_TEXT_EACH]
    )


def scrub(title, description, features, details) -> str:
    """
    Return a cleansed full string with product numbers and unimportant details removed
    """
    for remove in REMOVALS:
        details.pop(remove, None)
    result = title + "\n"
    if description:
        result += simplify(description) + "\n"
    if features:
        result += simplify(features) + "\n"
    if details:
        result += json.dumps(details) + "\n"
    # Remove part-number-like codes: 7+ capital letters and digits containing at least one of each
    pattern = r"\b(?=[A-Z0-9]{7,}\b)(?=.*[A-Z])(?=.*\d)[A-Z0-9]+\b"
    return re.sub(pattern, "", result).strip()[:MAX_TEXT_TOTAL]


def get_weight(details):
    """
    Return the product's "Item Weight" detail converted to pounds, or 0 if it is
    missing or in a unit not handled here
    """
    weight_str = details.get("Item Weight")
    if weight_str:
        # e.g. "2.5 Pounds", "12 ounces", "3 hundredths pounds"
        parts = weight_str.split(" ")
        amount = float(parts[0])
        unit = parts[1].lower()
        if unit == "pounds":
            return amount
        elif unit == "ounces":
            return amount / 16
        elif unit == "grams":
            return amount / 453.592
        elif unit == "milligrams":
            return amount / 453592
        elif unit == "kilograms":
            return amount / 0.453592
        elif unit == "hundredths" and parts[2].lower() == "pounds":
            return amount / 100
    return 0


def parse(datapoint, category):
    """
    Turn one raw Amazon metadata row into an Item, or return None to skip it

    Skipped when the price is missing or not a number, outside MIN_PRICE..MAX_PRICE,
    or when the cleaned text is shorter than MIN_CHARS (too little to price from)
    """
    try:
        price = float(datapoint["price"])
    except ValueError:
        return None
    if MIN_PRICE <= price <= MAX_PRICE:
        title = datapoint["title"]
        description = datapoint["description"]
        features = datapoint["features"]
        details = json.loads(datapoint["details"])
        weight = get_weight(details)
        full = scrub(title, description, features, details)
        if len(full) >= MIN_CHARS:
            return Item(
                title=title,
                category=category,
                price=price,
                full=full,
                weight=weight,
            )
