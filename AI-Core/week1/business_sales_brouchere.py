"""
Build a company brochure from its website.

The pipeline has three stages:

    1. Discover  - crawl the site's product catalogue for products and URLs
    2. Gather    - fetch the landing page plus the most relevant linked pages
    3. Write     - hand it all to the model and stream back a markdown brochure

Run it as a script to save markdown into ./output, or import the functions
into a notebook, where they render inline instead:

    python business_sales_brouchere.py                     # brochure
    python business_sales_brouchere.py --products          # catalogue only
    python business_sales_brouchere.py --company "Acme" --url https://acme.com
"""

import argparse
import json
import os
import re
import sys
from functools import lru_cache
from urllib.parse import urlparse

from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
from openai import OpenAI

from scrapers import fetch_website_contents, fetch_website_links_with_text

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_COMPANY = "Hexagon"
DEFAULT_URL = "https://hexagon.com"

CHEAP_MODEL = "gpt-5-nano"      # link triage: high volume, low judgement
WRITING_MODEL = "gpt-4.1-mini"  # naming products and writing prose

# How much source material to put in front of the model. These budgets are
# applied at fetch time, so we only pay for pages we actually use.
LANDING_PAGE_CHARS = 3_000       # characters kept from the landing page
MAX_RELEVANT_PAGES = 12          # linked pages fetched for the brochure
CHARS_PER_PAGE = 1_500           # characters kept from each linked page
MAX_PRODUCTS_IN_BROCHURE = 40    # products listed in the brochure prompt

# The product crawl. Depth 0 is the catalogue hub, depth 1 its categories,
# depth 2 their sub-categories - which is where the deepest products sit.
# The page budget bounds the whole crawl; raise it if a run reports pages
# left queued. A full hexagon.com crawl is ~170 pages and takes a few minutes.
CRAWL_PAGE_BUDGET = 200
CRAWL_MAX_DEPTH = 2

load_dotenv(override=True)
openai = OpenAI()

if not (os.getenv("OPENAI_API_KEY") or "").startswith("sk-proj-"):
    print("OPENAI_API_KEY looks wrong or missing - see the troubleshooting notebook")

# --------------------------------------------------------------------------
# Output: render inline in a notebook, write a markdown file from a script
# --------------------------------------------------------------------------

def in_notebook() -> bool:
    """True when running under IPython/Jupyter, where rich display works."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if not in_notebook():
    # Streaming to a plain terminal or a pipe: keep em dashes and the like.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

try:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
except NameError:                       # notebooks have no __file__
    OUTPUT_DIR = os.path.join(os.getcwd(), "output")


def slugify(text: str) -> str:
    """'Hexagon AB' -> 'hexagon-ab', safe to use as a filename."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "company"


def deliver(markdown: str, company: str, kind: str,
            save: bool | None = None, already_shown: bool = False) -> str | None:
    """
    Present `markdown` the way that suits where we are running.

    A notebook renders it inline; a script has already printed it, so it is
    written to output/<company>-<kind>.md instead. Pass save=True/False to
    override, or already_shown=True if the caller streamed it live.
    """
    if save is None:
        save = not in_notebook()

    if in_notebook() and not already_shown:
        display(Markdown(markdown))

    if not save:
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"{slugify(company)}-{kind}.md")
    with open(path, "w", encoding="utf-8", newline="\n") as markdown_file:
        markdown_file.write(markdown.rstrip() + "\n")
    print(f"\nSaved {kind} to {path}")
    return path


def ask_for_json(model: str, system_prompt: str, user_prompt: str) -> dict:
    """One chat completion constrained to a JSON object, parsed for you."""
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


@lru_cache(maxsize=None)
def page_links(url: str, same_domain_only: bool = True) -> tuple:
    """
    Links on `url` as a cached tuple of (anchor text, absolute URL) pairs.

    Several stages start from the same landing page, and the crawl reaches the
    same category pages by different routes; caching keeps each page to one
    HTTP request. Tuples because lru_cache needs a hashable return value.

    The product crawl passes same_domain_only=False: a group's catalogue page
    is often just a shop window linking out to a brand site, so restricting to
    one domain throws the actual products away.
    """
    links = fetch_website_links_with_text(url, same_domain_only=same_domain_only)
    return tuple((link["text"], link["url"]) for link in links)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

RELEVANT_LINKS_PROMPT = """
You are provided with a list of links found on a webpage.
Decide which are most relevant to a brochure about the company - About,
Company, Products, Careers/Jobs pages and the like. Ignore Terms of Service,
Privacy and email links. Respond in JSON as in this example:

{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""

PRODUCT_HUB_PROMPT = """
You are given a list of links found on a company's landing page.
Pick the links leading to the company's PRODUCTS or SOLUTIONS catalogue - the
hub pages listing what the company sells. Prefer an "all products" index over
a single product. Pick at most 3. Respond in JSON as in this example:

{
    "hubs": [
        {"type": "all products", "url": "https://full.url/goes/here/products"}
    ]
}
"""

PRODUCT_FILTER_PROMPT = """
You are given a list of links (anchor text plus URL) harvested from a
company's product pages. Identify which are actual PRODUCTS the company
sells: named commercial offerings such as instruments, hardware, software,
platforms or services.

Exclude everything that is not a product: navigation and footer links, broad
category or "product group" index pages, blog posts, news, events, press
releases, careers, investor, contact and support pages.

Use the anchor text as the name when it reads like a product name, otherwise
derive a readable name from the URL slug. Copy each URL EXACTLY as given -
never invent or edit one. Respond in JSON as in this example:

{
    "products": [
        {"name": "Leica Absolute Tracker ATS800",
         "url": "https://full.url/goes/here/products/leica-absolute-tracker-ats800"}
    ]
}
"""

BROCHURE_PROMPT = """
You are an assistant that analyses the contents of several pages from a
company website and creates a short brochure about the company for
prospective customers, investors and recruits.

Respond in markdown without code blocks. Include company culture, products,
customers and careers/jobs where the information supports it.

When a "Products" list is supplied, include a Products section naming the
notable ones as markdown links, using the exact URLs given. Group them
sensibly by category rather than listing every one, and never invent a
product or a URL.
"""

# --------------------------------------------------------------------------
# Stage 1: discover the product catalogue
# --------------------------------------------------------------------------

# Locale prefixes come in both shapes: hexagon.com/de/... and
# leica-geosystems.com/es-es/... - matching only the short one lets the long
# one through as a duplicate of a product already listed.
LOCALE_SEGMENT = re.compile(r"^[a-z]{2}([-_][a-z]{2})?$")


def is_locale_variant(url: str) -> bool:
    """True for translated duplicates such as /de/products/ or /es-es/products/."""
    segments = [segment for segment in urlparse(url).path.split("/") if segment]
    return bool(segments) and bool(LOCALE_SEGMENT.match(segments[0].lower()))


def is_category_page(text: str, url: str) -> bool:
    """
    True for index pages listing further products, which the crawl expands.

    Three shapes, all present on hexagon.com:
      - .../products/product-groups/<category>   the category pages
      - .../products/all-products                the catalogue root
      - any link reading "Explore all ... products" - the per-vertical hubs.
        Some of those sit under /industries/ rather than /products/, so
        matching on the URL alone silently drops a whole vertical.
    """
    path = urlparse(url).path.lower()
    return ("product-groups/" in path
            or path.rstrip("/").endswith("/all-products")
            or text.lower().startswith("explore all"))


def is_product_candidate(url: str) -> bool:
    """
    True for links worth offering the model as possible products.

    Matched on the path alone, deliberately, so that products on a company's
    other brand domains still count. Hexagon is the case in point: its
    catalogue pages list Leica Geosystems, NovAtel, VeriPos and GeoMax
    hardware, but each "Learn more" points at leica-geosystems.com or
    novatel.com. Filtering to one domain loses the whole geospatial and
    positioning range.
    """
    path = urlparse(url).path.lower()
    return "/products/" in path or "/product-groups/" in path


def same_site(url: str, other: str) -> bool:
    """True when both URLs are on the same domain, ignoring any www prefix."""
    def host(value: str) -> str:
        return urlparse(value).netloc.lower().removeprefix("www.")
    return host(url) == host(other)


def category_name(url: str) -> str:
    """
    Readable category from a category page's URL, e.g. 'Total stations'.

    Taking the category from the crawl's structure, rather than asking the
    model to invent one per product, keeps the headings consistent. Left to
    the model, one run returns "CAD/CAM software", "CAM software" and
    "computer-aided manufacturing CAD/CAM software" as three separate groups.
    """
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    return slug.replace("-", " ").capitalize()


def find_product_hubs(url: str) -> list:
    """Ask the model which landing-page links lead to the product catalogue."""
    listing = "\n".join(link_url for _text, link_url in page_links(url))
    hubs = ask_for_json(CHEAP_MODEL, PRODUCT_HUB_PROMPT,
                        f"Links on {url}:\n\n{listing}")["hubs"]
    hub_urls = [hub["url"] for hub in hubs]
    print(f"Product hub pages: {hub_urls}")
    return hub_urls


def crawl_product_links(url: str, page_budget: int = CRAWL_PAGE_BUDGET,
                        max_depth: int = CRAWL_MAX_DEPTH,
                        follow_brands: bool = False) -> dict:
    """
    Breadth-first crawl of the catalogue, returning {url: {text, category}}.

    Breadth-first matters. The catalogue spreads ~50 category pages across six
    verticals (agriculture, geospatial, manufacturing, mining, positioning,
    surveying). Walking them depth-first and stopping at a budget would cover
    the alphabet from "aec" to "cad-cam" and silently miss the rest, so the
    queue drains level by level and the budget bounds the crawl as a whole.

    Products are harvested from every domain, because catalogue pages link out
    to brand sites. Those brand pages are only *crawled* when follow_brands is
    set, which finds far more products at the cost of a much longer run.
    """
    candidates = {}     # url -> {"text": ..., "category": ...}
    visited = set()
    queued = set()
    frontier = []       # (url, depth, category inherited from the parent page)

    for hub in find_product_hubs(url):
        if hub not in queued:
            queued.add(hub)
            frontier.append((hub, 0, ""))

    while frontier and len(visited) < page_budget:
        page_url, depth, category = frontier.pop(0)
        if page_url in visited:
            continue
        visited.add(page_url)
        print(f"  [{len(visited)}/{page_budget}] depth {depth}: {page_url}")

        try:
            links = page_links(page_url, same_domain_only=False)
        except Exception as error:   # one dead category page must not stop the crawl
            print(f"      skipped: {error}")
            continue

        for text, link_url in links:
            if is_locale_variant(link_url):
                continue

            # Products inherit the category of the page they were found on.
            if is_product_candidate(link_url):
                candidates.setdefault(link_url, {"text": text, "category": category})

            # Brand sites use their own layouts, so any product page on one
            # is worth expanding; on the main site, only real category pages are.
            offsite = not same_site(link_url, url)
            worth_expanding = (is_product_candidate(link_url) if offsite
                               else is_category_page(text, link_url))
            if offsite and not follow_brands:
                worth_expanding = False

            if depth < max_depth and link_url not in queued and worth_expanding:
                queued.add(link_url)
                frontier.append((link_url, depth + 1,
                                 category if offsite else category_name(link_url)))

    if frontier:
        print(f"Stopped at the {page_budget}-page budget with {len(frontier)} pages "
              f"still queued - raise page_budget for fuller coverage")
    offsite_count = sum(1 for link_url in candidates if not same_site(link_url, url))
    print(f"Crawled {len(visited)} pages, found {len(candidates)} candidate links "
          f"({offsite_count} on brand domains)")
    return candidates


def find_products(company: str, url: str, page_budget: int = CRAWL_PAGE_BUDGET,
                  follow_brands: bool = False, batch_size: int = 120) -> list:
    """Return [{"name", "category", "url"}] for the products found on the site."""
    candidates = crawl_product_links(url, page_budget, follow_brands=follow_brands)
    listing = [f"{data['text'] or '(no text)'} -> {link_url}"
               for link_url, data in candidates.items()]

    # Classified in batches: a single call covering hundreds of links quietly
    # drops entries once its response grows long, which looks like a crawl gap.
    products = {}
    for start in range(0, len(listing), batch_size):
        batch = listing[start:start + batch_size]
        print(f"  classifying links {start + 1}-{start + len(batch)} of {len(listing)}")
        answer = ask_for_json(WRITING_MODEL, PRODUCT_FILTER_PROMPT,
                              f"Company: {company}\n\nLinks:\n\n" + "\n".join(batch))

        for product in answer.get("products", []):
            # Only URLs we actually crawled, so a hallucinated link cannot slip through.
            product_url = product.get("url")
            if product_url in candidates and product_url not in products:
                products[product_url] = {
                    "name": product["name"],
                    "url": product_url,
                    "category": candidates[product_url]["category"] or "Other",
                }

    print(f"Identified {len(products)} products\n")
    return sorted(products.values(), key=lambda item: (item["category"], item["name"]))


def products_as_markdown(company: str, products: list) -> str:
    """Render products as a markdown document grouped by category."""
    lines = [f"# {company} products ({len(products)})"]
    current_category = None
    for product in products:
        if product["category"] != current_category:
            current_category = product["category"]
            lines += ["", f"## {current_category}", ""]
        lines.append(f"- [{product['name']}]({product['url']})")
    return "\n".join(lines)


def show_products(company: str = DEFAULT_COMPANY, url: str = DEFAULT_URL,
                  page_budget: int = CRAWL_PAGE_BUDGET,
                  follow_brands: bool = False, save: bool | None = None) -> list:
    """Find the company's products and show them as a grouped markdown list."""
    products = find_products(company, url, page_budget, follow_brands)
    markdown = products_as_markdown(company, products)
    if not in_notebook():
        print(markdown)
    deliver(markdown, company, "products", save)
    return products


# --------------------------------------------------------------------------
# Stage 2: gather the pages the brochure is written from
# --------------------------------------------------------------------------

def select_relevant_links(url: str) -> list:
    """Ask the model which of the page's links belong in a brochure."""
    listing = "\n".join(link_url for _text, link_url in page_links(url))
    user_prompt = (
        f"Here is the list of links on the website {url}.\n"
        "Decide which are relevant to a brochure about the company and respond "
        "with the full https URL in JSON format.\n\nLinks:\n\n" + listing
    )
    links = ask_for_json(CHEAP_MODEL, RELEVANT_LINKS_PROMPT, user_prompt)["links"]
    print(f"Found {len(links)} relevant links")
    return links


def gather_page_contents(url: str, max_pages: int = MAX_RELEVANT_PAGES,
                         chars_per_page: int = CHARS_PER_PAGE) -> str:
    """
    The landing page plus the most relevant linked pages, as one string.

    The budgets are applied when fetching, so every page we pay an HTTP
    request for actually reaches the model.
    """
    landing = fetch_website_contents(url, max_chars=LANDING_PAGE_CHARS)
    links = select_relevant_links(url)

    if len(links) > max_pages:
        print(f"Using the {max_pages} most relevant of {len(links)} links")
        links = links[:max_pages]

    sections = [f"## Landing page:\n\n{landing}\n\n## Relevant links:"]
    for link in links:
        try:
            contents = fetch_website_contents(link["url"], max_chars=chars_per_page)
        except Exception as error:   # one dead link must not sink the brochure
            print(f"  skipped {link['url']}: {error}")
            continue
        sections.append(f"### {link['type']} ({link['url']})\n{contents}")
    return "\n\n".join(sections)


def products_section(company: str, url: str, page_budget: int = CRAWL_PAGE_BUDGET,
                     follow_brands: bool = False,
                     max_products: int = MAX_PRODUCTS_IN_BROCHURE) -> str:
    """The discovered products as a name/category/URL block for the prompt."""
    products = find_products(company, url, page_budget, follow_brands)
    if not products:
        return ""

    if len(products) > max_products:
        print(f"Listing {max_products} of {len(products)} products in the brochure")
        products = products[:max_products]

    lines = ["## Products (these URLs are verified - link to them exactly as written):"]
    lines += [f"- {product['name']} [{product['category']}]: {product['url']}"
              for product in products]
    return "\n".join(lines)


def build_brochure_prompt(company: str, url: str, include_products: bool = True,
                          page_budget: int = CRAWL_PAGE_BUDGET,
                          follow_brands: bool = False) -> str:
    """Assemble everything the model needs to write the brochure."""
    parts = [
        f"You are looking at a company called: {company}\n"
        "Here are the contents of its landing page, its product catalogue and "
        "other relevant pages. Use them to build a short brochure of the "
        "company in markdown without code blocks."
    ]
    if include_products:
        parts.append(products_section(company, url, page_budget, follow_brands))
    parts.append(gather_page_contents(url))

    prompt = "\n\n".join(part for part in parts if part)
    print(f"Brochure prompt: {len(prompt):,} characters")
    return prompt


# --------------------------------------------------------------------------
# Stage 3: write the brochure
# --------------------------------------------------------------------------

def create_brochure(company: str = DEFAULT_COMPANY, url: str = DEFAULT_URL,
                    include_products: bool = True,
                    page_budget: int = CRAWL_PAGE_BUDGET,
                    follow_brands: bool = False,
                    save: bool | None = None) -> str:
    """Write the brochure in one shot and return the markdown."""
    response = openai.chat.completions.create(
        model=WRITING_MODEL,
        messages=[
            {"role": "system", "content": BROCHURE_PROMPT},
            {"role": "user",
             "content": build_brochure_prompt(company, url, include_products,
                                             page_budget, follow_brands)},
        ],
    )
    brochure = response.choices[0].message.content
    if not in_notebook():
        print(brochure)
    deliver(brochure, company, "brochure", save)
    return brochure


def stream_brochure(company: str = DEFAULT_COMPANY, url: str = DEFAULT_URL,
                    include_products: bool = True,
                    page_budget: int = CRAWL_PAGE_BUDGET,
                    follow_brands: bool = False,
                    save: bool | None = None) -> str:
    """Write the brochure, showing it as it arrives, and return the markdown."""
    stream = openai.chat.completions.create(
        model=WRITING_MODEL,
        messages=[
            {"role": "system", "content": BROCHURE_PROMPT},
            {"role": "user",
             "content": build_brochure_prompt(company, url, include_products,
                                             page_budget, follow_brands)},
        ],
        stream=True,
    )

    # display(display_id=True) returns a handle under IPython and None in a
    # script, so a script prints each delta as it arrives instead.
    handle = display(Markdown(""), display_id=True) if in_notebook() else None

    brochure = ""
    for chunk in stream:
        if not chunk.choices:            # usage-only chunks carry no choices
            continue
        delta = chunk.choices[0].delta.content or ""
        brochure += delta
        if handle is not None:
            update_display(Markdown(brochure), display_id=handle.display_id)
        else:
            print(delta, end="", flush=True)

    if handle is None:
        print()
    deliver(brochure, company, "brochure", save, already_shown=True)
    return brochure


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def main(argv: list | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a company brochure from its website.")
    parser.add_argument("--company", default=DEFAULT_COMPANY)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--products", action="store_true",
                        help="list the product catalogue instead of a brochure")
    parser.add_argument("--no-products", action="store_true",
                        help="skip the product crawl when writing the brochure")
    parser.add_argument("--page-budget", type=int, default=CRAWL_PAGE_BUDGET,
                        help=f"max pages to crawl (default {CRAWL_PAGE_BUDGET})")
    parser.add_argument("--follow-brands", action="store_true",
                        help="also crawl the company's other brand domains "
                             "(e.g. leica-geosystems.com), for much deeper "
                             "coverage and a much longer run")
    args = parser.parse_args(argv)

    if args.products:
        show_products(args.company, args.url, args.page_budget, args.follow_brands)
    else:
        stream_brochure(args.company, args.url,
                        include_products=not args.no_products,
                        page_budget=args.page_budget,
                        follow_brands=args.follow_brands)


if __name__ == "__main__":
    main()
