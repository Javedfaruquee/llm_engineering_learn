"""
scraper.py — fetch page contents and links, WAF-resistant version.

Works against sites behind Akamai/Cloudflare (e.g. hexagon.com) that 403
plain `requests`, by impersonating a real Chrome TLS fingerprint via
curl_cffi. If curl_cffi is not installed, it falls back to plain requests
(fine for friendly sites). Optional Playwright fallback for JS-heavy sites.

Install:
    pip install curl_cffi beautifulsoup4
    # optional, only if you need the browser fallback:
    # pip install playwright && playwright install chromium

Usage:
    from scraper import fetch_website_contents, fetch_website_links

    text  = fetch_website_contents("https://hexagon.com")
    links = fetch_website_links("https://hexagon.com")                        # all links
    links = fetch_website_links("https://hexagon.com", same_domain_only=True) # internal only
"""

from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# HTTP layer: curl_cffi (Chrome TLS fingerprint) if available, else requests
# --------------------------------------------------------------------------
try:
    from curl_cffi import requests as _http

    _session = _http.Session(impersonate="chrome")
    _IMPERSONATING = True
except ImportError:  # curl_cffi not installed — plain requests fallback
    import requests as _http

    _session = _http.Session()
    _IMPERSONATING = False

# Browser-like headers. With curl_cffi these top up the impersonated
# profile; with plain requests they're the best we can do.
HEADERS = {
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
if not _IMPERSONATING:
    HEADERS["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    )
_session.headers.update(HEADERS)


def _get_html(url: str) -> tuple[str, bytes]:
    """
    Return (final_url, html_bytes) for the page at `url`.

    Order of attempts:
      1. curl_cffi / requests session
      2. Playwright real-browser fallback (only if installed) when we get
         a WAF-style block (403/503) or any other fetch failure
    """
    try:
        response = _session.get(url, timeout=30, allow_redirects=True)
        if response.status_code in (403, 503):
            raise PermissionError(f"HTTP {response.status_code} (WAF block?)")
        response.raise_for_status()
        return str(response.url), response.content
    except Exception as first_error:
        html = _get_html_playwright(url)
        if html is not None:
            return html
        # No Playwright available — surface the original error with advice.
        hint = (
            ""
            if _IMPERSONATING
            else " | TIP: `pip install curl_cffi` usually fixes 403s "
                 "from WAF-protected sites"
        )
        raise RuntimeError(f"Failed to fetch {url}: {first_error}{hint}") from first_error


def _get_html_playwright(url: str):
    """Render the page in headless Chromium. Returns (final_url, html_bytes)
    or None if Playwright is not installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            page.wait_for_timeout(2_000)  # let late JS (nav menus) settle
            return page.url, page.content().encode("utf-8")
        finally:
            browser.close()


# --------------------------------------------------------------------------
# Public API — same signatures as before
# --------------------------------------------------------------------------

def fetch_website_contents(url: str, max_chars: int = 2_000) -> str:
    """
    Return the title and visible text of the page at `url`,
    truncated to `max_chars` characters as a sensible limit.
    """
    _final_url, html = _get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.string.strip()
        if soup.title and soup.title.string
        else "No title found"
    )

    if soup.body:
        for irrelevant in soup.body(
            ["script", "style", "img", "input", "noscript", "svg"]
        ):
            irrelevant.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""

    return (title + "\n\n" + text)[:max_chars]


def fetch_website_links(url: str, same_domain_only: bool = False) -> list:
    """
    Return all unique links on the page at `url` as sorted absolute URLs.

    same_domain_only=True keeps only links on the same domain as the final
    (post-redirect) URL, treating `www.example.com` and `example.com` as the
    same site — handy for the brochure exercise so LinkedIn/Twitter/cookie
    links don't pollute the list.
    """
    final_url, html = _get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    base_domain = urlparse(final_url).netloc.lower().removeprefix("www.")

    links = set()
    for a in soup.find_all("a", href=True):
        absolute_url = urljoin(final_url, a["href"])
        absolute_url, _fragment = urldefrag(absolute_url)  # drop #anchors

        if not absolute_url.startswith(("http://", "https://")):
            continue  # skips mailto:, tel:, javascript:, data: ...

        if same_domain_only:
            link_domain = urlparse(absolute_url).netloc.lower().removeprefix("www.")
            if link_domain != base_domain:
                continue

        links.add(absolute_url)

    return sorted(links)


def fetch_website_links_with_text(url: str, same_domain_only: bool = False) -> list:
    """
    Like `fetch_website_links`, but keeps each link's anchor text.

    Returns a list of {"text": ..., "url": ...} dicts, sorted by URL. The
    anchor text is what a human sees on the page (e.g. "Leica Absolute
    Tracker ATS800"), which is far better product-naming material than the
    URL slug. Where the same URL appears more than once, the longest
    non-empty anchor text wins — nav bars tend to use short labels while
    content cards spell the full name out.
    """
    final_url, html = _get_html(url)
    soup = BeautifulSoup(html, "html.parser")

    base_domain = urlparse(final_url).netloc.lower().removeprefix("www.")

    best = {}
    for a in soup.find_all("a", href=True):
        absolute_url = urljoin(final_url, a["href"])
        absolute_url, _fragment = urldefrag(absolute_url)

        if not absolute_url.startswith(("http://", "https://")):
            continue

        if same_domain_only:
            link_domain = urlparse(absolute_url).netloc.lower().removeprefix("www.")
            if link_domain != base_domain:
                continue

        text = " ".join(a.get_text(separator=" ", strip=True).split())
        if len(text) > len(best.get(absolute_url, "")):
            best[absolute_url] = text

    return [{"text": best[u], "url": u} for u in sorted(best)]

if __name__ == "__main__":
    url = "https://hexagon.com"

    mode = "curl_cffi (Chrome TLS impersonation)" if _IMPERSONATING else "plain requests"
    print(f"[scraper] HTTP mode: {mode}\n")

    print(fetch_website_contents(url))

    all_links = fetch_website_links(url)
    internal_links = fetch_website_links(url, same_domain_only=True)

    print(f"\nFound {len(all_links)} links total, {len(internal_links)} internal.")
    print("\n Internal links:")
    for link in internal_links:
        print(" ", link)