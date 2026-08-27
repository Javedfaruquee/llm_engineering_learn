import requests

from urllib.parse import urlencode, urlparse, parse_qs

from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

IGNORED_TAGS = ["script", "style", "img", "input", "noscript", "svg"]


class Website:
    """Title and readable body text scraped from a single web page."""

    def __init__(self, url, title, text):
        self.url = url
        self.title = title
        self.text = text


def _google_news_rss_url(url):
    """Map any news.google.com page onto its RSS equivalent, keeping the locale."""
    query = parse_qs(urlparse(url).query)
    params = {key: query[key][0] for key in ("hl", "gl", "ceid") if key in query}
    return "https://news.google.com/rss" + (f"?{urlencode(params)}" if params else "")


def _fetch_google_news(url, timeout):
    """Google News renders headlines in JavaScript and gates the HTML behind a
    cookie-consent page, so read the server-rendered RSS feed instead."""
    response = requests.get(_google_news_rss_url(url), headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    channel = soup.find("channel")
    title = channel.title.get_text(strip=True) if channel and channel.title else "Google News"

    headlines = [
        item.title.get_text(strip=True)
        for item in soup.find_all("item")
        if item.title and item.title.get_text(strip=True)
    ]

    return Website(url, title, "\n".join(headlines))


def fetch_website_contents(url, timeout=20):
    if urlparse(url).netloc.endswith("news.google.com"):
        return _fetch_google_news(url, timeout)

    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "No title found"

    if soup.body:
        for tag in soup.body(IGNORED_TAGS):
            tag.decompose()
        text = soup.body.get_text(separator="\n", strip=True)
    else:
        text = ""

    return Website(url, title, text)

def fetch_website_links(url):
    """
    Return the links on the webiste at the given url
    I realize this is inefficient as we're parsing twice! This is to keep the code in the lab simple.
    Feel free to use a class and optimize it!
    """
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.content, "html.parser")
    links = [link.get("href") for link in soup.find_all("a")]
    return [link for link in links if link]