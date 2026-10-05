"""
SignPath documentation RAG assistant.

Crawls the SignPath documentation (https://docs.signpath.io/) and knowledge base
(https://signpath.io/knowledge-base), extracts text, code examples and images, indexes them in a
Chroma vector DB, and answers questions with hybrid retrieval (vector + BM25 keyword search, fused and
re-ranked by a cross-encoder) and an LLM that is instructed to answer from the documentation alone.
(signpath_Ingest.py + signpath_answer.py are the same program split into two simpler files.)

How a question is answered:
  1. A follow-up ("and how do I test it?") is rewritten into a standalone question using the chat history.
  2. Vector search (meaning) and BM25 keyword search (exact names) each return candidate chunks;
     reciprocal-rank fusion merges the two rankings.
  3. A cross-encoder re-ranks the candidates. If even the best one is unrelated, the assistant refuses
     without calling the LLM (the "relevance gate"), so it can't make things up.
  4. The best chunks go to the LLM with strict rules: answer only from them, copy code exactly, cite [n].
  5. The cited sections' images are shown under the answer.

Usage (run with the project's .venv):
    python week5/Signpath_RAG.py                 # launch the Gradio UI (crawls + indexes on first run)
    python week5/Signpath_RAG.py --crawl         # re-crawl the docs, rebuild the whole index, then launch
    python week5/Signpath_RAG.py --reindex       # rebuild the whole index from the saved crawl, then launch
    python week5/Signpath_RAG.py --refresh       # re-crawl and update only what changed, then exit
    python week5/Signpath_RAG.py --refresh --eval  # ...then check known questions still answer correctly
    python week5/Signpath_RAG.py --eval          # check the answers to known questions
    python week5/Signpath_RAG.py --eval-retrieval  # measure search quality: MRR, nDCG@K, Recall@K, Precision@K
    python week5/Signpath_RAG.py --ask "How do I install the Windows KSP?"   # answer in the terminal
    docker exec signpath-rag python Signpath_RAG.py --refresh --eval # for dockerized referesh
Keeping the data fresh: while the UI runs, the docs are re-crawled every SIGNPATH_REFRESH_HOURS (default 360).
Only new or changed sections are re-embedded, a crawl that finds far fewer pages than before is rejected,
and the running app reloads the index on its next question. See week5/Dockerfile to run it as a container.
"""

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import warnings
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from dotenv import load_dotenv
from openai import OpenAI

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Sites to crawl: every page on `host` whose path is under `prefix` is followed and indexed
SOURCES = [
    {"name": "SignPath Documentation", "start": "https://docs.signpath.io/", "host": "docs.signpath.io", "prefix": "/"},
    {"name": "SignPath Knowledge Base", "start": "https://signpath.io/knowledge-base", "host": "signpath.io",
     "prefix": "/knowledge-base"},
]
IMAGE_HOSTS = {"docs.signpath.io", "signpath.io", "framerusercontent.com"}  # framerusercontent hosts the knowledge base's images
USER_AGENT = "SignPath-RAG-learning-crawler/1.0"
MAX_PAGES = 2000
CRAWL_DELAY = 0.3  # seconds between requests, to be polite to the docs server

DATA_DIR = Path(os.getenv("SIGNPATH_DATA_DIR", Path(__file__).parent / "signpath_data"))
PAGES_FILE = DATA_DIR / "pages.json"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_INFO_FILE = DATA_DIR / "index_info.json"
COLLECTION = "signpath_docs"

REFRESH_HOURS = float(os.getenv("SIGNPATH_REFRESH_HOURS", "360"))  # background re-crawl interval while the UI runs; 0 = off
MIN_PAGE_RATIO = 0.8     # a refresh that finds fewer than 80% of the previous pages is treated as a failed crawl
OPEN_BROWSER = os.getenv("SIGNPATH_OPEN_BROWSER", "1") == "1"

LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_CHARS = 1800       # target chunk size; code blocks are never split unless huge
VECTOR_K = 25            # candidates from vector search
KEYWORD_K = 25           # candidates from BM25 keyword search
RERANK_CANDIDATES = 30   # fused candidates passed to the cross-encoder
CONTEXT_K = 8            # excerpts given to the LLM
RERANK_MIN_SCORE = -4.0  # if the best re-rank score is below this, the docs don't cover the question
RERANK_KEEP_SCORE = -8.0 # excerpts scoring below this are never given to the LLM
MIN_SIMILARITY = 0.30    # fallback gate on cosine similarity when the re-ranker is unavailable
MAX_IMAGES = 6           # documentation images shown with an answer
BLEND_K = 10             # how strongly top positions count when blending re-rank order with search order
CHANGELOG_SECTION = "Product updates"   # changelog entries are only searched for release/version questions
CHANGELOG_QUESTION = re.compile(r"\b(release|released|version|versions|changelog|new in|what's new|"
                                r"update|updated|deprecat\w*)\b", re.I)

NO_ANSWER = "I don't have information about that in the SignPath documentation I have indexed."

SKIP_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".msi", ".exe",
    ".xml", ".json", ".css", ".js", ".txt", ".rss", ".atom", ".mp4", ".woff", ".woff2", ".ttf",
}


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Crawling and HTML -> markdown extraction
# ---------------------------------------------------------------------------

def source_for(url):
    """The SOURCES entry a URL belongs to, or None."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    for source in SOURCES:
        prefix = source["prefix"]
        if parsed.netloc == source["host"] and (prefix == "/" or path == prefix or path.startswith(prefix + "/")):
            return source
    return None


def normalize_url(url):
    """Canonical form of a page URL, or None if it's outside the crawled sources or not a page."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not source_for(url):
        return None
    path = re.sub(r"/index(\.html?)?$", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    if Path(path).suffix.lower() in SKIP_EXTENSIONS or "/feeds/" in path:  # feeds repeat /changelog as raw XML
        return None
    return f"https://{parsed.netloc}{path}"


def code_language(pre):
    for element in [pre, *list(pre.parents)[:3]]:
        for cls in element.get("class") or []:
            if cls.startswith("language-"):
                lang = cls[len("language-"):]
                return "" if lang == "plaintext" else lang
    return ""


class MarkdownConverter:
    """Converts a docs <article> into markdown, keeping code blocks, tables, callouts and images."""

    def __init__(self, page_url):
        self.page_url = page_url
        self.images = []  # images found in the block currently being converted

    def convert(self, node):
        if isinstance(node, Comment):
            return ""
        if isinstance(node, NavigableString):
            return re.sub(r"\s+", " ", str(node))
        if not isinstance(node, Tag):
            return ""

        name = node.name
        classes = node.get("class") or []
        if name in ("script", "style", "noscript", "button", "svg", "form"):
            return ""
        if name == "pre":
            code = node.get_text().strip("\n")
            return f"\n\n```{code_language(node)}\n{code}\n```\n\n"
        if re.fullmatch(r"h[1-6]", name):
            return f"\n\n{'#' * int(name[1])} {node.get_text(' ', strip=True)}\n\n"
        if name == "img":
            src = node.get("src")
            if not src:
                return ""
            alt = (node.get("alt") or node.get("title") or "").strip()
            self.images.append({"src": urljoin(self.page_url, src), "alt": alt})
            return f" [Image: {alt or 'screenshot'}] "
        if name == "br":
            return "\n"
        if name == "code":
            return f"`{node.get_text()}`"
        if name == "table":
            return self.table(node)
        if name in ("ul", "ol"):
            return self.list(node, ordered=(name == "ol"))
        if name == "blockquote" or (name == "div" and "panel" in classes):
            label = next((c for c in classes if c in ("info", "tip", "warning", "note")), "")
            inner = self.children(node).strip()
            if label:
                inner = f"**{label.capitalize()}:** {inner}"
            return "\n\n" + "\n".join(f"> {line}" for line in inner.splitlines()) + "\n\n"
        if name in ("p", "div", "section", "dl", "dd", "dt", "figure", "figcaption"):
            return "\n\n" + self.children(node).strip() + "\n\n"
        return self.children(node)

    def children(self, node):
        return "".join(self.convert(child) for child in node.children)

    def list(self, node, ordered):
        lines = []
        for i, item in enumerate(node.find_all("li", recursive=False), start=1):
            content = re.sub(r"\n{3,}", "\n\n", self.children(item).strip())
            bullet = f"{i}. " if ordered else "- "
            indented = content.replace("\n", "\n" + " " * len(bullet))
            lines.append(bullet + indented)
        return "\n\n" + "\n".join(lines) + "\n\n"

    def table(self, node):
        rows = []
        for tr in node.find_all("tr"):
            cells = [
                re.sub(r"\s+", " ", self.children(cell)).strip().replace("|", "\\|")
                for cell in tr.find_all(["th", "td"], recursive=False)
            ]
            if cells:
                rows.append(cells)
        if not rows:
            return ""
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        return "\n\n" + "\n".join(lines) + "\n\n"


def clean_markdown(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


SECTION_HEADINGS = ("h1", "h2", "h3", "h4")  # some docs pages use h1 for sections inside the article


def content_blocks(node):
    """Children in document order, descending into wrappers that contain headings (Framer nests them deeply)."""
    for child in node.children:
        if isinstance(child, Tag) and child.name not in SECTION_HEADINGS and child.find(SECTION_HEADINGS):
            yield from content_blocks(child)
        else:
            yield child


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def sectionize(root, url, breadcrumb, anchor_of):
    """Split content into sections (one per h1-h4 heading) of markdown text, dropping repeated sections."""
    converter = MarkdownConverter(url)
    sections, seen = [], set()
    headings = {}
    current = {"path": [breadcrumb], "anchor": "", "parts": [], "images": []}

    def flush():
        text = clean_markdown("".join(current["parts"]))
        images = list({img["src"]: img for img in current["images"]}.values())
        key = (tuple(current["path"]), text)
        if (text or images) and key not in seen:
            seen.add(key)
            sections.append({"path": current["path"], "anchor": current["anchor"], "text": text, "images": images})

    for child in content_blocks(root):
        if isinstance(child, Tag) and child.name in SECTION_HEADINGS:
            flush()
            level = int(child.name[1])
            headings = {lvl: txt for lvl, txt in headings.items() if lvl < level}
            headings[level] = child.get_text(" ", strip=True)
            path = [breadcrumb] + [headings[lvl] for lvl in sorted(headings)]
            current = {"path": path, "anchor": anchor_of(child), "parts": [], "images": []}
        else:
            converter.images = []
            current["parts"].append(converter.convert(child))
            current["images"].extend(converter.images)
    flush()
    return sections


def extract_docs_page(soup, url):
    """docs.signpath.io (Jekyll): content is a flat <article> inside <main>.
    Returns None for anything that isn't a docs page (redirect stubs)."""
    main = soup.find("main")
    article = main.find("article") if main else None
    if article is None:
        return None
    title = soup.title.get_text(strip=True).split(" | ")[0] if soup.title else url
    # The page header holds a breadcrumb like "Crypto Providers ❯ Windows KSP"; some pages (the FAQ)
    # have an empty header, so fall back to the page title
    header = soup.select_one("main h1")
    breadcrumb = header.get_text(" ", strip=True).replace("❯", ">") if header else ""
    breadcrumb = re.sub(r"\s+", " ", breadcrumb) or title
    sections = sectionize(article, url, breadcrumb, lambda h: h.get("id", ""))
    return {"url": url, "title": title, "breadcrumb": breadcrumb, "sections": sections}


def extract_framer_page(soup, url):
    """signpath.io/knowledge-base (Framer): parts of the page are rendered once per screen size with all but one
    copy hidden, and headings are nested deep in <div>s. The content is the smallest block holding a title (h1)
    and section headings, which picks a single copy; the navigation before the title is dropped."""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = soup.title.get_text(strip=True).split(" - ")[-1] if soup.title else ""
    candidates = []
    for h1 in soup.find_all("h1"):
        root = next((a for a in h1.parents if a.find("h2")), None)
        if root is not None:
            # Pages can also contain other pages' titles (previews); prefer the h1 that matches the <title>
            match = difflib.SequenceMatcher(None, h1.get_text(" ", strip=True).lower(), title.lower()).ratio()
            candidates.append((round(match, 1), -len(root.get_text(" ", strip=True)), h1, root))
    if not candidates:
        return None
    *_, h1, root = max(candidates, key=lambda c: (c[0], c[1]))
    node = h1  # remove everything before the title: the page index and "On this page" links
    while node is not None and node is not root:
        for sibling in list(node.previous_siblings):
            sibling.extract()
        node = node.parent

    page_title = h1.get_text(" ", strip=True)
    h1.extract()  # it's the breadcrumb, not content
    title = title or page_title
    breadcrumb = f"Knowledge Base > {page_title}"
    sections = sectionize(root, url, breadcrumb, lambda h: slugify(h.get_text(" ", strip=True)))
    return {"url": url, "title": f"Knowledge Base: {title}", "breadcrumb": breadcrumb, "sections": sections}


def extract_page(soup, url):
    source = source_for(url)
    extractor = extract_framer_page if source and source["host"] == "signpath.io" else extract_docs_page
    page = extractor(soup, url)
    if page:
        page["source"] = source["name"]
    return page


SVG_FALLBACK_FONTS = "Carlito, DejaVu Sans, Arial, Segoe UI"


def svg_to_png(svg_bytes):
    """Diagrams are rendered to PNG on a white background: Gradio serves cached SVGs as downloads (so they
    show as broken images), and the docs' SVGs are transparent with black text, unreadable in dark mode."""
    try:
        import resvg_py
        svg = svg_bytes.decode("utf-8")
        # The diagrams ask for Calibri; add fallbacks that exist on Linux (Carlito, DejaVu) so text isn't dropped
        svg = re.sub(r'font-family="([^"]*)"', rf'font-family="\1, {SVG_FALLBACK_FONTS}"', svg)
        svg = re.sub(r"font-family:\s*([^;\"']*)", rf"font-family: \1, {SVG_FALLBACK_FONTS}", svg)
        png = resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", zoom=1.5)
        return bytes(png), ".png"
    except Exception as error:
        log(f"  ! SVG to PNG conversion failed ({error}); keeping the SVG")
        return svg_bytes, ".svg"


def flatten_on_white(data, ext):
    """Put images with transparent areas on white: in a dark-themed UI, black text on transparency is unreadable."""
    try:
        from io import BytesIO
        from PIL import Image
        with Image.open(BytesIO(data)) as image:
            if not (image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)):
                return data, ext
            rgba = image.convert("RGBA")
            if rgba.getchannel("A").getextrema()[0] == 255:
                return data, ext  # has an alpha channel but nothing is transparent
            canvas = Image.new("RGB", rgba.size, "white")
            canvas.paste(rgba, mask=rgba.getchannel("A"))
            output = BytesIO()
            canvas.save(output, format="PNG")
            return output.getvalue(), ".png"
    except Exception as error:
        log(f"  ! could not flatten image ({error}); keeping it as is")
        return data, ext


def download_image(session, src, downloaded):
    """Save an image locally once; returns its path relative to DATA_DIR, or None."""
    if src in downloaded:
        return downloaded[src]
    downloaded[src] = None
    parsed = urlparse(src)
    if parsed.netloc not in IMAGE_HOSTS:
        return None  # external badges (marketplace, PowerShell Gallery) aren't documentation content
    try:
        # Framer image URLs carry resizing parameters (e.g. ?scale-down-to=512); fetch the full-size original
        response = session.get(parsed._replace(query="").geturl() if parsed.netloc == "framerusercontent.com" else src,
                               timeout=20)
        content_type = response.headers.get("content-type", "")
        if response.status_code != 200 or not content_type.startswith("image/"):
            return None
        ext = Path(urlparse(src).path).suffix.lower()
        if ext not in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"):
            ext = "." + content_type.split("/")[1].split("+")[0].split(";")[0]
        data = response.content
        if ext == ".svg":
            data, ext = svg_to_png(data)
        else:
            data, ext = flatten_on_white(data, ext)
        name = hashlib.sha1(src.encode()).hexdigest()[:12] + "_" + Path(urlparse(src).path).stem[:60] + ext
        (IMAGES_DIR / name).write_bytes(data)
        downloaded[src] = f"images/{name}"
    except requests.RequestException as error:
        log(f"  ! image failed {src}: {error}")
    return downloaded[src]


def sitemap_entries(session, host):
    """{normalized url: lastmod date} for the pages in a host's sitemap that belong to a crawled source."""
    try:
        xml = session.get(f"https://{host}/sitemap.xml", timeout=20).text
    except requests.RequestException:
        return {}
    entries = {}
    for block in re.findall(r"<url>(.*?)</url>", xml, re.DOTALL):
        loc = re.search(r"<loc>\s*([^<\s]+)\s*</loc>", block)
        lastmod = re.search(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", block)
        url = normalize_url(loc.group(1)) if loc else None
        if url:
            entries[url] = lastmod.group(1)[:10] if lastmod else ""
    return entries


def crawl(max_pages=MAX_PAGES):
    """Breadth-first crawl of every source (seeded with their sitemaps), honouring each site's robots.txt.
    Returns the pages; the caller decides whether to save them (see save_pages)."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    robots, lastmods = {}, {}
    for host in dict.fromkeys(source["host"] for source in SOURCES):
        parser = RobotFileParser(f"https://{host}/robots.txt")
        try:
            parser.read()
            robots[host] = parser
        except Exception:
            pass
        lastmods.update(sitemap_entries(session, host))

    # Start from each source's start page and every sitemap URL; `seen` holds every URL already queued
    # (so each page is fetched once), and links found on pages are added to the end of the queue
    queue = deque()
    seen = set()
    for url in [*(normalize_url(source["start"]) for source in SOURCES), *lastmods]:
        if url and url not in seen:
            seen.add(url)
            queue.append(url)

    pages, downloaded, fingerprints = [], {}, {}
    while queue and len(pages) < max_pages:
        url = queue.popleft()
        parser = robots.get(urlparse(url).netloc)
        if parser and not parser.can_fetch(USER_AGENT, url):
            log(f"  - skipped (robots.txt) {url}")
            continue
        try:
            response = session.get(url, timeout=20)
        except requests.RequestException as error:
            log(f"  ! failed {url}: {error}")
            continue
        time.sleep(CRAWL_DELAY)

        # Only successful HTML pages inside our sources count; follow server redirects to their target
        final_url = normalize_url(response.url)
        if response.status_code != 200 or not final_url or "text/html" not in response.headers.get("content-type", ""):
            continue
        if final_url != url:
            if final_url in seen:
                continue  # redirected to a page that is already crawled or queued
            seen.add(final_url)

        # signpath.io doesn't say which character set its pages use, so requests would assume Latin-1 and
        # turn characters like ’ and – into garbage ("â€™"). Both sites actually send UTF-8.
        if "charset" not in response.headers.get("content-type", ""):
            response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "lxml")

        # Queue every in-scope link on the page (plus the target of an HTML redirect page)
        links = [link["href"] for link in soup.find_all("a", href=True)]
        refresh = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
        if refresh and "url=" in refresh.get("content", "").lower():
            links.append(refresh["content"].split("=", 1)[1].strip("'\" "))  # HTML redirect stub
        for href in links:
            target = normalize_url(urljoin(final_url, href))
            if target and target not in seen:
                seen.add(target)
                queue.append(target)
        if refresh:
            continue  # a redirect stub: its target is queued, the stub itself has no content

        # Extract the page's sections; skip pages without content (e.g. layouts with no article)
        page = extract_page(soup, final_url)
        if not page or not any(s["text"] for s in page["sections"]):
            continue
        # The fingerprint (hash of the content) catches the same page under two URLs, and is later compared
        # with the previous crawl to see which pages changed (compare_pages)
        fingerprint = hashlib.sha1(json.dumps(page["sections"]).encode()).hexdigest()
        if fingerprint in fingerprints:
            log(f"  - skipped (duplicate of {fingerprints[fingerprint]}) {final_url}")
            continue
        fingerprints[fingerprint] = final_url
        for section in page["sections"]:
            for image in section["images"]:
                image["path"] = download_image(session, image["src"], downloaded)
        page["lastmod"] = lastmods.get(final_url, "")
        page["hash"] = fingerprint
        pages.append(page)
        log(f"  [{len(pages):3}] {final_url}")

    image_count = sum(1 for path in downloaded.values() if path)
    log(f"Crawled {len(pages)} pages and saved {image_count} images")
    return pages


def load_pages():
    return json.loads(PAGES_FILE.read_text(encoding="utf-8")) if PAGES_FILE.exists() else []


def save_pages(pages):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = PAGES_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")
    temp.replace(PAGES_FILE)


def compare_pages(old_pages, new_pages):
    """Which pages were added, changed or removed between two crawls."""
    old = {p["url"]: p.get("hash") for p in old_pages}
    new = {p["url"]: p.get("hash") for p in new_pages}
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": sorted(url for url in set(old) & set(new) if old[url] != new[url]),
    }


# ---------------------------------------------------------------------------
# Chunking and indexing
# ---------------------------------------------------------------------------

CODE_FENCE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)


def split_blocks(text):
    """Split markdown into blocks at blank lines, never inside a fenced code block."""
    blocks, current, in_fence = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_oversized(block, limit):
    """Split a block that is far too large (e.g. a huge XML reference) into fenced pieces."""
    match = CODE_FENCE.fullmatch(block.strip())
    lang, body = (match.group(1), match.group(2)) if match else ("", block)
    pieces, current = [], []
    for line in body.splitlines():
        if current and sum(len(l) + 1 for l in current) + len(line) > limit:
            pieces.append(current)
            current = []
        current.append(line)
    if current:
        pieces.append(current)
    if match:
        return [f"```{lang}\n" + "\n".join(p) + "\n```" for p in pieces]
    return ["\n".join(p) for p in pieces]


def split_text(text, limit=CHUNK_CHARS):
    blocks = []
    for block in split_blocks(text):
        blocks.extend(split_oversized(block, limit) if len(block) > limit * 2.5 else [block])
    chunks, current = [], ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > limit:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks


def build_chunks(pages):
    """Chunks with content-based ids, so an unchanged section keeps its id (and embedding) across crawls."""
    chunks = {}

    def add(kind, text, meta):
        chunk_id = hashlib.sha1(f"{kind}\n{text}".encode()).hexdigest()[:24]
        chunks.setdefault(chunk_id, {"id": chunk_id, "text": text, "meta": meta})

    for page_number, page in enumerate(pages):
        for number, section in enumerate(page["sections"]):
            url = page["url"] + (f"#{section['anchor']}" if section["anchor"] else "")
            section_name = " > ".join(section["path"])
            source = page.get("source", SOURCES[0]["name"])
            header = f"Source: {source}\nPage: {page['title']}\nSection: {section_name}\nURL: {url}\n\n"
            section_images = [{"path": img["path"], "alt": img["alt"]} for img in section["images"] if img.get("path")]
            base = {"url": url, "page": page["title"], "section": section_name, "images": json.dumps(section_images),
                    "lastmod": page.get("lastmod", ""),
                    "position": page_number * 1000 + number}  # document order, used to sort images

            # Text chunks: the section's prose, code and tables together
            for piece in split_text(section["text"]):
                add("text", header + piece, {**base, "type": "text", "lang": ""})

            # Code chunks: each example with the sentence that introduces it, so "show me an example" ranks well
            for match in CODE_FENCE.finditer(section["text"]):
                lang, code = match.group(1), match.group(2)
                if len(code) < 40:
                    continue
                before = CODE_FENCE.sub("", section["text"][:match.start()]).strip()
                intro = before.split("\n\n")[-1][-600:] if before else ""
                body = f"Example ({lang or 'code'}):\n{intro}\n\n```{lang}\n{code[:CHUNK_CHARS * 3]}\n```"
                add("code", header + body, {**base, "type": "code", "lang": lang})

            # Image chunks: alt text plus the surrounding section text
            prose = CODE_FENCE.sub("", section["text"]).strip()[:700]
            for image in section_images:
                body = f"Image: {image['alt'] or 'screenshot'}\nShown in section: {section_name}\n\n{prose}"
                add("image", header + body, {**base, "type": "image", "lang": "", "images": json.dumps([image])})
    return list(chunks.values())


def embed(client, texts):
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    texts = [encoding.decode(encoding.encode(t)[:8000]) for t in texts]
    vectors = []
    for start in range(0, len(texts), 100):
        response = client.embeddings.create(model=EMBED_MODEL, input=texts[start:start + 100])
        vectors.extend(item.embedding for item in response.data)
    return vectors


def chroma_client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))


def read_index_info():
    try:
        return json.loads(INDEX_INFO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def now_utc():
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def sync_index(client, pages, rebuild=False, page_changes=None):
    """Bring the vector index in line with `pages`: embed only new chunks and delete stale ones.
    A full rebuild happens when asked, or when the embedding model differs from the one the index was built with."""
    chunks = build_chunks(pages)
    info = read_index_info()
    db = chroma_client()
    exists = COLLECTION in [c.name for c in db.list_collections()]
    # Vectors from different embedding models can't be compared, so a model change means starting over
    if exists and (rebuild or info.get("embed_model") != EMBED_MODEL):
        db.delete_collection(COLLECTION)
        exists = False
    collection = db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    # Chunk ids are hashes of their content (see build_chunks), so comparing ids is enough to know what changed:
    # ids only in the new crawl are new or edited text (embed them), ids only in the index are gone (delete them)
    existing = set(collection.get(include=[])["ids"]) if exists else set()
    new_chunks = [c for c in chunks if c["id"] not in existing]
    stale = sorted(existing - {c["id"] for c in chunks})

    if stale:
        collection.delete(ids=stale)
    if new_chunks:
        counts = Counter(c["meta"]["type"] for c in new_chunks)
        log(f"Embedding {len(new_chunks)} new/changed chunks ({counts['text']} text, {counts['code']} code, {counts['image']} image)...")
        vectors = embed(client, [c["text"] for c in new_chunks])
        for start in range(0, len(new_chunks), 500):
            batch = new_chunks[start:start + 500]
            collection.add(ids=[c["id"] for c in batch], documents=[c["text"] for c in batch],
                           embeddings=vectors[start:start + 500], metadatas=[c["meta"] for c in batch])
    unchanged = [c for c in chunks if c["id"] in existing]
    for start in range(0, len(unchanged), 500):  # page order and last-modified dates may change without the text changing
        batch = unchanged[start:start + 500]
        collection.update(ids=[c["id"] for c in batch], metadatas=[c["meta"] for c in batch])

    # index_info.json records what's in the index; a running UI watches "last_changed" to know when to reload
    checked = now_utc()
    changed = bool(new_chunks or stale) or not info
    info = {
        "embed_model": EMBED_MODEL,
        "pages": len(pages),
        "chunks": collection.count(),
        "last_checked": checked,
        "last_checked_ts": time.time(),
        "last_changed": checked if changed else info.get("last_changed", checked),
        "newest_page_update": max((p.get("lastmod", "") for p in pages), default=""),
        "last_sync": {"added_chunks": len(new_chunks), "removed_chunks": len(stale), **(page_changes or {})},
    }
    INDEX_INFO_FILE.write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"Index in sync: {info['chunks']} chunks ({len(new_chunks)} added, {len(stale)} removed)")
    return info


def prune_images(pages):
    """Delete downloaded images that no page references any more."""
    referenced = {img["path"] for p in pages for s in p["sections"] for img in s["images"] if img.get("path")}
    for file in IMAGES_DIR.glob("*"):
        if f"images/{file.name}" not in referenced:
            file.unlink(missing_ok=True)


def refresh(client, max_pages=MAX_PAGES):
    """Re-crawl the docs and update the index incrementally. Keeps the current data if the crawl looks broken."""
    log(f"Refreshing from {', '.join(s['start'] for s in SOURCES)} ...")
    old_pages = load_pages()
    new_pages = crawl(max_pages)
    if old_pages and len(new_pages) < MIN_PAGE_RATIO * len(old_pages):
        log(f"! Crawl found only {len(new_pages)} pages (previously {len(old_pages)}); keeping the existing index")
        return None
    changes = compare_pages(old_pages, new_pages)
    log(f"Pages: {len(changes['added'])} added, {len(changes['changed'])} changed, {len(changes['removed'])} removed")
    for kind in ("added", "changed", "removed"):
        for url in changes[kind]:
            log(f"  {kind}: {url}")
    save_pages(new_pages)
    info = sync_index(client, new_pages, page_changes={k: len(v) for k, v in changes.items()})
    prune_images(new_pages)
    return info


# ---------------------------------------------------------------------------
# Retrieval: vector + BM25, reciprocal-rank fusion, cross-encoder re-ranking
# ---------------------------------------------------------------------------

STOPWORDS = set("""a an and are as at be by can do does for from how i in is it its me my of on or
should so that the this to use using what when where which who why will with you your""".split())


def tokenize(text):
    """Lower-case words without stopwords; names like 'pe-file' are kept whole and also split into 'pe' and 'file'."""
    tokens = []
    for token in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()):
        if token not in STOPWORDS:
            tokens.append(token)
        parts = re.split(r"[-_.]", token)
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


class BM25:
    """Keyword search. Complements the vector search by matching exact names ('certutil', '<pe-file>')."""

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.term_freqs = [Counter(tokenize(d)) for d in documents]   # per chunk: how often each word appears
        self.lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        doc_freq = Counter(term for tf in self.term_freqs for term in tf)  # per word: how many chunks contain it
        n = len(documents)
        # IDF: high for words found in few chunks (they say a lot), close to 0 for words found almost everywhere
        self.idf = {term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def top(self, query, k):
        """Positions of the k best chunks. For each query word in a chunk, BM25 adds
        idf * f * (k1 + 1) / (f + k1 * length_norm): more occurrences score higher with diminishing returns
        (k1), and long chunks are penalised a little (b) so a word in a short, focused chunk counts more."""
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        scores = []
        for i, tf in enumerate(self.term_freqs):
            score = 0.0
            for term in terms:
                f = tf.get(term, 0)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                    score += self.idf[term] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scores.append((score, i))
        return [i for _, i in sorted(scores, reverse=True)[:k]]


@dataclass
class Hit:
    """One retrieved chunk with its scores."""
    id: str
    text: str
    meta: dict                     # url, page, section, type, lang, images, lastmod, position
    similarity: float = 0.0        # cosine similarity from the vector search (0 if only BM25 found it)
    rerank: float | None = None    # cross-encoder score: above 0 relevant, below RERANK_MIN_SCORE unrelated
    fused_rank: int = 0            # position after vector + keyword fusion, before re-ranking (--eval-retrieval)


@dataclass
class IndexSnapshot:
    collection: object
    ids: list
    documents: list
    metas: list
    position: dict
    bm25: BM25


class Retriever:
    def __init__(self, client):
        self.client = client
        self.lock = threading.Lock()
        self.loaded_version = None
        self.load()
        try:
            from sentence_transformers import CrossEncoder
            log(f"Loading re-ranker {RERANK_MODEL}...")
            self.reranker = CrossEncoder(RERANK_MODEL)
        except Exception as error:
            log(f"Re-ranker unavailable ({error}); falling back to vector similarity only")
            self.reranker = None

    def load(self):
        """(Re)load the chunks and keyword index; called again whenever a refresh changes the index."""
        version = read_index_info().get("last_changed")
        collection = chroma_client().get_collection(COLLECTION)
        data = collection.get(include=["documents", "metadatas"])
        # One object swapped in a single assignment, so a search running during a reload sees old or new, never a mix
        self.index = IndexSnapshot(collection, data["ids"], data["documents"], data["metadatas"],
                                   {chunk_id: i for i, chunk_id in enumerate(data["ids"])}, BM25(data["documents"]))
        self.loaded_version = version
        log(f"Retriever loaded {len(data['ids'])} chunks (index changed {version})")

    def reload_if_changed(self):
        if read_index_info().get("last_changed") != self.loaded_version:
            with self.lock:
                if read_index_info().get("last_changed") != self.loaded_version:
                    self.load()

    def search(self, query):
        """Returns (relevant hits for the LLM, all ranked candidates for display)."""
        self.reload_if_changed()
        index = self.index

        # 1. Vector search: the chunks whose embeddings are closest in meaning to the question
        #    (cosine distance; similarity = 1 - distance). Good at paraphrases, weaker on exact names.
        query_vector = embed(self.client, [query])[0]
        result = index.collection.query(query_embeddings=[query_vector], n_results=VECTOR_K, include=["distances"])
        vector_ids = [cid for cid in result["ids"][0] if cid in index.position]
        similarity = {cid: 1 - d for cid, d in zip(result["ids"][0], result["distances"][0])}

        # 2. Keyword search (BM25): the chunks containing the question's exact words,
        #    e.g. 'Submit-SigningRequest' or '<pe-file>', which embeddings can blur
        keyword_ids = [index.ids[i] for i in index.bm25.top(query, KEYWORD_K)]

        # 3. Reciprocal-rank fusion: each list gives a chunk 1 / (60 + its position). A chunk near the top of
        #    either list scores well, near the top of both scores best. Only positions are used, so the two
        #    searches' different score scales don't matter; 60 is the usual constant that keeps the very
        #    first places from dominating.
        fused = Counter()
        for ranking in (vector_ids, keyword_ids):
            for rank, cid in enumerate(ranking):
                fused[cid] += 1 / (60 + rank)
        candidates = [cid for cid, _ in fused.most_common(RERANK_CANDIDATES)]
        hits = [Hit(cid, index.documents[index.position[cid]], index.metas[index.position[cid]],
                    similarity.get(cid, 0.0), fused_rank=rank) for rank, cid in enumerate(candidates)]

        # 3b. Leave out changelog entries unless the question is about releases or versions. The changelog
        #     ("Product updates") has hundreds of short entries that mention features in passing; they reached
        #     the top 5 for 15 of the 83 test questions and pushed the real how-to sections down.
        if not CHANGELOG_QUESTION.search(query):
            hits = [h for h in hits if not h.meta["section"].startswith(CHANGELOG_SECTION)]

        # 4. Re-rank with the cross-encoder. The searches above compare question and chunk separately; the
        #    cross-encoder reads both together, so it judges relevance far better (too slow for every chunk,
        #    so it only re-orders the top candidates). Scores: roughly > 0 relevant, < -4 unrelated.
        if self.reranker:
            scores = self.reranker.predict([(query, h.text[:2500]) for h in hits])
            for hit, score in zip(hits, scores):
                hit.rerank = float(score)
            hits.sort(key=lambda h: h.rerank, reverse=True)
            # Relevance gate: if even the best chunk is unrelated, the docs don't cover the question.
            # Returning no hits makes answer() refuse without asking the LLM, so it can't guess.
            if not hits or hits[0].rerank < RERANK_MIN_SCORE:
                return [], hits
            # 4b. Blend the re-ranker's order with the search order (both as 1 / (BLEND_K + position)) instead of
            #     letting the small re-ranker overrule the search completely. A chunk that both rate highly wins;
            #     one the re-ranker alone likes (e.g. a similar-sounding section of another page) no longer jumps
            #     ahead of it. In the tests this raised MRR from 0.906 to 0.932 and fixed both top-5 misses.
            blended = {h.id: 1 / (BLEND_K + rank) + 1 / (BLEND_K + h.fused_rank) for rank, h in enumerate(hits)}
            hits.sort(key=lambda h: blended[h.id], reverse=True)
            relevant = [h for h in hits if h.rerank >= RERANK_KEEP_SCORE]
        else:
            # Fallback if the re-ranker model couldn't load: rank and gate by vector similarity instead
            hits.sort(key=lambda h: h.similarity, reverse=True)
            if not hits or hits[0].similarity < MIN_SIMILARITY:
                return [], hits
            relevant = hits

        # 5. Pick up to CONTEXT_K hits for the LLM, best first, dropping code/image chunks whose content is
        #    already inside a selected text chunk (no point sending the same thing twice)
        selected = []
        for hit in relevant:
            body = hit.text.split("\n\n", 1)[-1]
            if hit.meta["type"] == "code":
                code = CODE_FENCE.search(body)
                if code and any(code.group(2)[:300] in s.text for s in selected):
                    continue
            if hit.meta["type"] == "image" and any(s.meta["url"] == hit.meta["url"] for s in selected):
                continue
            selected.append(hit)
            if len(selected) == CONTEXT_K:
                break
        return selected, hits


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are the SignPath documentation assistant. You answer questions about SignPath: \
code signing, projects, artifact configurations, signing policies, certificates, crypto providers, \
trusted build systems, origin verification, the PowerShell module, users and administration — and about \
code-signing concepts such as certificates, time stamping, Windows signing and private key storage.

Each excerpt names its source: "SignPath Documentation" (the product docs: how to set up and use SignPath) \
or "SignPath Knowledge Base" (general code-signing background). For how-to questions about SignPath itself, \
rely on the product documentation.

Follow these rules strictly:
1. Use ONLY the numbered documentation excerpts in the user's message. Do not use any prior knowledge \
about SignPath or other products, and never guess.
2. If the excerpts do not contain the information needed to answer, reply with exactly this sentence \
and nothing else: "{NO_ANSWER}"
3. If the excerpts answer only part of the question, answer that part and say clearly which part \
is not covered by the documentation.
4. When the excerpts contain relevant configuration, commands or code, include them as fenced code \
blocks, copied exactly. Never invent XML elements, attributes, parameters, switches, file paths, URLs \
or values that are not in the excerpts. If you adapt an example, say so and change only what the \
question requires.
5. Cite the excerpts you used by number in square brackets, e.g. [1] or [2][3], after the statements \
they support.
6. Be concise and precise. Use numbered steps for procedures.
7. Excerpts may contain image markers like [Image: ...]. The images from the excerpts you cite are shown \
to the user below your answer, so when one illustrates a step, refer to it by its caption \
(e.g. "see the screenshot 'Microsoft Entra ID - create app roles'"). Never describe what an image \
shows beyond its caption and the surrounding text.
8. Each excerpt shows when its page was last updated. If excerpts contradict each other, follow the \
most recently updated one and briefly mention the difference."""

CONDENSE_PROMPT = """Rewrite the user's latest message as a single standalone question about the SignPath \
documentation, resolving references like "it", "that" or "the previous example" from the conversation. \
Do not answer it. Output only the rewritten question."""


def message_text(message):
    """Plain text of a chat message (Gradio 6 stores content as a list of parts)."""
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content).split("\n\n**Sources:**")[0]


class SignPathAssistant:
    def __init__(self):
        self.client = OpenAI()
        self.retriever = Retriever(self.client)

    def standalone_question(self, question, history):
        """Turn a follow-up like 'and how do I test it?' into a full question, so the search knows what 'it' is.
        Uses the last 6 messages; the first question of a chat is used as it is."""
        if not history:
            return question
        transcript = "\n".join(f"{m['role']}: {message_text(m)[:1500]}" for m in history[-6:])
        response = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=0, seed=42,
            messages=[{"role": "system", "content": CONDENSE_PROMPT},
                      {"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest message: {question}"}],
        )
        return response.choices[0].message.content.strip()

    def answer(self, question, history=None):
        """Yields (partial answer, sources markdown, image gallery, context markdown) while streaming."""
        # 1. Search with a standalone version of the question
        query = self.standalone_question(question, history or [])
        relevant, candidates = self.retriever.search(query)
        context_md = format_retrieval(query, relevant, candidates)
        # 2. Nothing relevant (the relevance gate in search): refuse without calling the LLM, so it can't guess
        if not relevant:
            yield NO_ANSWER, "_No sufficiently relevant documentation was found, so the LLM was not asked._", [], context_md
            return

        # 3. Give the LLM the numbered excerpts (with each page's last-updated date) and stream its answer.
        #    temperature=0 and a fixed seed make the same question get the same answer.
        excerpts = "\n\n---\n\n".join(
            f"[{n}] Last updated: {hit.meta.get('lastmod') or 'unknown'}\n{hit.text}" for n, hit in enumerate(relevant, start=1))
        stream = self.client.chat.completions.create(
            model=LLM_MODEL, temperature=0, seed=42, stream=True,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Documentation excerpts:\n\n{excerpts}\n\n---\n\nQuestion: {query}"}],
        )
        answer = ""
        for event in stream:
            if event.choices and event.choices[0].delta.content:
                answer += event.choices[0].delta.content
                yield answer, "", [], context_md

        # 4. Find which excerpts the answer cited ([1], [2]...). An uncited "I don't have information" means the
        #    LLM found the excerpts didn't answer the question: show no sources then.
        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(relevant)})
        if answer.strip().startswith("I don't have information") and not cited:
            yield answer, "_The retrieved documentation did not contain the answer._", [], context_md
            return
        # 5. Append source links and return the cited sections' images (all excerpts if nothing was cited)
        used = [relevant[n - 1] for n in cited] or relevant
        links = []
        for n, hit in zip(cited or range(1, len(used) + 1), used):
            links.append(f"[{n}] [{hit.meta['section']}]({hit.meta['url']})")
        answer_with_sources = answer + "\n\n**Sources:** " + " · ".join(links)
        yield answer_with_sources, format_sources(relevant, cited), gallery(used), context_md


def format_sources(hits, cited):
    lines = ["### Sources"]
    for n, hit in enumerate(hits, start=1):
        mark = "✅" if n in cited else "▫️"
        score = f"re-rank {hit.rerank:.1f}" if hit.rerank is not None else f"similarity {hit.similarity:.2f}"
        updated = f", page updated {hit.meta['lastmod']}" if hit.meta.get("lastmod") else ""
        lines.append(f"{mark} **[{n}]** [{hit.meta['section']}]({hit.meta['url']}) — _{hit.meta['type']}, {score}{updated}_")
    return "\n\n".join(lines)


def format_retrieval(query, relevant, candidates):
    lines = [f"**Search query:** {query}", "", "| # | Used | Score | Type | Section |", "| --- | --- | --- | --- | --- |"]
    used = {h.id for h in relevant}
    for n, hit in enumerate(candidates[:15], start=1):
        score = f"{hit.rerank:.2f}" if hit.rerank is not None else f"{hit.similarity:.2f}"
        section = hit.meta["section"].replace("|", "\\|")
        lines.append(f"| {n} | {'yes' if hit.id in used else ''} | {score} | {hit.meta['type']} | {section} |")
    return "\n".join(lines)


def gallery(hits):
    """(local path, caption) for the documentation images in the given excerpts' sections."""
    images, seen = [], set()
    for hit in sorted(hits, key=lambda h: h.meta.get("position", 0)):
        for image in json.loads(hit.meta.get("images") or "[]"):
            path = DATA_DIR / image["path"]
            if image["path"] not in seen and path.exists():
                seen.add(image["path"])
                images.append((str(path), image["alt"] or hit.meta["section"]))
    return images[:MAX_IMAGES]


# ---------------------------------------------------------------------------
# Evaluation: known questions with expected keywords, or None when the assistant must refuse
# ---------------------------------------------------------------------------

EVAL_CASES = [
    ("How do I install the Windows KSP crypto provider?", ["msiexec"]),
    ("How do I verify that the SignPath KSP is registered?", ["certutil"]),
    ("Which Submit-SigningRequest parameter waits until signing is complete?", ["WaitForCompletion"]),
    ("What crypto providers does SignPath support?", ["Cryptoki", "KSP"]),
    ("Show the artifact configuration for deep-signing an MSI with nested executables.", ["<msi-file", "authenticode-sign"]),
    ("How do I set up directory synchronization with Microsoft Entra ID?", ["provisioning"]),
    ("How do I sign with GPG using SignPath?", ["gpg"]),
    ("How do I submit a signing request from GitHub Actions?", ["github"]),
    ("What are the risks of storing code signing keys in PFX files?", ["pfx"]),          # knowledge base
    ("What is Microsoft SmartScreen and how does it relate to code signing?", ["smartscreen"]),
    ("How can I create a self-signed test certificate?", ["self-signed"]),
    ("What are SignPath's pricing tiers?", None),
    ("How do I configure AWS KMS as a key store in SignPath?", None),
    ("What is the capital of France?", None),
]


def is_refusal(answer):
    return answer.strip().startswith("I don't have information") and "**Sources:**" not in answer


def run_eval(assistant):
    """Returns True when every case passes. Run it after a refresh to catch answers the new data broke."""
    failures = 0
    for question, keywords in EVAL_CASES:
        *_, (answer, *_rest) = assistant.answer(question)
        if keywords is None:
            passed, detail = is_refusal(answer), "refused" if is_refusal(answer) else "answered (should refuse)"
        else:
            missing = [k for k in keywords if k.lower() not in answer.lower()]
            passed = not is_refusal(answer) and not missing
            detail = "refused (should answer)" if is_refusal(answer) else (f"missing {missing}" if missing else "ok")
        failures += not passed
        log(f"  {'PASS' if passed else 'FAIL'}  {question}  ->  {detail}")
    log(f"Evaluation: {len(EVAL_CASES) - failures}/{len(EVAL_CASES)} passed")
    return failures == 0


# ---------------------------------------------------------------------------
# Retrieval evaluation: MRR, nDCG@K, Recall@K and Precision@K
# ---------------------------------------------------------------------------
#
# --eval (above) checks the final answers. --eval-retrieval checks the step before that: does the search
# find the documentation sections that answer a question, and how high does it rank them? If retrieval
# misses the right section, the LLM can't answer well however good the prompt is.
#
# signpath_retrieval_tests.json lists test questions, each with the sections that answer it. A retrieved
# chunk is "relevant" if it comes from one of those sections (or a sub-section of one).
# For each question, three ranked lists are scored:
#   1. "Vector + keyword" - the fused search results, before re-ranking
#   2. "After re-rank"    - the final order: cross-encoder scores blended with the search order (step 4b)
#   3. "Sent to the LLM"  - the hits finally given to the LLM (after the relevance gate and de-duplication)
# Comparing 1 and 2 shows how much the re-ranker helps; 3 shows what the LLM actually gets to read.
#
# The metrics (each between 0 and 1, higher is better; reported as the average over all questions):
#   MRR (mean reciprocal rank) - 1 / position of the first relevant chunk: 1.0 if it is first, 0.5 if
#                                second, 0.33 if third ... and 0 if no relevant chunk was found at all.
#   nDCG@K - looks at all relevant chunks in the top K and rewards higher positions more; 1.0 means the
#            ranking is as good as possible (all relevant chunks before any irrelevant ones).
#   Recall@K - share of the expected sections that appear at least once in the top K
#              ("did we find everything we need?").
#   Precision@K - share of the top K chunks that come from an expected section ("how much is noise?").
#              Low precision is normal when a question has only one relevant section and K is 5.
# This is the same evaluation as in signpath_answer.py, so both programs report the same numbers.

RETRIEVAL_TESTS_FILE = Path(__file__).parent / "signpath_retrieval_tests.json"
EVAL_K = 5                 # K for the search rankings; the "Sent to the LLM" list uses CONTEXT_K


def is_relevant(meta, relevant_sections):
    """True if a chunk (given by its metadata) comes from an expected section or one of its sub-sections."""
    return any(meta["section"] == expected or meta["section"].startswith(expected + " > ")
               for expected in relevant_sections)


def reciprocal_rank(flags):
    """flags has one True/False per ranked chunk (True = relevant). Returns 1 / position of the first True."""
    for position, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / position
    return 0.0


def ndcg_at_k(flags, k, total_relevant):
    """Normalised discounted cumulative gain of the top k.
    DCG: a relevant chunk at position p adds 1 / log2(p + 1), so position 1 adds 1.0, position 2 adds 0.63,
    position 5 adds 0.39. It is divided by the DCG of the best possible ranking (all relevant chunks first),
    where total_relevant is how many relevant chunks exist in the whole index."""
    dcg = sum(1.0 / math.log2(position + 1) for position, flag in enumerate(flags[:k], start=1) if flag)
    ideal_dcg = sum(1.0 / math.log2(position + 1) for position in range(1, min(total_relevant, k) + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def recall_at_k(hits, relevant_sections, k):
    """Share of the expected sections that have at least one chunk in the top k."""
    found = sum(1 for expected in relevant_sections if any(is_relevant(h.meta, [expected]) for h in hits[:k]))
    return found / len(relevant_sections)


def precision_at_k(flags, k):
    """Share of the top k chunks that are relevant. When fewer than k hits were returned (the LLM can get
    fewer than CONTEXT_K), it is the share of the returned hits."""
    top = flags[:k]
    return sum(top) / len(top) if top else 0.0


def score_ranking(hits, relevant_sections, k, total_relevant):
    """All four metrics for one ranked list of hits."""
    flags = [is_relevant(h.meta, relevant_sections) for h in hits]
    return {"MRR": reciprocal_rank(flags),             # MRR looks at the whole list, not just the top k
            "nDCG": ndcg_at_k(flags, k, total_relevant),
            "Recall": recall_at_k(hits, relevant_sections, k),
            "Precision": precision_at_k(flags, k)}


METRICS = ("MRR", "nDCG", "Recall", "Precision")


def evaluate_retrieval(assistant, k=EVAL_K, progress=None):
    """Run every test question through the search and score it. Used by --eval-retrieval and the Evaluation tab.
    Costs one small embedding call per question (no LLM calls). `progress(fraction, text)` is called per question.

    Answerable questions are scored with the four metrics at each stage. Out-of-scope questions (no relevant
    sections) check the relevance gate instead: the search should find nothing relevant for them, and it
    should NOT reject an answerable question ("false rejection" - the assistant would wrongly refuse)."""
    tests = json.loads(RETRIEVAL_TESTS_FILE.read_text(encoding="utf-8"))["tests"]
    retriever = assistant.retriever
    stages = [("Vector + keyword", k), ("After re-rank", k), ("Sent to the LLM", CONTEXT_K)]
    totals = {stage: Counter() for stage, _ in stages}
    by_category = {}                 # category -> list of MRR values (after re-rank)
    per_question, bad_labels = [], []
    out_of_scope = {"total": 0, "rejected": 0}
    false_rejections = 0

    for number, test in enumerate(tests, start=1):
        if progress:
            progress(number / len(tests), f"Question {number} of {len(tests)}")
        selected, candidates = retriever.search(test["question"])     # candidates sorted by re-rank score
        relevant_sections = test["relevant"]

        if not relevant_sections:    # out of scope: the relevance gate should reject it (no chunks for the LLM)
            out_of_scope["total"] += 1
            out_of_scope["rejected"] += not selected
            per_question.append({"question": test["question"], "category": test["category"],
                                 "result": "rejected ✓" if not selected else "NOT rejected ✗"})
            continue

        # How many chunks in the whole index belong to the expected sections (nDCG's best case)
        total_relevant = sum(1 for meta in retriever.index.metas if is_relevant(meta, relevant_sections))
        if total_relevant == 0:
            bad_labels.append(relevant_sections)   # a typo in the test file, or the docs were reorganised
        false_rejections += not selected

        rankings = {"Vector + keyword": sorted(candidates, key=lambda h: h.fused_rank),
                    "After re-rank": candidates,
                    "Sent to the LLM": selected}
        for stage, stage_k in stages:
            totals[stage].update(score_ranking(rankings[stage], relevant_sections, stage_k, total_relevant))
        after = score_ranking(candidates, relevant_sections, k, total_relevant)
        by_category.setdefault(test["category"], []).append(after["MRR"])
        per_question.append({"question": test["question"], "category": test["category"],
                             "MRR": after["MRR"], "Recall": after["Recall"]})

    answerable = len(tests) - out_of_scope["total"]
    return {
        "k": k,
        "answerable": answerable,
        "stages": [(stage, stage_k, {m: totals[stage][m] / answerable for m in METRICS}) for stage, stage_k in stages],
        "by_category": {category: sum(values) / len(values) for category, values in sorted(by_category.items())},
        "per_question": per_question,
        "out_of_scope": out_of_scope,
        "false_rejections": false_rejections,
        "bad_labels": bad_labels,
    }


def run_retrieval_eval(assistant, k=EVAL_K):
    """--eval-retrieval: print the evaluation in the terminal."""
    log(f"Retrieval evaluation with {RETRIEVAL_TESTS_FILE.name} (K = {k})")
    result = evaluate_retrieval(assistant, k)
    for labels in result["bad_labels"]:
        log(f"  ! no chunk in the index belongs to {labels}; check the test file")

    log("Answerable questions, after re-ranking:\n   MRR  Recall  Category                 Question")
    for row in result["per_question"]:
        if "MRR" in row:
            warning = "   <- relevant section not in the top K" if row["Recall"] < 1 else ""
            log(f"  {row['MRR']:4.2f}   {row['Recall']:4.2f}  {row['category']:24} {row['question']}{warning}")
    log("Out-of-scope questions (the relevance gate should reject them):")
    for row in result["per_question"]:
        if "result" in row:
            log(f"  {row['result']:15} {row['question']}")

    log(f"\nAverages over {result['answerable']} answerable questions (higher is better, 1.0 is perfect):")
    log(f"  {'Stage':28} {'MRR':>6} {'nDCG@K':>8} {'Recall@K':>9} {'Precision@K':>12}")
    for stage, stage_k, average in result["stages"]:
        log(f"  {stage + f' (K={stage_k})':28} {average['MRR']:6.3f} {average['nDCG']:8.3f} "
            f"{average['Recall']:9.3f} {average['Precision']:12.3f}")
    log("\nMRR by category (after re-rank): " +
        ", ".join(f"{category} {value:.2f}" for category, value in result["by_category"].items()))
    oos = result["out_of_scope"]
    log(f"Relevance gate: {oos['rejected']}/{oos['total']} out-of-scope questions rejected, "
        f"{result['false_rejections']} answerable questions wrongly rejected")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    "How do I install and configure the Windows KSP crypto provider?",
    "Show an artifact configuration example for signing an MSI inside a ZIP file.",
    "How do I submit a signing request from GitHub Actions?",
    "What crypto providers does SignPath support?",
    "How do I use Submit-SigningRequest with a wait for completion?",
    "How do I sign with GPG using SignPath?",
    "How do I set up user synchronization from Microsoft Entra ID?",
]


# Evaluation tab: colour thresholds for the metric cards (green >= first value, orange >= second, else red).
# Precision is naturally low here (most questions have one relevant section, so most of the top K is
# neighbouring material), so its thresholds are lower.
CARD_THRESHOLDS = {"MRR": (0.9, 0.75), "nDCG": (0.85, 0.7), "Recall": (0.9, 0.75), "Precision": (0.4, 0.25),
                   "Gate": (1.0, 0.8)}
CARD_LABELS = {"MRR": "Mean Reciprocal Rank (MRR)", "nDCG": "Normalized DCG (nDCG@{k})",
               "Recall": "Recall@{k}", "Precision": "Precision@{k}"}


def metric_card(label, value, thresholds, text=None):
    """One coloured metric box, in the style of week5/evaluator.py."""
    green, amber = thresholds
    color = "green" if value >= green else "orange" if value >= amber else "red"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{text or f'{value:.4f}'}</div>
    </div>"""


def evaluation_report(result):
    """Turn evaluate_retrieval's result into the Evaluation tab's cards (HTML), chart data and details (markdown)."""
    import pandas as pd

    k = result["k"]
    after_rerank = result["stages"][1][2]          # the cards show the ranking after re-ranking
    cards = "".join(metric_card(CARD_LABELS[m].format(k=k), after_rerank[m], CARD_THRESHOLDS[m]) for m in METRICS)
    oos, answerable = result["out_of_scope"], result["answerable"]
    gate_score = (oos["rejected"] + answerable - result["false_rejections"]) / (oos["total"] + answerable)
    cards += metric_card("Relevance gate (refuses out-of-scope, answers in-scope)", gate_score, CARD_THRESHOLDS["Gate"],
                         text=f"{oos['rejected']}/{oos['total']} rejected · {result['false_rejections']} wrongly")
    cards += ("<div style='margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; "
              "text-align: center; border: 1px solid #c3e6cb;'><span style='font-size: 14px; color: #155724; "
              f"font-weight: bold;'>✓ Evaluation complete: {answerable} answerable + {oos['total']} out-of-scope questions"
              "</span></div>")

    chart = pd.DataFrame([{"Category": category, "Average MRR": value}
                          for category, value in result["by_category"].items()])

    lines = ["### Ranking quality at each stage", "",
             "| Stage | MRR | nDCG@K | Recall@K | Precision@K |", "| --- | --- | --- | --- | --- |"]
    for stage, stage_k, average in result["stages"]:
        lines.append(f"| {stage} (K={stage_k}) | {average['MRR']:.3f} | {average['nDCG']:.3f} | "
                     f"{average['Recall']:.3f} | {average['Precision']:.3f} |")
    lines += ["", "_Vector + keyword → After re-rank shows what the cross-encoder adds; "
              "\"Sent to the LLM\" is what the model actually reads._", "",
              "### Questions where the right section was not ranked first", "",
              "| MRR | Recall@K | Category | Question |", "| --- | --- | --- | --- |"]
    misses = [row for row in result["per_question"] if "MRR" in row and row["MRR"] < 1]
    for row in sorted(misses, key=lambda r: r["MRR"]):
        lines.append(f"| {row['MRR']:.2f} | {row['Recall']:.2f} | {row['category']} | {row['question']} |")
    if not misses:
        lines.append("| – | – | – | All answerable questions had a relevant section in first place |")
    lines += ["", "### Out-of-scope questions", "", "| Result | Question |", "| --- | --- |"]
    lines += [f"| {row['result']} | {row['question']} |" for row in result["per_question"] if "result" in row]
    if result["bad_labels"]:
        lines += ["", "⚠️ Labels that match nothing in the index (check the test file): " +
                  "; ".join(" / ".join(labels) for labels in result["bad_labels"])]
    return cards, chart, "\n".join(lines)


def index_status():
    info = read_index_info()
    if not info:
        return ""
    status = (f"📚 {info.get('pages', '?')} pages · {info.get('chunks', '?')} chunks · "
              f"docs last checked {info.get('last_checked', '?')} · index last changed {info.get('last_changed', '?')}")
    if REFRESH_HOURS > 0:
        status += f" · auto-refresh every {REFRESH_HOURS:g}h"
    return status


def start_auto_refresh():
    """Background thread that re-crawls the docs every REFRESH_HOURS; the retriever picks up changes on its next search."""
    def loop():
        while True:
            # Wait until REFRESH_HOURS after the last check, so a restarted app catches up right away when overdue
            last_checked = read_index_info().get("last_checked_ts", 0)
            time.sleep(max(60, last_checked + REFRESH_HOURS * 3600 - time.time()))
            try:
                refresh(OpenAI())
            except Exception as error:
                log(f"! Auto-refresh failed, keeping the current index: {error}")

    threading.Thread(target=loop, name="signpath-refresh", daemon=True).start()
    log(f"Auto-refresh every {REFRESH_HOURS:g} hours")


def launch_ui(assistant):
    import gradio as gr

    if REFRESH_HOURS > 0:
        start_auto_refresh()

    def respond(message, history):
        if not message.strip():
            yield "", history, gr.skip(), gr.skip()
            return
        # The LLM only needs the text turns; the displayed history keeps earlier answers' images
        previous = [{"role": m["role"], "content": message_text(m)} for m in history if message_text(m).strip()]
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "_Searching the documentation..._"}]
        answer_index = len(history) - 1
        yield "", history, "", ""
        try:
            images = []
            for answer, sources, images, context in assistant.answer(message, previous):
                history[answer_index] = {"role": "assistant", "content": answer}
                yield "", history, sources or gr.skip(), context
            if images:  # documentation images shown under the answer they belong to
                history.append({"role": "assistant", "content": "**Images from the documentation:**"})
                for path, caption in images:
                    history.append({"role": "assistant", "content": f"_{caption}_"})
                    history.append({"role": "assistant", "content": {"path": path, "alt_text": caption}})
                yield "", history, gr.skip(), gr.skip()
        except Exception as error:
            history[answer_index] = {"role": "assistant", "content": f"⚠️ Error: {error}"}
            yield "", history, "", ""

    def run_evaluation(progress=gr.Progress()):
        """Evaluation tab button: score retrieval on the test questions against the current Chroma index."""
        result = evaluate_retrieval(assistant, EVAL_K, progress=lambda fraction, text: progress(fraction, desc=text))
        return evaluation_report(result)

    with gr.Blocks(title="SignPath Docs Assistant") as ui:
        sources_md = " and ".join(f"[{s['name']}]({s['start']})" for s in SOURCES)
        gr.Markdown("# SignPath Documentation Assistant\nAnswers come only from the indexed "
                    f"{sources_md}. If they don't cover a question, it says so.")
        gr.Markdown(index_status)  # re-evaluated on every page load, so it shows the latest refresh
        with gr.Tab("💬 Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Chat", height=620)
                    question = gr.Textbox(placeholder="Ask about setup, configuration, crypto providers, build integrations...",
                                          show_label=False, lines=2)
                    with gr.Row():
                        send = gr.Button("Ask", variant="primary")
                        clear = gr.Button("Clear")
                    gr.Examples(EXAMPLES, inputs=question)
                with gr.Column(scale=2):
                    sources = gr.Markdown("### Sources\n_Sources for each answer appear here._")
                    with gr.Accordion("Retrieval details (ranking)", open=False):
                        context = gr.Markdown()

        with gr.Tab("📊 Evaluation"):
            gr.Markdown("## 🔍 Retrieval evaluation\n"
                        f"Runs the {RETRIEVAL_TESTS_FILE.name} test questions through the search over the vector "
                        "database and checks whether the sections that answer them are found and ranked high. "
                        "Takes about 3 minutes; one small embedding call per question, no LLM calls.")
            evaluate_button = gr.Button("Run Evaluation", variant="primary", size="lg")
            with gr.Row():
                with gr.Column(scale=1):
                    evaluation_cards = gr.HTML("<div style='padding: 20px; text-align: center; color: #999;'>"
                                               "Click 'Run Evaluation' to start</div>")
                with gr.Column(scale=1):
                    evaluation_chart = gr.BarPlot(x="Category", y="Average MRR", title="Average MRR by Category",
                                                  y_lim=[0, 1], height=400, x_label_angle=-35)
            evaluation_details = gr.Markdown()

        outputs = [question, chatbot, sources, context]
        question.submit(respond, [question, chatbot], outputs)
        send.click(respond, [question, chatbot], outputs)
        clear.click(lambda: ("", [], "### Sources", ""), None, outputs)
        evaluate_button.click(run_evaluation, None, [evaluation_cards, evaluation_chart, evaluation_details])

    ui.launch(inbrowser=OPEN_BROWSER, allowed_paths=[str(DATA_DIR)])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SignPath documentation RAG assistant")
    parser.add_argument("--crawl", action="store_true", help="re-crawl the docs and rebuild the whole index, then launch")
    parser.add_argument("--reindex", action="store_true", help="rebuild the whole index from the saved crawl, then launch")
    parser.add_argument("--refresh", action="store_true",
                        help="re-crawl and update only what changed, then exit (for a scheduled task / cron)")
    parser.add_argument("--eval", action="store_true", help="run the evaluation questions and exit (non-zero if any fail)")
    parser.add_argument("--eval-retrieval", action="store_true",
                        help="measure search quality (MRR, nDCG@K, Recall@K, Precision@K) on signpath_retrieval_tests.json")
    parser.add_argument("--k", type=int, default=EVAL_K, help="K for --eval-retrieval (default 5)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="maximum pages to crawl")
    parser.add_argument("--ask", help="answer one question in the terminal instead of launching the UI")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set (add it to your .env file or pass it to the container)")
    client = OpenAI()

    if args.crawl or not PAGES_FILE.exists():
        log(f"Crawling {', '.join(s['start'] for s in SOURCES)} ...")
        pages = crawl(args.max_pages)
        save_pages(pages)
        sync_index(client, pages, rebuild=True)
    elif args.refresh:
        if refresh(client, args.max_pages) is None and not args.eval:
            sys.exit(1)
    elif args.reindex or not read_index_info() or read_index_info().get("embed_model") != EMBED_MODEL:
        sync_index(client, load_pages(), rebuild=args.reindex)

    # --refresh on its own just updates the data; combined with --eval/--eval-retrieval/--ask it continues
    if args.refresh and not (args.eval or args.eval_retrieval or args.ask):
        return
    assistant = SignPathAssistant()
    if args.eval_retrieval:
        run_retrieval_eval(assistant, args.k)
        if not args.eval:
            return
    if args.eval:
        sys.exit(0 if run_eval(assistant) else 1)
    if args.ask:
        *_, (answer, sources, images, context) = assistant.answer(args.ask)  # last streamed update is the full answer
        image_lines = "\n".join(f"- {caption}: {path}" for path, caption in images)
        print(f"\n{answer}\n\n{sources}\n\n" + (f"### Images\n{image_lines}\n\n" if images else "") + context)
    else:
        launch_ui(assistant)


if __name__ == "__main__":
    main()
