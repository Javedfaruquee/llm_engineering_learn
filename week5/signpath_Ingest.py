"""
SignPath RAG - step 1: ingest.

Crawls the SignPath documentation (https://docs.signpath.io/) and the SignPath knowledge base
(https://signpath.io/knowledge-base), extracts the text, code examples and images of every page,
splits them into chunks, embeds the chunks with OpenAI and stores them in a Chroma vector database.

Run it whenever you want to pick up changes in the docs (for example once a day from a scheduled task).
Only new or changed chunks are embedded again, so a run with no changes costs nothing.

Usage (from the project root, with the .venv):
    .venv/Scripts/python.exe week5/signpath_Ingest.py              # crawl and update the database
    .venv/Scripts/python.exe week5/signpath_Ingest.py --rebuild    # crawl and re-embed everything

Then start the chat UI with week5/signpath_answer.py.
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import chromadb
import requests
import tiktoken
from bs4 import BeautifulSoup, Comment, NavigableString, Tag
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

# ---------------------------------------------------------------------------
# Settings (DATA_DIR, COLLECTION and EMBED_MODEL must match signpath_answer.py)
# ---------------------------------------------------------------------------

SOURCES = [
    {"name": "SignPath Documentation", "start": "https://docs.signpath.io/",
     "host": "docs.signpath.io", "prefix": "/"},
    {"name": "SignPath Knowledge Base", "start": "https://signpath.io/knowledge-base",
     "host": "signpath.io", "prefix": "/knowledge-base"},
]
IMAGE_HOSTS = ["docs.signpath.io", "signpath.io", "framerusercontent.com"]
USER_AGENT = "SignPath-RAG-learning-crawler/1.0"
MAX_PAGES = 2000
CRAWL_DELAY = 0.3          # seconds between requests, to be polite to the web servers

DATA_DIR = Path(os.getenv("SIGNPATH_DATA_DIR", Path(__file__).parent / "signpath_data"))
PAGES_FILE = DATA_DIR / "pages.json"
IMAGES_DIR = DATA_DIR / "images"
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_INFO_FILE = DATA_DIR / "index_info.json"
COLLECTION = "signpath_docs"

EMBED_MODEL = "text-embedding-3-large"
CHUNK_CHARS = 1800         # target chunk size; code blocks are only split when they are huge
MIN_PAGE_RATIO = 0.8       # a crawl finding fewer than 80% of the previous pages is treated as failed

SKIP_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf", ".zip", ".msi", ".exe",
                   ".xml", ".json", ".css", ".js", ".txt", ".rss", ".atom", ".mp4", ".woff", ".woff2", ".ttf"]
SECTION_HEADINGS = ["h1", "h2", "h3", "h4"]
SVG_FALLBACK_FONTS = "Carlito, DejaVu Sans, Arial, Segoe UI"
CODE_FENCE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------

def find_source(url):
    """Return the SOURCES entry the URL belongs to, or None."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    for source in SOURCES:
        if parsed.netloc != source["host"]:
            continue
        prefix = source["prefix"]
        if prefix == "/" or path == prefix or path.startswith(prefix + "/"):
            return source
    return None


def normalize_url(url):
    """Return a canonical page URL, or None if the URL is not a page we want to crawl."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if find_source(url) is None:
        return None
    path = parsed.path or "/"
    path = re.sub(r"/index(\.html?)?$", "/", path)
    if path != "/":
        path = path.rstrip("/")
    if Path(path).suffix.lower() in SKIP_EXTENSIONS:
        return None
    if "/feeds/" in path:          # feeds repeat the changelog as raw XML
        return None
    return "https://" + parsed.netloc + path


# ---------------------------------------------------------------------------
# HTML to markdown
# ---------------------------------------------------------------------------

def code_language(pre):
    """Language of a code block, from a 'language-xxx' class on the <pre> or its parents."""
    elements = [pre] + list(pre.parents)[:3]
    for element in elements:
        for css_class in element.get("class") or []:
            if css_class.startswith("language-"):
                language = css_class[len("language-"):]
                if language == "plaintext":
                    return ""
                return language
    return ""


def children_to_markdown(node, page_url, images):
    text = ""
    for child in node.children:
        text += to_markdown(child, page_url, images)
    return text


def list_to_markdown(node, page_url, images):
    lines = []
    number = 1
    for item in node.find_all("li", recursive=False):
        content = children_to_markdown(item, page_url, images).strip()
        content = re.sub(r"\n{3,}", "\n\n", content)
        if node.name == "ol":
            bullet = str(number) + ". "
        else:
            bullet = "- "
        content = content.replace("\n", "\n" + " " * len(bullet))
        lines.append(bullet + content)
        number += 1
    return "\n\n" + "\n".join(lines) + "\n\n"


def table_to_markdown(node, page_url, images):
    rows = []
    for tr in node.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"], recursive=False):
            cell_text = children_to_markdown(cell, page_url, images)
            cell_text = re.sub(r"\s+", " ", cell_text).strip().replace("|", "\\|")
            cells.append(cell_text)
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    for row in rows:
        while len(row) < width:
            row.append("")
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n\n" + "\n".join(lines) + "\n\n"


def to_markdown(node, page_url, images):
    """Convert an HTML node to markdown. Images found are appended to the `images` list."""
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
        return "\n\n```" + code_language(node) + "\n" + code + "\n```\n\n"
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return "\n\n" + "#" * int(name[1]) + " " + node.get_text(" ", strip=True) + "\n\n"
    if name == "img":
        src = node.get("src")
        if not src:
            return ""
        alt = (node.get("alt") or node.get("title") or "").strip()
        images.append({"src": urljoin(page_url, src), "alt": alt})
        return " [Image: " + (alt or "screenshot") + "] "
    if name == "br":
        return "\n"
    if name == "code":
        return "`" + node.get_text() + "`"
    if name == "table":
        return table_to_markdown(node, page_url, images)
    if name in ("ul", "ol"):
        return list_to_markdown(node, page_url, images)
    if name == "blockquote" or (name == "div" and "panel" in classes):
        inner = children_to_markdown(node, page_url, images).strip()
        for label in ("info", "tip", "warning", "note"):
            if label in classes:
                inner = "**" + label.capitalize() + ":** " + inner
                break
        quoted = ["> " + line for line in inner.splitlines()]
        return "\n\n" + "\n".join(quoted) + "\n\n"
    if name in ("p", "div", "section", "dl", "dd", "dt", "figure", "figcaption"):
        return "\n\n" + children_to_markdown(node, page_url, images).strip() + "\n\n"
    return children_to_markdown(node, page_url, images)


def clean_markdown(text):
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Splitting a page into sections (one per heading)
# ---------------------------------------------------------------------------

def content_blocks(node, blocks):
    """Collect the content blocks in document order. Wrappers that contain headings are opened up,
    because the knowledge base (built with Framer) nests its headings deep inside <div>s."""
    for child in node.children:
        if isinstance(child, Tag) and child.name not in SECTION_HEADINGS and child.find(SECTION_HEADINGS):
            content_blocks(child, blocks)
        else:
            blocks.append(child)
    return blocks


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def finish_section(current, sections, seen):
    """Add the section being built to `sections`, unless it is empty or a repeat."""
    text = clean_markdown("".join(current["parts"]))
    unique_images = {}
    for image in current["images"]:
        unique_images[image["src"]] = image
    images = list(unique_images.values())
    key = (tuple(current["path"]), text)
    if (text or images) and key not in seen:
        seen.add(key)
        sections.append({"path": current["path"], "anchor": current["anchor"], "text": text, "images": images})


def split_into_sections(root, page_url, breadcrumb, slug_anchors):
    """Split content into sections at h1-h4 headings. Each section has a heading path, an anchor,
    markdown text and its images. If slug_anchors is True, anchors are made from the heading text."""
    sections = []
    seen = set()
    headings = {}
    current = {"path": [breadcrumb], "anchor": "", "parts": [], "images": []}

    for block in content_blocks(root, []):
        if isinstance(block, Tag) and block.name in SECTION_HEADINGS:
            finish_section(current, sections, seen)
            level = int(block.name[1])
            for old_level in list(headings.keys()):
                if old_level >= level:
                    del headings[old_level]
            headings[level] = block.get_text(" ", strip=True)
            path = [breadcrumb]
            for heading_level in sorted(headings):
                path.append(headings[heading_level])
            if slug_anchors:
                anchor = slugify(block.get_text(" ", strip=True))
            else:
                anchor = block.get("id", "")
            current = {"path": path, "anchor": anchor, "parts": [], "images": []}
        else:
            images = []
            current["parts"].append(to_markdown(block, page_url, images))
            current["images"].extend(images)
    finish_section(current, sections, seen)
    return sections


def extract_docs_page(soup, url):
    """docs.signpath.io: the content is an <article> inside <main>. Returns None for other pages."""
    main = soup.find("main")
    if main is None:
        return None
    article = main.find("article")
    if article is None:
        return None
    if soup.title:
        title = soup.title.get_text(strip=True).split(" | ")[0]
    else:
        title = url
    # The page header holds a breadcrumb like "Crypto Providers ❯ Windows KSP"; some pages (the FAQ)
    # have an empty header, so fall back to the page title
    header = soup.select_one("main h1")
    breadcrumb = ""
    if header:
        breadcrumb = header.get_text(" ", strip=True).replace("❯", ">")
    if not breadcrumb:
        breadcrumb = title
    breadcrumb = re.sub(r"\s+", " ", breadcrumb)
    sections = split_into_sections(article, url, breadcrumb, slug_anchors=False)
    return {"url": url, "title": title, "breadcrumb": breadcrumb, "sections": sections}


def extract_knowledge_base_page(soup, url):
    """signpath.io/knowledge-base (built with Framer). Parts of each page are repeated for different
    screen sizes, and other pages' titles can appear as previews. We pick the <h1> that best matches
    the page <title>, use the smallest block holding it and the section headings, and drop the
    navigation that comes before the title."""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True).split(" - ")[-1]

    best = None
    for h1 in soup.find_all("h1"):
        root = None
        for parent in h1.parents:
            if parent.find("h2"):
                root = parent
                break
        if root is None:
            continue
        match = difflib.SequenceMatcher(None, h1.get_text(" ", strip=True).lower(), title.lower()).ratio()
        score = (round(match, 1), -len(root.get_text(" ", strip=True)))
        if best is None or score > best["score"]:
            best = {"score": score, "h1": h1, "root": root}
    if best is None:
        return None
    h1 = best["h1"]
    root = best["root"]

    # Remove everything before the title inside the block: the page index and "On this page" links
    node = h1
    while node is not None and node is not root:
        for sibling in list(node.previous_siblings):
            sibling.extract()
        node = node.parent

    page_title = h1.get_text(" ", strip=True)
    h1.extract()
    if not title:
        title = page_title
    breadcrumb = "Knowledge Base > " + page_title
    sections = split_into_sections(root, url, breadcrumb, slug_anchors=True)
    return {"url": url, "title": "Knowledge Base: " + title, "breadcrumb": breadcrumb, "sections": sections}


def extract_page(soup, url):
    source = find_source(url)
    if source["host"] == "signpath.io":
        page = extract_knowledge_base_page(soup, url)
    else:
        page = extract_docs_page(soup, url)
    if page:
        page["source"] = source["name"]
    return page


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def svg_to_png(svg_bytes):
    """Render an SVG diagram to PNG on white. Gradio can't show SVGs in the chat, and the docs'
    diagrams are transparent with black text, which is unreadable in dark mode."""
    try:
        import resvg_py
        svg = svg_bytes.decode("utf-8")
        # The diagrams use Calibri; add fallback fonts that exist on Linux so the text isn't dropped
        svg = re.sub(r'font-family="([^"]*)"', r'font-family="\1, ' + SVG_FALLBACK_FONTS + '"', svg)
        svg = re.sub(r"font-family:\s*([^;\"']*)", r"font-family: \1, " + SVG_FALLBACK_FONTS, svg)
        png = resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", zoom=1.5)
        return bytes(png), ".png"
    except Exception as error:
        log(f"  ! SVG to PNG conversion failed ({error}); keeping the SVG")
        return svg_bytes, ".svg"


def flatten_on_white(data, ext):
    """Put images with transparent areas on a white background (readable in dark mode)."""
    try:
        image = Image.open(BytesIO(data))
        has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
        if not has_alpha:
            return data, ext
        rgba = image.convert("RGBA")
        lowest_alpha = rgba.getchannel("A").getextrema()[0]
        if lowest_alpha == 255:
            return data, ext          # has an alpha channel, but nothing is transparent
        canvas = Image.new("RGB", rgba.size, "white")
        canvas.paste(rgba, mask=rgba.getchannel("A"))
        output = BytesIO()
        canvas.save(output, format="PNG")
        return output.getvalue(), ".png"
    except Exception as error:
        log(f"  ! could not flatten image ({error}); keeping it as is")
        return data, ext


def download_image(session, src, downloaded):
    """Save an image once and return its path relative to DATA_DIR (or None)."""
    if src in downloaded:
        return downloaded[src]
    downloaded[src] = None
    parsed = urlparse(src)
    if parsed.netloc not in IMAGE_HOSTS:
        return None               # external badges are not documentation content

    fetch_url = src
    if parsed.netloc == "framerusercontent.com":
        fetch_url = src.split("?")[0]     # drop Framer's resize parameters to get the full-size image
    try:
        response = session.get(fetch_url, timeout=20)
    except requests.RequestException as error:
        log(f"  ! image failed {src}: {error}")
        return None
    content_type = response.headers.get("content-type", "")
    if response.status_code != 200 or not content_type.startswith("image/"):
        return None

    ext = Path(parsed.path).suffix.lower()
    if ext not in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"):
        ext = "." + content_type.split("/")[1].split("+")[0].split(";")[0]
    data = response.content
    if ext == ".svg":
        data, ext = svg_to_png(data)
    else:
        data, ext = flatten_on_white(data, ext)

    name = hashlib.sha1(src.encode()).hexdigest()[:12] + "_" + Path(parsed.path).stem[:60] + ext
    (IMAGES_DIR / name).write_bytes(data)
    downloaded[src] = "images/" + name
    return downloaded[src]


def delete_unused_images(pages):
    used = set()
    for page in pages:
        for section in page["sections"]:
            for image in section["images"]:
                if image.get("path"):
                    used.add(image["path"])
    for file in IMAGES_DIR.glob("*"):
        if "images/" + file.name not in used:
            file.unlink()


# ---------------------------------------------------------------------------
# Crawling
# ---------------------------------------------------------------------------

def read_sitemap(session, host):
    """Return {url: last modified date} for the sitemap pages that belong to one of our sources."""
    try:
        xml = session.get("https://" + host + "/sitemap.xml", timeout=20).text
    except requests.RequestException:
        return {}
    entries = {}
    for block in re.findall(r"<url>(.*?)</url>", xml, re.DOTALL):
        loc = re.search(r"<loc>\s*([^<\s]+)\s*</loc>", block)
        lastmod = re.search(r"<lastmod>\s*([^<\s]+)\s*</lastmod>", block)
        if loc is None:
            continue
        url = normalize_url(loc.group(1))
        if url:
            if lastmod:
                entries[url] = lastmod.group(1)[:10]
            else:
                entries[url] = ""
    return entries


def crawl(max_pages):
    """Visit every page of every source, starting from the start URLs and sitemaps and following links."""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    robots = {}
    lastmods = {}
    for source in SOURCES:
        host = source["host"]
        if host in robots:
            continue
        parser = RobotFileParser("https://" + host + "/robots.txt")
        try:
            parser.read()
        except Exception:
            parser = None
        robots[host] = parser
        lastmods.update(read_sitemap(session, host))

    queue = []
    seen = set()
    start_urls = [normalize_url(source["start"]) for source in SOURCES] + list(lastmods.keys())
    for url in start_urls:
        if url and url not in seen:
            seen.add(url)
            queue.append(url)

    pages = []
    downloaded = {}
    fingerprints = {}
    while queue and len(pages) < max_pages:
        url = queue.pop(0)
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

        final_url = normalize_url(response.url)
        if response.status_code != 200 or final_url is None:
            continue
        if "text/html" not in response.headers.get("content-type", ""):
            continue
        if final_url != url:
            if final_url in seen:
                continue          # redirected to a page that is already crawled or queued
            seen.add(final_url)

        # signpath.io doesn't say which character set its pages use, so requests would assume Latin-1 and
        # turn characters like ’ and – into garbage ("â€™"). Both sites actually send UTF-8.
        if "charset" not in response.headers.get("content-type", ""):
            response.encoding = "utf-8"
        soup = BeautifulSoup(response.text, "lxml")

        # Queue the links on this page
        links = []
        for link in soup.find_all("a", href=True):
            links.append(link["href"])
        redirect = soup.find("meta", attrs={"http-equiv": re.compile("refresh", re.I)})
        if redirect and "url=" in redirect.get("content", "").lower():
            links.append(redirect["content"].split("=", 1)[1].strip("'\" "))
        for href in links:
            target = normalize_url(urljoin(final_url, href))
            if target and target not in seen:
                seen.add(target)
                queue.append(target)
        if redirect:
            continue              # an HTML redirect page has no content of its own

        page = extract_page(soup, final_url)
        if page is None:
            continue
        has_text = False
        for section in page["sections"]:
            if section["text"]:
                has_text = True
        if not has_text:
            continue

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

    image_count = 0
    for path in downloaded.values():
        if path:
            image_count += 1
    log(f"Crawled {len(pages)} pages and saved {image_count} images")
    return pages


def load_pages():
    if PAGES_FILE.exists():
        return json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    return []


def save_pages(pages):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PAGES_FILE.write_text(json.dumps(pages, indent=1, ensure_ascii=False), encoding="utf-8")


def report_page_changes(old_pages, new_pages):
    old = {}
    for page in old_pages:
        old[page["url"]] = page.get("hash")
    new = {}
    for page in new_pages:
        new[page["url"]] = page.get("hash")

    added = [url for url in new if url not in old]
    removed = [url for url in old if url not in new]
    changed = [url for url in new if url in old and old[url] != new[url]]
    log(f"Pages: {len(added)} added, {len(changed)} changed, {len(removed)} removed")
    for url in added:
        log("  added: " + url)
    for url in changed:
        log("  changed: " + url)
    for url in removed:
        log("  removed: " + url)
    return {"added": len(added), "changed": len(changed), "removed": len(removed)}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def split_blocks(text):
    """Split markdown into blocks at blank lines, but never inside a fenced code block."""
    blocks = []
    current = []
    in_code = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if not line.strip() and not in_code:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def split_large_block(block, limit):
    """Split a block that is far too large (e.g. a long XML reference) into smaller pieces."""
    match = CODE_FENCE.fullmatch(block.strip())
    if match:
        language = match.group(1)
        body = match.group(2)
    else:
        language = ""
        body = block

    pieces = []
    current = []
    current_size = 0
    for line in body.splitlines():
        if current and current_size + len(line) > limit:
            pieces.append(current)
            current = []
            current_size = 0
        current.append(line)
        current_size += len(line) + 1
    if current:
        pieces.append(current)

    result = []
    for piece in pieces:
        if match:
            result.append("```" + language + "\n" + "\n".join(piece) + "\n```")
        else:
            result.append("\n".join(piece))
    return result


def split_text(text, limit=CHUNK_CHARS):
    """Split a section's text into chunks of about `limit` characters."""
    blocks = []
    for block in split_blocks(text):
        if len(block) > limit * 2.5:
            blocks.extend(split_large_block(block, limit))
        else:
            blocks.append(block)

    chunks = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > limit:
            chunks.append(current)
            current = block
        elif current:
            current = current + "\n\n" + block
        else:
            current = block
    if current:
        chunks.append(current)
    return chunks


def add_chunk(chunks, kind, text, metadata):
    """The chunk id is a hash of its content, so an unchanged chunk keeps its id (and embedding)."""
    chunk_id = hashlib.sha1((kind + "\n" + text).encode()).hexdigest()[:24]
    if chunk_id not in chunks:
        chunks[chunk_id] = {"id": chunk_id, "text": text, "meta": metadata}


def make_metadata(base, kind, language, images):
    metadata = dict(base)
    metadata["type"] = kind
    metadata["lang"] = language
    metadata["images"] = json.dumps(images)      # Chroma metadata values must be strings or numbers
    return metadata


def build_chunks(pages):
    """Three kinds of chunks per section: text, each code example, and each image."""
    chunks = {}
    for page_number, page in enumerate(pages):
        for section_number, section in enumerate(page["sections"]):
            url = page["url"]
            if section["anchor"]:
                url = url + "#" + section["anchor"]
            section_name = " > ".join(section["path"])
            source = page.get("source", SOURCES[0]["name"])
            header = f"Source: {source}\nPage: {page['title']}\nSection: {section_name}\nURL: {url}\n\n"

            section_images = []
            for image in section["images"]:
                if image.get("path"):
                    section_images.append({"path": image["path"], "alt": image["alt"]})

            # Stored with every chunk; "position" is the document order, used to sort images in answers
            base = {"url": url, "page": page["title"], "section": section_name,
                    "lastmod": page.get("lastmod", ""), "position": page_number * 1000 + section_number}

            # Text chunks: the section's prose, code and tables together
            for piece in split_text(section["text"]):
                add_chunk(chunks, "text", header + piece, make_metadata(base, "text", "", section_images))

            # Code chunks: each example with the sentence before it, so "show me an example" finds it
            for match in CODE_FENCE.finditer(section["text"]):
                language = match.group(1)
                code = match.group(2)
                if len(code) < 40:
                    continue
                before = CODE_FENCE.sub("", section["text"][:match.start()]).strip()
                intro = ""
                if before:
                    intro = before.split("\n\n")[-1][-600:]
                body = f"Example ({language or 'code'}):\n{intro}\n\n```{language}\n{code[:CHUNK_CHARS * 3]}\n```"
                add_chunk(chunks, "code", header + body, make_metadata(base, "code", language, section_images))

            # Image chunks: the image caption plus the start of the section's text
            prose = CODE_FENCE.sub("", section["text"]).strip()[:700]
            for image in section_images:
                body = f"Image: {image['alt'] or 'screenshot'}\nShown in section: {section_name}\n\n{prose}"
                add_chunk(chunks, "image", header + body, make_metadata(base, "image", "", [image]))
    return list(chunks.values())


# ---------------------------------------------------------------------------
# Embedding and storing in Chroma
# ---------------------------------------------------------------------------

def embed(client, texts):
    encoding = tiktoken.get_encoding("cl100k_base")
    shortened = []
    for text in texts:
        shortened.append(encoding.decode(encoding.encode(text)[:8000]))   # the model's input limit
    vectors = []
    for start in range(0, len(shortened), 100):
        response = client.embeddings.create(model=EMBED_MODEL, input=shortened[start:start + 100])
        for item in response.data:
            vectors.append(item.embedding)
    return vectors


def read_index_info():
    try:
        return json.loads(INDEX_INFO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def update_vector_db(client, pages, rebuild, page_changes):
    """Make the Chroma collection match the pages: embed new chunks and delete chunks that are gone."""
    chunks = build_chunks(pages)
    info = read_index_info()

    db = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    collection_names = [c.name for c in db.list_collections()]
    if COLLECTION in collection_names and (rebuild or info.get("embed_model") != EMBED_MODEL):
        db.delete_collection(COLLECTION)
        log("Deleted the old collection; everything will be embedded again")
    collection = db.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    existing_ids = set(collection.get(include=[])["ids"])
    new_ids = set()
    for chunk in chunks:
        new_ids.add(chunk["id"])
    new_chunks = [chunk for chunk in chunks if chunk["id"] not in existing_ids]
    unchanged_chunks = [chunk for chunk in chunks if chunk["id"] in existing_ids]
    stale_ids = [chunk_id for chunk_id in existing_ids if chunk_id not in new_ids]

    if stale_ids:
        collection.delete(ids=stale_ids)
    if new_chunks:
        log(f"Embedding {len(new_chunks)} new or changed chunks...")
        vectors = embed(client, [chunk["text"] for chunk in new_chunks])
        for start in range(0, len(new_chunks), 500):
            batch = new_chunks[start:start + 500]
            collection.add(ids=[chunk["id"] for chunk in batch],
                           documents=[chunk["text"] for chunk in batch],
                           embeddings=vectors[start:start + 500],
                           metadatas=[chunk["meta"] for chunk in batch])
    # Unchanged text can still have new metadata (page order, last-modified date, image paths)
    for start in range(0, len(unchanged_chunks), 500):
        batch = unchanged_chunks[start:start + 500]
        collection.update(ids=[chunk["id"] for chunk in batch], metadatas=[chunk["meta"] for chunk in batch])

    # index_info.json tells signpath_answer.py what is in the database and when it last changed
    now = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    last_changed = info.get("last_changed", now)
    if new_chunks or stale_ids or not info:
        last_changed = now
    newest_page = ""
    for page in pages:
        if page.get("lastmod", "") > newest_page:
            newest_page = page["lastmod"]
    info = {
        "embed_model": EMBED_MODEL,
        "pages": len(pages),
        "chunks": collection.count(),
        "last_checked": now,
        "last_checked_ts": time.time(),
        "last_changed": last_changed,
        "newest_page_update": newest_page,
        "last_sync": {"added_chunks": len(new_chunks), "removed_chunks": len(stale_ids),
                      "added": page_changes["added"], "changed": page_changes["changed"],
                      "removed": page_changes["removed"]},
    }
    INDEX_INFO_FILE.write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"Vector database ready: {info['chunks']} chunks ({len(new_chunks)} added, {len(stale_ids)} removed)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Crawl the SignPath docs and knowledge base into a Chroma vector database")
    parser.add_argument("--rebuild", action="store_true", help="embed everything again instead of only changes")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="maximum number of pages to crawl")
    args = parser.parse_args()

    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (add it to your .env file)")
    client = OpenAI()

    log("Crawling " + ", ".join(source["start"] for source in SOURCES) + " ...")
    old_pages = load_pages()
    new_pages = crawl(args.max_pages)

    # Safety check: if the sites were down or changed shape, keep the existing data
    if old_pages and len(new_pages) < MIN_PAGE_RATIO * len(old_pages):
        log(f"! Crawl found only {len(new_pages)} pages (previously {len(old_pages)}); keeping the existing data")
        raise SystemExit(1)

    page_changes = report_page_changes(old_pages, new_pages)
    save_pages(new_pages)
    update_vector_db(client, new_pages, args.rebuild, page_changes)
    delete_unused_images(new_pages)


if __name__ == "__main__":
    main()
