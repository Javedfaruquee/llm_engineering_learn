"""Offline tests: no network, no OpenAI key. Run from the SignPath_RAG folder:  python -m pytest -q"""

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OPENAI_API_KEY", "sk-test-not-used")

import Signpath_RAG as rag  # noqa: E402


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point every data path of the module at an empty temporary folder."""
    paths = {
        "DATA_DIR": tmp_path, "PAGES_FILE": tmp_path / "pages.json", "IMAGES_DIR": tmp_path / "images",
        "CHROMA_DIR": tmp_path / "chroma", "INDEX_INFO_FILE": tmp_path / "index_info.json",
        "PREVIOUS_PAGES_FILE": tmp_path / "pages.previous.json",
        "PREVIOUS_INFO_FILE": tmp_path / "index_info.previous.json",
        "LOG_DIR": tmp_path / "logs", "EVAL_DIR": tmp_path / "eval",
        "REFRESH_LOCK_FILE": tmp_path / "refresh.lock",
    }
    for name, value in paths.items():
        monkeypatch.setattr(rag, name, value)
    (tmp_path / "images").mkdir()
    return tmp_path


class FakeEmbeddings:
    """Deterministic 8-dimensional 'embeddings' from a text hash; counts the texts it embedded."""

    def __init__(self):
        self.embedded = 0

    def create(self, model, input):
        self.embedded += len(input)
        data = []
        for text in input:
            digest = hashlib.sha256(text.encode()).digest()
            data.append(SimpleNamespace(embedding=[b / 255 + 0.01 for b in digest[:8]]))
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=sum(len(t.split()) for t in input)))


@pytest.fixture
def fake_client():
    return SimpleNamespace(embeddings=FakeEmbeddings())


def make_page(url, sections):
    """A crawled page in the pages.json format; sections are (heading, text) pairs."""
    page = {"url": url, "title": url.rsplit("/", 1)[-1] or "Home", "breadcrumb": "Docs", "source": "SignPath Documentation",
            "lastmod": "2026-10-01",
            "sections": [{"path": ["Docs", heading], "anchor": rag.slugify(heading), "text": text, "images": []}
                         for heading, text in sections]}
    page["hash"] = hashlib.sha1(repr(page["sections"]).encode()).hexdigest()
    return page


@pytest.fixture
def pages():
    return [
        make_page("https://docs.signpath.io/ksp", [("Installation", "Run msiexec /i SignPathKSP.msi to install the KSP."),
                                                   ("Verify", "Run certutil -csplist and look for SignPathKSP.")]),
        make_page("https://docs.signpath.io/gpg", [("GPG", "Use gpg --sign -u $GPG_KEY_ID to sign files.")]),
    ]


@pytest.fixture
def no_reranker(monkeypatch):
    monkeypatch.setattr(rag, "load_reranker", lambda: None)


