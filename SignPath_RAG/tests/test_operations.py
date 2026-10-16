"""Index lifecycle (staging, quality gate, promotion, rollback, repair) and the operational helpers."""

import json

import pytest

import Signpath_RAG as rag
from conftest import make_page

GOOD = {"mrr": 0.95, "recall": 1.0, "ndcg": 0.9, "false_rejections": 0, "bad_labels": 0, "tests": "t"}
BAD = {**GOOD, "mrr": 0.5}


def live_ids():
    db = rag.chroma_client()
    return set(db.get_collection(rag.COLLECTION).get(include=[])["ids"])


def build_live(client, pages):
    rag.save_pages(pages)
    rag.sync_index(client, pages, rebuild=True)


@pytest.fixture
def gate(monkeypatch):
    """Replace the retrieval evaluation in refresh() with a fixed score (the real one needs the re-ranker model)."""
    def use(quality):
        monkeypatch.setattr(rag, "Retriever", lambda *a, **k: None)
        monkeypatch.setattr(rag, "evaluate_retrieval", lambda *a, **k: None)
        monkeypatch.setattr(rag, "quality_summary", lambda result: dict(quality))
    return use


def changed_pages(pages):
    new = json.loads(json.dumps(pages))
    new[1]["sections"][0]["text"] = "Use gpg --detach-sign -u $GPG_KEY_ID to create a detached signature."
    return new


def test_first_build_and_unchanged_refresh_embed_nothing_twice(data_dir, fake_client, pages, monkeypatch):
    build_live(fake_client, pages)
    first = fake_client.embeddings.embedded
    assert first > 0 and rag.read_index_info()["chunks"] == len(live_ids())
    monkeypatch.setattr(rag, "crawl", lambda max_pages=0: pages)
    info = rag.refresh(fake_client)
    assert fake_client.embeddings.embedded == first          # no content change: no embedding calls
    assert info["last_refresh"]["status"] == "no content change"


def test_refresh_passing_the_gate_is_promoted_and_can_be_rolled_back(data_dir, fake_client, pages, gate, monkeypatch):
    build_live(fake_client, pages)
    before = live_ids()
    gate(GOOD)
    new_pages = changed_pages(pages)
    monkeypatch.setattr(rag, "crawl", lambda max_pages=0: new_pages)
    embedded = fake_client.embeddings.embedded
    info = rag.refresh(fake_client)

    assert info["last_refresh"]["status"] == "promoted" and info["quality"]["mrr"] == GOOD["mrr"]
    assert fake_client.embeddings.embedded - embedded == 1    # only the edited section's one text chunk
    after = live_ids()
    assert after != before and len(after & before) > 0        # unchanged chunks were reused
    assert rag.load_pages() == new_pages and rag.load_pages_file(rag.PREVIOUS_PAGES_FILE) == pages
    names = rag.collection_names(rag.chroma_client())
    assert rag.PREVIOUS_COLLECTION in names and rag.STAGING_COLLECTION not in names

    rag.rollback()
    assert live_ids() == before and rag.load_pages() == pages
    assert rag.read_index_info()["last_refresh"]["status"] == "rolled back"
    rag.rollback()                                            # again = roll forward
    assert live_ids() == after


def test_refresh_failing_the_gate_is_held(data_dir, fake_client, pages, gate, monkeypatch):
    build_live(fake_client, pages)
    info = rag.read_index_info()
    info["quality"] = GOOD
    rag.write_index_info(info)
    before = live_ids()
    gate(BAD)
    monkeypatch.setattr(rag, "crawl", lambda max_pages=0: changed_pages(pages))

    assert rag.refresh(fake_client) is None
    assert live_ids() == before and rag.load_pages() == pages
    last = rag.read_index_info()["last_refresh"]
    assert last["status"] == "held" and "MRR" in last["reason"]
    assert rag.STAGING_COLLECTION not in rag.collection_names(rag.chroma_client())

    assert rag.refresh(fake_client, force=True)["last_refresh"]["status"] == "promoted"   # --force overrides


def test_crawl_with_too_few_pages_is_rejected(data_dir, fake_client, pages, monkeypatch):
    build_live(fake_client, pages)
    monkeypatch.setattr(rag, "crawl", lambda max_pages=0: pages[:1])   # 1 of 2 pages < 80 %
    assert rag.refresh(fake_client) is None
    assert rag.read_index_info()["last_refresh"]["status"] == "rejected"
    assert rag.load_pages() == pages


def test_repair_finishes_an_interrupted_promotion(data_dir, fake_client, pages):
    build_live(fake_client, pages)
    db = rag.chroma_client()
    rag.build_staging(fake_client, changed_pages(pages))
    db.get_collection(rag.COLLECTION).modify(name=rag.PREVIOUS_COLLECTION)   # crash between the two renames
    rag.repair_collections()
    names = rag.collection_names(rag.chroma_client())
    assert rag.COLLECTION in names and rag.STAGING_COLLECTION not in names


def test_refresh_lock_is_exclusive_and_stale_locks_expire(data_dir):
    with rag.RefreshLock():
        with pytest.raises(RuntimeError):
            with rag.RefreshLock():
                pass
    assert not rag.REFRESH_LOCK_FILE.exists()
    rag.REFRESH_LOCK_FILE.write_text("1 old")
    import os
    import time
    old = time.time() - (rag.REFRESH_LOCK_STALE_HOURS + 1) * 3600
    os.utime(rag.REFRESH_LOCK_FILE, (old, old))
    with rag.RefreshLock():
        pass


def test_rate_limiter():
    limiter = rag.RateLimiter(per_minute=2, per_day=3)
    assert limiter.check("a", now=1000) is None
    assert limiter.check("a", now=1001) is None
    assert "too quickly" in limiter.check("a", now=1002)
    assert limiter.check("a", now=1061) is None              # a minute later
    assert "daily" in limiter.check("b", now=1062)           # 3 questions today in total


def test_question_log_and_feedback_report(data_dir, capsys):
    log = rag.QuestionLog()
    trace = {"tokens": rag.Counter(prompt=1000, completion=100), "refused": None, "cited": [1], "query": "q",
             "hits": []}
    log.write(rag.question_record("id1", "alice", "How do I install the KSP?", trace, "Run msiexec [1]"))
    log.write(rag.question_record("id2", "bob", "Pricing?", {**trace, "refused": "gate"}, rag.NO_ANSWER))
    log.write({"type": "feedback", "id": "id1", "time": rag.now_utc(), "user": "alice", "liked": False})
    rag.feedback_report(days=1)
    out = capsys.readouterr().out
    assert "2 questions from 2 users" in out and "1 👎" in out and "Pricing?" in out
    assert "How do I install the KSP?" in out


def test_health_reports_problems(data_dir, monkeypatch):
    assistant = type("A", (), {"retriever": type("R", (), {"reranker": None})()})()
    rag.write_index_info({"pages": 1, "chunks": 2, "last_checked_ts": 0,
                          "last_refresh": {"status": "held", "at": "x", "reason": "MRR dropped"}})
    report = rag.health(assistant, rag.UsageStats())
    assert report["status"] == "degraded" and len(report["problems"]) == 3


def test_tests_files_are_valid():
    tests = rag.load_tests()
    assert tests and all({"question", "relevant", "category", "set"} <= set(t) for t in tests)
    assert len({t["question"] for t in tests}) == len(tests), "duplicate questions"


def test_make_page_helper_is_consistent():
    page = make_page("https://docs.signpath.io/x", [("A", "text")])
    assert page["sections"][0]["path"] == ["Docs", "A"]
