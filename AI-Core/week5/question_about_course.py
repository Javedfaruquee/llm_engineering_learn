import os
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # OpenRouter-compatible API
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
import gradio as gr
import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)  # reads OPENROUTER_API_KEY from the project's .env

# OpenRouter: main model for answering
MODEL_ANSWER = "openai/gpt-4o-mini"
# Optional: cheaper/smaller for rewrite, rerank, judge
MODEL_REWRITE = "openai/gpt-4o-mini"
MODEL_RERANK = "openai/gpt-4o-mini"
MODEL_JUDGE = "openai/gpt-4o-mini"

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 5
RERANK_TOP_N = 5

# Data: try repo knowledge-base, then fallback to in-notebook sample
NOTEBOOK_DIR = Path(".").resolve()
DATA_DIRS = [
    # NOTEBOOK_DIR / "../../knowledge-base",
    NOTEBOOK_DIR / "../knowledge-base",
    Path("knowledge-base"),
]
CHROMA_DIR = NOTEBOOK_DIR / "chroma_copilot"

#Load & chunk documents
def load_documents_from_dir(data_dir: Path) -> list[dict]:
    """Load .md and .txt files from a directory (recursive). Return list of {text, source}."""
    docs = []
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return docs
    for ext in ("*.md", "*.txt"):
        for path in data_dir.rglob(ext):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    docs.append({"text": text, "source": path.relative_to(data_dir).as_posix()})
            except Exception as e:
                print(f"Skip {path}: {e}")
    return docs


def get_fallback_docs() -> list[dict]:
    """In-notebook fallback: small sample for demo when no data folder is available."""
    return [
        {"text": "# RAG Overview\n\nRetrieval-augmented generation (RAG) combines retrieval with LLM generation. You embed documents, store them in a vector store like Chroma, then retrieve relevant chunks for each query and pass them as context to the LLM.", "source": "fallback/rag.md"},
        {"text": "# Embeddings\n\nEmbeddings are dense vector representations of text. Models like sentence-transformers (e.g. all-MiniLM-L6-v2) produce fixed-size vectors so similar texts are close in vector space.", "source": "fallback/embeddings.md"},
        {"text": "# Chunking\n\nChunking splits long documents into smaller segments. Common strategies: fixed size with overlap, sentence boundaries, or semantic splitting. Chunk size 500 with overlap 100 is a typical starting point.", "source": "fallback/chunking.md"},
        {"text": "# Query rewriting\n\nQuery rewriting improves retrieval by expanding or clarifying the user question. Use a small LLM to turn the question into a search-optimized query or multiple queries.", "source": "fallback/query_rewrite.md"},
        {"text": "# Reranking\n\nReranking takes a larger set of retrieved candidates and scores them for relevance. A cross-encoder or LLM can rerank so the top-k chunks are better suited for answer generation.", "source": "fallback/rerank.md"},
    ]


def load_data() -> list[dict]:
    for d in DATA_DIRS:
        resolved = Path(d).resolve()
        docs = load_documents_from_dir(resolved)
        if docs:
            print(f"Loaded {len(docs)} documents from {resolved}")
            return docs
    print("No data folder found; using in-notebook fallback.")
    return get_fallback_docs()


documents = load_data()

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap if end < len(words) else len(words)
    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    out = []
    for doc in docs:
        for i, c in enumerate(chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
            out.append({"text": c, "source": doc["source"], "chunk_id": i})
    return out


chunks = build_chunks(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} documents.")

#Embedding and Chroma

def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embed_model = get_embedding_model()


def embed(texts: list[str], model=None):
    model = model or embed_model
    return model.encode(texts, show_progress_bar=len(texts) > 20).tolist()


def build_chroma(chunks: list[dict], persist_path: Path, recreate: bool = True):
    persist_path = Path(persist_path)
    if recreate and persist_path.exists():
        import shutil
        shutil.rmtree(persist_path, ignore_errors=True)
    persist_path.mkdir(parents=True, exist_ok=True)

    texts = [c["text"] for c in chunks]
    ids = [f"c{i}" for i in range(len(chunks))]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]

    client = chromadb.PersistentClient(path=str(persist_path), settings=Settings(anonymized_telemetry=False))
    coll = client.get_or_create_collection("copilot", metadata={"description": "RAG chunks"})

    # Embed in batches for large corpora
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]
        emb = embed(batch_texts)
        coll.add(ids=batch_ids, embeddings=emb, metadatas=batch_meta, documents=batch_texts)

    return client, coll


chroma_client, chroma_coll = build_chroma(chunks, CHROMA_DIR)
print(f"Chroma collection: {chroma_coll.count()} vectors.")

# Visualize in 2D with t-SNE and 3D with PCA
# Get all embeddings and metadata from Chroma for visualization
n_total = chroma_coll.count()
result = chroma_coll.get(include=["embeddings", "documents", "metadatas"], limit=n_total)
vectors = np.array(result["embeddings"])
documents_viz = result["documents"]
metadatas_viz = result["metadatas"]
# Use source (e.g. week1/day1.md) for labels; use first path segment for color grouping
source_labels = [m.get("source", "?") for m in metadatas_viz]
doc_types = [s.split("/")[0] if "/" in s else s for s in source_labels]

doc_list = sorted(set(doc_types))
color_palette = px.colors.qualitative.Plotly
color_map = [color_palette[i % len(color_palette)] for i in range(len(doc_list))]
colors = [color_map[doc_list.index(t)] for t in doc_types]

# 2D t-SNE
tsne_2d = TSNE(n_components=2, random_state=42)
reduced_2d = tsne_2d.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter(
    x=reduced_2d[:, 0],
    y=reduced_2d[:, 1],
    mode="markers",
    marker=dict(size=4, color=colors, opacity=0.8),
    text=[f"Source: {s}<br>Text: {d[:80]}..." for s, d in zip(source_labels, documents_viz)],
    hoverinfo="text",
)])
fig.update_layout(
    title="2D Chroma vector store (t-SNE)",
    xaxis_title="t-SNE 1",
    yaxis_title="t-SNE 2",
    width=800,
    height=600,
    margin=dict(r=20, b=10, l=10, t=40),
)
fig.show()

# 3D t-SNE
tsne_3d = TSNE(n_components=3, random_state=43)
reduced_3d = tsne_3d.fit_transform(vectors)

fig = go.Figure(data=[go.Scatter3d(
    x=reduced_3d[:, 0],
    y=reduced_3d[:, 1],
    z=reduced_3d[:, 2],
    mode="markers",
    marker=dict(size=4, color=colors, opacity=0.8),
    text=[f"Source: {s}<br>Text: {d[:80]}..." for s, d in zip(source_labels, documents_viz)],
    hoverinfo="text",
)])
fig.update_layout(
    title="3D Chroma vector store (t-SNE)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    width=900,
    height=700,
    margin=dict(r=20, b=10, l=10, t=40),
)
fig.show()

#Baseline Retrieval 
def baseline_retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    q_emb = embed([query])
    res = chroma_coll.query(query_embeddings=q_emb, n_results=top_k, include=["documents", "metadatas", "distances"])
    out = []
    for i, (doc, meta, dist) in enumerate(zip(res["documents"][0], res["metadatas"][0], res["distances"][0])):
        out.append({"text": doc, "source": meta.get("source", "?"), "chunk_id": meta.get("chunk_id", 0), "distance": dist, "rank": i + 1})
    return out


# Open Router helpers

def get_openrouter_client():
    if not OPENROUTER_API_KEY:
        raise ValueError("Set OPENROUTER_API_KEY in the environment.")
    return OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)


def openrouter_completion(messages: list[dict], model: str = MODEL_ANSWER) -> str:
    client = get_openrouter_client()
    r = client.chat.completions.create(model=model, messages=messages, temperature=0.3)
    return r.choices[0].message.content or ""


def rewrite_query(question: str) -> str:
    messages = [
        {"role": "user", "content": f"Rewrite this question into a single, clear search query for finding relevant passages in a knowledge base. Output only the rewritten query, no explanation.\n\nQuestion: {question}"}
    ]
    return openrouter_completion(messages, model=MODEL_REWRITE).strip()


def rerank_chunks(question: str, chunks: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    if len(chunks) <= top_n:
        return chunks[:top_n]
    # Build a single prompt: score each chunk 0-10 for relevance to the question
    lines = [f"[Chunk {i+1}] {c['text'][:400]}..." if len(c["text"]) > 400 else f"[Chunk {i+1}] {c['text']}" for i, c in enumerate(chunks)]
    prompt = f"Question: {question}\n\nChunks:\n" + "\n".join(lines)
    prompt += "\n\nReply with only a single line of comma-separated scores (one integer 0-10 per chunk, same order). Example: 8,3,7,2,9"
    messages = [{"role": "user", "content": prompt}]
    raw = openrouter_completion(messages, model=MODEL_RERANK).strip()
    scores = []
    for part in raw.replace(" ", "").split(","):
        try:
            scores.append(int(part))
        except ValueError:
            scores.append(0)
    while len(scores) < len(chunks):
        scores.append(0)
    scored = list(zip(chunks, scores[: len(chunks)]))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:top_n]]


def answer_with_context(question: str, context_chunks: list[dict], model: str = MODEL_ANSWER) -> str:
    context = "\n\n".join([f"[{c.get('source', '?')}] {c['text']}" for c in context_chunks])
    system = "You are a helpful assistant. Answer the question using only the provided context. If the context does not contain the answer, say so. Keep answers concise and cite the source when relevant."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return openrouter_completion(messages, model=model)


def llm_judge_score(question: str, reference: str, answer: str) -> dict:
    prompt = f"Question: {question}\nReference answer: {reference}\nModel answer: {answer}\n\nRate the model answer: 1 (wrong/incomplete) to 5 (accurate and complete). Reply with one line: score=X (and optional one sentence feedback)."
    raw = openrouter_completion([{"role": "user", "content": prompt}], model=MODEL_JUDGE)
    score = 3
    for part in raw.split():
        if "score=" in part.lower():
            try:
                score = int(part.split("=")[-1].strip("."))
                score = max(1, min(5, score))
            except (ValueError, IndexError):
                pass
            break
    return {"score": score, "feedback": raw[:200]}

#Improved retrieval pipeline

def improved_retrieve(query: str, top_k: int = DEFAULT_TOP_K, rerank_top: int = RERANK_TOP_N) -> list[dict]:
    rewritten = rewrite_query(query)
    base = baseline_retrieve(query, top_k=top_k * 2)
    rew = baseline_retrieve(rewritten, top_k=top_k * 2)
    seen = set()
    merged = []
    for c in base + rew:
        key = (c["text"][:80], c["source"])
        if key not in seen:
            seen.add(key)
            merged.append(c)
    reranked = rerank_chunks(query, merged, top_n=rerank_top)
    return reranked

def format_chunks_display(chunks: list[dict]) -> str:
    return "\n\n---\n".join([f"**[{c.get('source', '?')}]** (rank {c.get('rank', i+1)})\n{c['text']}" for i, c in enumerate(chunks)])


def answer_question(question: str, mode: str = "improved", top_k: int = DEFAULT_TOP_K) -> tuple[str, str]:
    if mode == "baseline":
        context_chunks = baseline_retrieve(question, top_k=top_k)
    else:
        context_chunks = improved_retrieve(question, top_k=top_k, rerank_top=min(RERANK_TOP_N, top_k))
    answer = answer_with_context(question, context_chunks)
    chunks_display = format_chunks_display(context_chunks)
    return answer, chunks_display

# Evaluation 

# Evlauation 
# Load golden test set: tests.jsonl (same format as week5/evaluation) or in-notebook fallback
def load_eval_tests() -> list[dict]:
    path = NOTEBOOK_DIR / "tests.jsonl"
    if path.exists():
        tests = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                tests.append({
                    "question": row["question"],
                    "keywords": row["keywords"],
                    "reference": row.get("reference_answer", row.get("reference", "")),
                    "category": row.get("category", "direct_fact"),
                })
        if tests:
            print(f"Loaded {len(tests)} test questions from {path}")
            return tests
    # Fallback: small in-notebook set
    return [
        {"question": "What is RAG?", "keywords": ["retrieval", "generation", "embed"], "reference": "RAG combines retrieval with LLM generation; you embed documents and retrieve relevant chunks."},
        {"question": "What is a good chunk size?", "keywords": ["500", "overlap", "100"], "reference": "Chunk size 500 with overlap 100 is a typical starting point."},
        {"question": "How does query rewriting help?", "keywords": ["rewriting", "search", "query"], "reference": "Query rewriting improves retrieval by expanding or clarifying the user question."},
        {"question": "What are embeddings?", "keywords": ["vector", "sentence", "similar"], "reference": "Embeddings are dense vector representations; similar texts are close in vector space."},
    ]


EVAL_TESTS = load_eval_tests()


def calc_mrr(keywords: list[str], chunks: list[dict]) -> float:
    mrr = 0.0
    for kw in keywords:
        kw_lower = kw.lower()
        for rank, c in enumerate(chunks, start=1):
            if kw_lower in c["text"].lower():
                mrr += 1.0 / rank
                break
    return mrr / len(keywords) if keywords else 0.0


def calc_keyword_coverage(keywords: list[str], chunks: list[dict]) -> float:
    found = sum(1 for kw in keywords if any(kw.lower() in c["text"].lower() for c in chunks))
    return (found / len(keywords) * 100) if keywords else 0.0


def run_eval(mode: str, top_k: int = 5) -> dict:
    mrr_list, cov_list, judge_scores = [], [], []
    for t in EVAL_TESTS:
        if mode == "baseline":
            chunks = baseline_retrieve(t["question"], top_k=top_k)
        else:
            chunks = improved_retrieve(t["question"], top_k=top_k)
        mrr_list.append(calc_mrr(t["keywords"], chunks))
        cov_list.append(calc_keyword_coverage(t["keywords"], chunks))
        answer = answer_with_context(t["question"], chunks)
        judge = llm_judge_score(t["question"], t["reference"], answer)
        judge_scores.append(judge["score"])
    return {
        "MRR": sum(mrr_list) / len(mrr_list) if mrr_list else 0,
        "Keyword_cov_%": sum(cov_list) / len(cov_list) if cov_list else 0,
        "Judge_avg": sum(judge_scores) / len(judge_scores) if judge_scores else 0,
    }


if OPENROUTER_API_KEY:
    base_metrics = run_eval("baseline")
    impr_metrics = run_eval("improved")
    eval_df = pd.DataFrame({"Baseline": base_metrics, "Improved": impr_metrics}).T
    print(eval_df)
    print("\nImproved performs better on average:", impr_metrics["Judge_avg"] >= base_metrics["Judge_avg"])

#Gradio UI

def gradio_answer(question: str, mode: str, top_k: int) -> tuple[str, str, str]:
    if not question or not question.strip():
        return "", "", "Ask a question above."
    if not OPENROUTER_API_KEY:
        return "", "", "Error: Set OPENROUTER_API_KEY."
    try:
        answer, chunks_display = answer_question(question.strip(), mode=mode, top_k=top_k)
        return answer, chunks_display, f"Model: {MODEL_ANSWER} | Mode: {mode} | top_k: {top_k}"
    except Exception as e:
        return "", "", f"Error: {e}"


with gr.Blocks(title="AI Course Knowledge Copilot") as demo:
    gr.Markdown("# AI Course Knowledge Copilot")
    gr.Markdown("Ask a question about the course materials. Choose **baseline** (direct retrieval) or **improved** (query rewrite + rerank).")
    with gr.Row():
        mode = gr.Dropdown(choices=["improved", "baseline"], value="improved", label="Retrieval mode")
        top_k = gr.Slider(1, 15, value=5, step=1, label="Top-k chunks")
    question = gr.Textbox(placeholder="e.g. What is RAG? How does chunking work?", label="Question", lines=2)
    submit_btn = gr.Button("Get answer")
    model_info = gr.Textbox(label="Model / settings", interactive=False)
    answer_out = gr.Textbox(label="Answer", lines=6, interactive=False)
    chunks_out = gr.Textbox(label="Retrieved context / supporting chunks", lines=10, interactive=False)

    submit_btn.click(
        fn=lambda q, m, k: gradio_answer(q, m, int(k)),
        inputs=[question, mode, top_k],
        outputs=[answer_out, chunks_out, model_info],
    )

demo.launch(share=False, inbrowser=True, theme=gr.themes.Soft())  # Gradio 6 takes the theme in launch()