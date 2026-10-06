"""
SignPath RAG - step 2: answer questions.

A Gradio chat UI that answers questions about SignPath using only the documentation stored in the
Chroma vector database by signpath_Ingest.py. For each question it:
  1. rewrites follow-up questions into a standalone question (using the chat history),
  2. finds candidate chunks with a vector search and a keyword (BM25) search,
  3. re-ranks the candidates with a cross-encoder model and refuses if nothing is relevant enough,
  4. asks the LLM to answer from the best chunks only, citing them, and shows their images.

Usage (from the project root, with the .venv; run signpath_Ingest.py first):
    .venv/Scripts/python.exe week5/signpath_answer.py                        # open the chat UI
    .venv/Scripts/python.exe week5/signpath_answer.py --ask "How do I sign with GPG?"   # answer in the terminal
    .venv/Scripts/python.exe week5/signpath_answer.py --eval                 # check the answers to known questions
    .venv/Scripts/python.exe week5/signpath_answer.py --eval-retrieval       # measure search quality (MRR, nDCG...)
"""

import argparse
import json
import math
import os
import re
import time
import warnings
from pathlib import Path

import chromadb
import gradio as gr
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Settings (DATA_DIR, COLLECTION and EMBED_MODEL must match signpath_Ingest.py)
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv("SIGNPATH_DATA_DIR", Path(__file__).parent / "signpath_data"))
CHROMA_DIR = DATA_DIR / "chroma"
INDEX_INFO_FILE = DATA_DIR / "index_info.json"
COLLECTION = "signpath_docs"

LLM_MODEL = "gpt-4.1-mini"
EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

VECTOR_K = 25              # candidates from the vector search
KEYWORD_K = 25             # candidates from the keyword search
RERANK_CANDIDATES = 30     # candidates given to the re-ranker
CONTEXT_K = 8              # chunks given to the LLM
RERANK_MIN_SCORE = -4.0    # if the best re-rank score is below this, the docs don't cover the question
RERANK_KEEP_SCORE = -8.0   # chunks scoring below this are never given to the LLM
RELATIVE_CUTOFF = 3.0      # ... nor chunks scoring more than this below the best one (see search, step 5)
MAX_IMAGES = 6             # images shown under an answer
BLEND_K = 10               # how strongly top positions count when blending re-rank order with search order
CHANGELOG_SECTION = "Product updates"   # changelog entries are only searched for release/version questions
CHANGELOG_QUESTION = re.compile(r"\b(release|released|version|versions|changelog|new in|what's new|"
                                r"update|updated|deprecat\w*)\b", re.I)
OPEN_BROWSER = os.getenv("SIGNPATH_OPEN_BROWSER", "1") == "1"
UI_UPDATE_SECONDS = float(os.getenv("SIGNPATH_UI_UPDATE_SECONDS", "0.15"))   # stream to the browser in batches at most this often (see respond)

NO_ANSWER = "I don't have information about that in the SignPath documentation I have indexed."

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
2. Reply with exactly this sentence and nothing else only when the excerpts contain nothing that helps \
answer the question: "{NO_ANSWER}". Never combine that sentence with an answer.
3. Questions can be worded loosely or as fragments ("To sign an MSI inside a ZIP?"); answer what the user \
means. If no single excerpt covers the exact case but the excerpts document the pieces needed (for example \
which file types can contain other files and how nested elements are written), combine those pieces into \
an answer and say that it is put together from the documented elements. If the excerpts answer only part \
of the question, answer that part and say clearly which part is not covered. Never claim that a product, \
integration or feature is supported when the excerpts don't mention it; say it isn't mentioned instead.
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

REWRITE_PROMPT = """Rewrite the user's latest message as a single standalone question about the SignPath \
documentation, resolving references like "it", "that" or "the previous example" from the conversation. \
Do not answer it. Output only the rewritten question."""

STOPWORDS = """a an and are as at be by can do does for from how i in is it its me my of on or
should so that the this to use using what when where which who why will with you your""".split()

CODE_FENCE = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)


def log(message):
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Loading the vector database
# ---------------------------------------------------------------------------

openai_client = None
reranker = None
current_index = None       # the loaded chunks and keyword index (see load_index)


def read_index_info():
    try:
        return json.loads(INDEX_INFO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_index():
    """Load all chunks from Chroma and build the keyword index. Returns a dict."""
    info = read_index_info()
    if not info:
        raise SystemExit(f"No vector database found in {DATA_DIR}. Run signpath_Ingest.py first.")
    if info.get("embed_model") != EMBED_MODEL:
        raise SystemExit(f"The database was built with {info.get('embed_model')}, but this program uses {EMBED_MODEL}.")

    db = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    collection = db.get_collection(COLLECTION)
    data = collection.get(include=["documents", "metadatas"])
    position = {}
    for i, chunk_id in enumerate(data["ids"]):
        position[chunk_id] = i
    log(f"Loaded {len(data['ids'])} chunks (database last changed {info.get('last_changed')})")
    return {
        "version": info.get("last_changed"),
        "collection": collection,
        "ids": data["ids"],
        "documents": data["documents"],
        "metadatas": data["metadatas"],
        "position": position,
        "keywords": build_keyword_index(data["documents"]),
    }


def reload_index_if_changed():
    """signpath_Ingest.py writes index_info.json when it changes the database; reload when that happens."""
    global current_index
    if read_index_info().get("last_changed") != current_index["version"]:
        log("The database has changed; reloading it")
        current_index = load_index()


def setup():
    global openai_client, reranker, current_index
    load_dotenv(override=True)
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (add it to your .env file)")
    openai_client = OpenAI()
    current_index = load_index()
    log(f"Loading re-ranker {RERANK_MODEL}...")
    reranker = CrossEncoder(RERANK_MODEL)


# ---------------------------------------------------------------------------
# Keyword search (BM25)
# ---------------------------------------------------------------------------

def tokenize(text):
    """Lower-case words; names like 'pe-file' are kept whole and also split into 'pe' and 'file'."""
    tokens = []
    for token in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()):
        if token not in STOPWORDS:
            tokens.append(token)
        parts = re.split(r"[-_.]", token)
        if len(parts) > 1:
            for part in parts:
                if part and part not in STOPWORDS:
                    tokens.append(part)
    return tokens


def build_keyword_index(documents):
    """Pre-compute what BM25 needs: how often each word appears in each chunk, each chunk's length,
    and how rare each word is across all chunks (rare words like 'certutil' matter more than common ones)."""
    term_counts = []           # for each document: {term: number of times it appears}
    lengths = []
    document_frequency = {}    # for each term: number of documents containing it
    for document in documents:
        counts = {}
        for token in tokenize(document):
            counts[token] = counts.get(token, 0) + 1
        term_counts.append(counts)
        lengths.append(sum(counts.values()))
        for term in counts:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    # IDF (inverse document frequency): high for words found in few chunks, close to 0 for words in almost all
    n = len(documents)
    idf = {}
    for term, df in document_frequency.items():
        idf[term] = math.log(1 + (n - df + 0.5) / (df + 0.5))
    average_length = sum(lengths) / max(n, 1)
    return {"term_counts": term_counts, "lengths": lengths, "average_length": average_length, "idf": idf}


def keyword_search(keywords, query, k, k1=1.5, b=0.75):
    """Return the positions of the k best documents for the query (BM25 scoring).

    For each query word found in a chunk, BM25 adds: idf * count * (k1 + 1) / (count + k1 * length_norm).
    - More occurrences score higher, but with diminishing returns (k1 controls how fast it levels off).
    - Long chunks are penalised a little (b controls how much), so a word in a short, focused chunk counts more.
    This complements the vector search: it finds exact names like 'Submit-SigningRequest' or '<pe-file>'."""
    terms = [term for term in set(tokenize(query)) if term in keywords["idf"]]
    scores = []
    for i, counts in enumerate(keywords["term_counts"]):
        score = 0.0
        for term in terms:
            count = counts.get(term, 0)
            if count:
                length_norm = k1 * (1 - b + b * keywords["lengths"][i] / keywords["average_length"])
                score += keywords["idf"][term] * count * (k1 + 1) / (count + length_norm)
        if score > 0:
            scores.append((score, i))
    scores.sort(reverse=True)
    return [i for score, i in scores[:k]]


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def embed_query(text):
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=[text])
    return response.data[0].embedding


def search(query):
    """Return (chunks to give the LLM, all ranked candidates). Each chunk is a dict."""
    reload_index_if_changed()
    index = current_index

    # 1. Vector search: embed the question and let Chroma find the chunks with the closest meaning
    #    (cosine distance; similarity = 1 - distance). Good at paraphrases, weaker on exact names.
    result = index["collection"].query(query_embeddings=[embed_query(query)], n_results=VECTOR_K,
                                       include=["distances"])
    vector_ids = []
    similarity = {}
    for chunk_id, distance in zip(result["ids"][0], result["distances"][0]):
        if chunk_id in index["position"]:
            vector_ids.append(chunk_id)
            similarity[chunk_id] = 1 - distance

    # 2. Keyword search (BM25): finds chunks containing the question's exact words
    keyword_ids = []
    for i in keyword_search(index["keywords"], query, KEYWORD_K):
        keyword_ids.append(index["ids"][i])

    # 3. Combine both rankings (reciprocal rank fusion). Each list gives a chunk 1 / (60 + its position):
    #    a chunk near the top of either list scores well, and one near the top of both scores best.
    #    The constant 60 is the usual choice; it stops the very first places from dominating.
    #    Only positions are used, so the different score scales of the two searches don't matter.
    fused = {}
    for ranking in (vector_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking):
            fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)
    best_ids = sorted(fused, key=fused.get, reverse=True)[:RERANK_CANDIDATES]

    candidates = []
    for fused_rank, chunk_id in enumerate(best_ids):
        i = index["position"][chunk_id]
        candidate = {"id": chunk_id, "text": index["documents"][i], "meta": index["metadatas"][i],
                     "similarity": similarity.get(chunk_id, 0.0), "rerank": 0.0,
                     "fused_rank": fused_rank}      # position before re-ranking (used for blending and --eval-retrieval)
        # Leave out changelog entries unless the question is about releases or versions. The changelog
        # ("Product updates") has hundreds of short entries that mention features in passing; they reached
        # the top 5 for 15 of the 83 test questions and pushed the real how-to sections down.
        if candidate["meta"]["section"].startswith(CHANGELOG_SECTION) and not CHANGELOG_QUESTION.search(query):
            continue
        candidates.append(candidate)

    # 4. Re-rank with the cross-encoder. The searches above compare the question and each chunk separately;
    #    the cross-encoder reads both together, so it judges relevance much more accurately (but is too slow
    #    to run on every chunk, which is why it only re-orders the top candidates). Its score is roughly:
    #    above 0 = relevant, below -4 = unrelated (e.g. +10 for the KSP install section vs -11 for "capital of France").
    pairs = [(query, candidate["text"][:2500]) for candidate in candidates]
    scores = reranker.predict(pairs)
    for candidate, score in zip(candidates, scores):
        candidate["rerank"] = float(score)
    candidates.sort(key=rerank_score, reverse=True)

    # Relevance gate: if even the best chunk scores below RERANK_MIN_SCORE, the docs don't cover the
    # question. Returning no chunks makes answer_question refuse without asking the LLM, so it can't guess.
    if not candidates or candidates[0]["rerank"] < RERANK_MIN_SCORE:
        return [], candidates

    # 4b. Blend the re-ranker's order with the search order (both as 1 / (BLEND_K + position)) instead of letting
    #     the small re-ranker overrule the search completely. A chunk that both rate highly wins; one the
    #     re-ranker alone likes (e.g. a similar-sounding section of another page) no longer jumps ahead of it.
    #     In the tests this raised MRR from 0.906 to 0.932 and fixed both top-5 misses.
    for rank, candidate in enumerate(candidates):
        candidate["blended"] = 1 / (BLEND_K + rank) + 1 / (BLEND_K + candidate["fused_rank"])
    candidates.sort(key=blended_score, reverse=True)

    # 5. Pick up to CONTEXT_K chunks for the LLM, best first. Skip chunks below the keep score, and
    #    code/image chunks whose content is already inside a chosen text chunk (no point sending it twice).
    #    Adaptive cut-off: a chunk must also score within RELATIVE_CUTOFF of the best chunk. When one section
    #    clearly answers the question, near-misses from other pages are left out (less noise, fewer tokens);
    #    when several sections score alike (combined questions), they all still go.
    best_score = candidates[0]["rerank"]
    for candidate in candidates:
        if candidate["rerank"] > best_score:
            best_score = candidate["rerank"]
    keep_from = max(RERANK_KEEP_SCORE, best_score - RELATIVE_CUTOFF)
    selected = []
    for candidate in candidates:
        if candidate["rerank"] < keep_from:
            continue      # not "break": after blending, a weaker chunk can come before a stronger one
        if is_duplicate(candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) == CONTEXT_K:
            break
    return selected, candidates


def rerank_score(candidate):
    return candidate["rerank"]


def blended_score(candidate):
    return candidate["blended"]


def is_duplicate(candidate, selected):
    kind = candidate["meta"]["type"]
    if kind == "code":
        body = candidate["text"].split("\n\n", 1)[-1]
        code = CODE_FENCE.search(body)
        if code:
            for chosen in selected:
                if code.group(2)[:300] in chosen["text"]:
                    return True
    if kind == "image":
        for chosen in selected:
            if chosen["meta"]["url"] == candidate["meta"]["url"]:
                return True
    return False


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

def message_text(message):
    """The text of a chat message (Gradio 6 stores the content as a list of parts)."""
    content = message.get("content", "")
    if isinstance(content, list):
        text = ""
        for part in content:
            if isinstance(part, dict):
                text += part.get("text", "")
        content = text
    return str(content).split("\n\n**Sources:**")[0]


def is_image_caption(text):
    """The chat shows images as extra messages: an "Images from the documentation" header and an italic caption
    per image. They carry no conversation, so they're left out of the history used to rewrite follow-ups
    (otherwise a few images would fill the 6-message window and push out the actual questions)."""
    text = text.strip()
    if text == "**Images from the documentation:**":
        return True
    return text.startswith("_") and text.endswith("_") and "\n" not in text


def remove_stray_refusal(answer):
    """The LLM sometimes explains what the excerpts say (e.g. "Bitbucket Pipelines isn't mentioned...") and then
    also appends the refusal sentence. Keep the explanation and drop the sentence; a plain refusal is unchanged."""
    if NO_ANSWER not in answer or answer.strip().strip('"') == NO_ANSWER:
        return answer
    cleaned = answer.replace('"' + NO_ANSWER + '"', "").replace(NO_ANSWER, "").strip()
    if cleaned:
        return cleaned
    return answer


def rewrite_question(question, history):
    """Turn a follow-up like 'and how do I test it?' into a standalone question."""
    if not history:
        return question
    conversation = []
    for message in history:
        if not is_image_caption(message_text(message)):
            conversation.append(message)
    transcript = ""
    for message in conversation[-6:]:
        transcript += message["role"] + ": " + message_text(message)[:1500] + "\n"
    response = openai_client.chat.completions.create(
        model=LLM_MODEL, temperature=0, seed=42,
        messages=[{"role": "system", "content": REWRITE_PROMPT},
                  {"role": "user", "content": f"Conversation:\n{transcript}\nLatest message: {question}"}])
    return response.choices[0].message.content.strip()


def answer_question(question, history):
    """Generator: yields (answer so far, sources markdown, images, retrieval details markdown)."""
    query = rewrite_question(question, history)
    relevant, candidates = search(query)
    details = format_retrieval_details(query, relevant, candidates)
    if not relevant:
        yield NO_ANSWER, "_No sufficiently relevant documentation was found, so the LLM was not asked._", [], details
        return

    excerpts = []
    for n, chunk in enumerate(relevant, start=1):
        excerpts.append(f"[{n}] Last updated: {chunk['meta'].get('lastmod') or 'unknown'}\n{chunk['text']}")
    user_message = "Documentation excerpts:\n\n" + "\n\n---\n\n".join(excerpts) + "\n\n---\n\nQuestion: " + query

    stream = openai_client.chat.completions.create(
        model=LLM_MODEL, temperature=0, seed=42, stream=True,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}])
    answer = ""
    for event in stream:
        if event.choices and event.choices[0].delta.content:
            answer += event.choices[0].delta.content
            yield answer, "", [], details

    answer = remove_stray_refusal(answer)

    # Which excerpts did the answer cite?
    cited = []
    for number in re.findall(r"\[(\d+)\]", answer):
        number = int(number)
        if 1 <= number <= len(relevant) and number not in cited:
            cited.append(number)
    cited.sort()

    if answer.strip().startswith("I don't have information") and not cited:
        yield answer, "_The retrieved documentation did not contain the answer._", [], details
        return

    if not cited:
        cited = list(range(1, len(relevant) + 1))
    used = [relevant[number - 1] for number in cited]
    links = []
    for number, chunk in zip(cited, used):
        links.append(f"[{number}] [{chunk['meta']['section']}]({chunk['meta']['url']})")
    answer += "\n\n**Sources:** " + " · ".join(links)
    yield answer, format_sources(relevant, cited), find_images(used), details


def format_sources(chunks, cited):
    lines = ["### Sources"]
    for n, chunk in enumerate(chunks, start=1):
        mark = "✅" if n in cited else "▫️"
        meta = chunk["meta"]
        updated = ""
        if meta.get("lastmod"):
            updated = ", page updated " + meta["lastmod"]
        lines.append(f"{mark} **[{n}]** [{meta['section']}]({meta['url']}) — "
                     f"_{meta['type']}, re-rank {chunk['rerank']:.1f}{updated}_")
    return "\n\n".join(lines)


def format_retrieval_details(query, relevant, candidates):
    used_ids = [chunk["id"] for chunk in relevant]
    lines = [f"**Search query:** {query}", "",
             "| # | Used | Score | Type | Section |", "| --- | --- | --- | --- | --- |"]
    for n, chunk in enumerate(candidates[:15], start=1):
        used = "yes" if chunk["id"] in used_ids else ""
        section = chunk["meta"]["section"].replace("|", "\\|")
        lines.append(f"| {n} | {used} | {chunk['rerank']:.2f} | {chunk['meta']['type']} | {section} |")
    return "\n".join(lines)


def find_images(chunks):
    """(file path, caption) of the images in the cited sections, in the order they appear in the docs."""
    chunks = sorted(chunks, key=document_position)
    images = []
    seen = []
    for chunk in chunks:
        for image in json.loads(chunk["meta"].get("images") or "[]"):
            path = DATA_DIR / image["path"]
            if image["path"] in seen or not path.exists():
                continue
            seen.append(image["path"])
            caption = image["alt"] or chunk["meta"]["section"]
            images.append((str(path), caption))
    return images[:MAX_IMAGES]


def document_position(chunk):
    return chunk["meta"].get("position", 0)


def final_answer(question):
    """Run answer_question to the end and return its last result."""
    result = None
    for result in answer_question(question, []):
        pass
    return result


# ---------------------------------------------------------------------------
# Evaluation: known questions with expected words, or None when the assistant must refuse
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
    ("What are the risks of storing code signing keys in PFX files?", ["pfx"]),
    ("What is Microsoft SmartScreen and how does it relate to code signing?", ["smartscreen"]),
    ("How can I create a self-signed test certificate?", ["self-signed"]),
    ("What are SignPath's pricing tiers?", None),
    ("How do I configure AWS KMS as a key store in SignPath?", None),
    ("What is the capital of France?", None),
]


def is_refusal(answer):
    return answer.strip().startswith("I don't have information") and "**Sources:**" not in answer


def run_eval():
    passed_count = 0
    for question, expected_words in EVAL_CASES:
        answer = final_answer(question)[0]
        if expected_words is None:
            passed = is_refusal(answer)
            detail = "refused" if passed else "answered (should refuse)"
        elif is_refusal(answer):
            passed = False
            detail = "refused (should answer)"
        else:
            missing = [word for word in expected_words if word.lower() not in answer.lower()]
            passed = not missing
            detail = "ok" if passed else f"missing {missing}"
        if passed:
            passed_count += 1
        log(f"  {'PASS' if passed else 'FAIL'}  {question}  ->  {detail}")
    log(f"Evaluation: {passed_count}/{len(EVAL_CASES)} passed")
    return passed_count == len(EVAL_CASES)


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
#   1. "Vector + keyword" - the combined search results, before re-ranking
#   2. "After re-rank"    - the final order: cross-encoder scores blended with the search order (step 4b)
#   3. "Sent to the LLM"  - the chunks finally given to the LLM (after the relevance gate and de-duplication)
# Comparing 1 and 2 shows how much the re-ranker helps; 3 shows what the LLM actually gets to read.
#
# The metrics (each is between 0 and 1, higher is better; the reported value is the average over questions):
#   MRR (mean reciprocal rank) - 1 / position of the first relevant chunk: 1.0 if it is first, 0.5 if
#                                second, 0.33 if third ... and 0 if no relevant chunk was found at all.
#   nDCG@K - looks at all relevant chunks in the top K and rewards higher positions more; 1.0 means the
#            ranking is as good as possible (all relevant chunks before any irrelevant ones).
#   Recall@K - share of the expected sections that appear at least once in the top K
#              ("did we find everything we need?").
#   Precision@K - share of the top K chunks that come from an expected section ("how much is noise?").
#              Low precision is normal when a question has only one relevant section and K is 5.

RETRIEVAL_TESTS_FILE = Path(__file__).parent / "signpath_retrieval_tests.json"
EVAL_K = 5                 # K for the search rankings; the "Sent to the LLM" list uses CONTEXT_K


def is_relevant(meta, relevant_sections):
    """True if a chunk (given by its metadata) comes from an expected section or one of its sub-sections."""
    section = meta["section"]
    for expected in relevant_sections:
        if section == expected or section.startswith(expected + " > "):
            return True
    return False


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
    dcg = 0.0
    for position, flag in enumerate(flags[:k], start=1):
        if flag:
            dcg += 1.0 / math.log2(position + 1)
    ideal_dcg = 0.0
    for position in range(1, min(total_relevant, k) + 1):
        ideal_dcg += 1.0 / math.log2(position + 1)
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def recall_at_k(ranking, relevant_sections, k):
    """Share of the expected sections that have at least one chunk in the top k."""
    found = 0
    for expected in relevant_sections:
        for chunk in ranking[:k]:
            if is_relevant(chunk["meta"], [expected]):
                found += 1
                break
    return found / len(relevant_sections)


def precision_at_k(flags, k):
    """Share of the top k chunks that are relevant. When fewer than k chunks were returned (the LLM can get
    fewer than CONTEXT_K), it is the share of the returned chunks."""
    top = flags[:k]
    if not top:
        return 0.0
    return sum(top) / len(top)


def score_ranking(ranking, relevant_sections, k, total_relevant):
    """All four metrics for one ranked list of chunks."""
    flags = [is_relevant(chunk["meta"], relevant_sections) for chunk in ranking]
    return {"MRR": reciprocal_rank(flags),             # MRR looks at the whole list, not just the top k
            "nDCG": ndcg_at_k(flags, k, total_relevant),
            "Recall": recall_at_k(ranking, relevant_sections, k),
            "Precision": precision_at_k(flags, k)}


def count_relevant_chunks(relevant_sections):
    """How many chunks in the whole index come from the expected sections (needed for nDCG's best case)."""
    count = 0
    for meta in current_index["metadatas"]:
        if is_relevant(meta, relevant_sections):
            count += 1
    return count


def fused_position(candidate):
    return candidate["fused_rank"]


def run_retrieval_eval(k=EVAL_K):
    """Score retrieval on every test question and print per-question results and averages per stage.
    Costs one small embedding call per question (no LLM calls)."""
    tests = json.loads(RETRIEVAL_TESTS_FILE.read_text(encoding="utf-8"))["tests"]
    stages = [("Vector + keyword", k), ("After re-rank", k), ("Sent to the LLM", CONTEXT_K)]
    totals = {}
    for stage, stage_k in stages:
        totals[stage] = {"MRR": 0.0, "nDCG": 0.0, "Recall": 0.0, "Precision": 0.0}

    # Out-of-scope questions have no relevant sections, so there is nothing to rank: instead they check the
    # relevance gate, which should reject them (search returns no chunks for the LLM)
    answerable = []
    out_of_scope = []
    for test in tests:
        if test["relevant"]:
            answerable.append(test)
        else:
            out_of_scope.append(test)

    log(f"Retrieval evaluation on {len(tests)} questions from {RETRIEVAL_TESTS_FILE.name}")
    log("Out-of-scope questions (the relevance gate should reject them):")
    rejected = 0
    for test in out_of_scope:
        selected, candidates = search(test["question"])
        if not selected:
            rejected += 1
            log("  rejected ✓      " + test["question"])
        else:
            log("  NOT rejected ✗  " + test["question"])

    tests = answerable
    log(f"Answerable questions, after re-ranking (K = {k}):")
    log("   MRR  Recall  Question")
    false_rejections = 0
    for test in tests:
        relevant_sections = test["relevant"]
        total_relevant = count_relevant_chunks(relevant_sections)
        if total_relevant == 0:
            # The label doesn't match anything: a typo in the test file, or the docs were reorganised
            log(f"  ! no chunk in the index belongs to {relevant_sections}; check the test file")

        selected, candidates = search(test["question"])     # candidates are sorted by re-rank score
        if not selected:
            false_rejections += 1      # the gate rejected an answerable question: the assistant would refuse
        before_rerank = sorted(candidates, key=fused_position)
        rankings = {"Vector + keyword": before_rerank, "After re-rank": candidates, "Sent to the LLM": selected}

        for stage, stage_k in stages:
            scores = score_ranking(rankings[stage], relevant_sections, stage_k, total_relevant)
            for metric in scores:
                totals[stage][metric] += scores[metric]

        after = score_ranking(candidates, relevant_sections, k, total_relevant)
        warning = "   <- relevant section not in the top K" if after["Recall"] < 1 else ""
        log(f"  {after['MRR']:4.2f}   {after['Recall']:4.2f}  {test['question']}{warning}")

    log("")
    log(f"Averages over {len(tests)} questions (higher is better, 1.0 is perfect):")
    log(f"  {'Stage':28} {'MRR':>6} {'nDCG@K':>8} {'Recall@K':>9} {'Precision@K':>12}")
    for stage, stage_k in stages:
        label = f"{stage} (K={stage_k})"
        averages = {}
        for metric in totals[stage]:
            averages[metric] = totals[stage][metric] / len(tests)
        log(f"  {label:28} {averages['MRR']:6.3f} {averages['nDCG']:8.3f} "
            f"{averages['Recall']:9.3f} {averages['Precision']:12.3f}")
    log(f"Relevance gate: {rejected}/{len(out_of_scope)} out-of-scope questions rejected, "
        f"{false_rejections} answerable questions wrongly rejected")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLES = [
    "How do I install and configure the Windows KSP crypto provider?",
    "Show an artifact configuration example for signing an MSI inside a ZIP file.",
    "How do I submit a signing request from GitHub Actions?",
    "What crypto providers does SignPath support?",
    "How do I sign with GPG using SignPath?",
    "How do I set up user synchronization from Microsoft Entra ID?",
    "What are the risks of storing code signing keys in PFX files?",
]


def database_status():
    info = read_index_info()
    return (f"📚 {info.get('pages', '?')} pages · {info.get('chunks', '?')} chunks · "
            f"docs last checked {info.get('last_checked', '?')} · database last changed {info.get('last_changed', '?')}")


def respond(message, history):
    """Called by Gradio for each question. Yields updates for: question box, chat, sources, details."""
    if not message.strip():
        yield "", history, gr.skip(), gr.skip()
        return

    # The LLM only needs the text of earlier turns; the chat keeps showing earlier answers' images
    previous = []
    for old_message in history:
        text = message_text(old_message)
        if text.strip():
            previous.append({"role": old_message["role"], "content": text})

    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "_Searching the documentation..._"}]
    answer_position = len(history) - 1
    yield "", history, "", ""

    try:
        images = []
        # Every update re-sends the whole chat (images included) to the browser, which redraws it. Updating on
        # every token made long chats crawl (measured: 109 tok/s on screen for the 1st answer, 22 by the 4th), so
        # the text is sent in small batches, at most every UI_UPDATE_SECONDS. The final update is always sent.
        last_update = 0.0
        pending = None
        for answer, sources, images, details in answer_question(message, previous):
            history[answer_position] = {"role": "assistant", "content": answer}
            pending = ("", history, sources or gr.skip(), details)
            if sources or time.monotonic() - last_update >= UI_UPDATE_SECONDS:
                last_update = time.monotonic()
                yield pending
                pending = None
        if pending:
            yield pending
        if images:
            history.append({"role": "assistant", "content": "**Images from the documentation:**"})
            for path, caption in images:
                history.append({"role": "assistant", "content": f"_{caption}_"})
                history.append({"role": "assistant", "content": {"path": path, "alt_text": caption}})
            yield "", history, gr.skip(), gr.skip()
    except Exception as error:
        history[answer_position] = {"role": "assistant", "content": f"⚠️ Error: {error}"}
        yield "", history, "", ""


def clear_chat():
    return "", [], "### Sources", ""


def launch_ui():
    with gr.Blocks(title="SignPath Docs Assistant") as ui:
        gr.Markdown("# SignPath Documentation Assistant\n"
                    "Answers come only from the indexed [SignPath Documentation](https://docs.signpath.io/) and "
                    "[SignPath Knowledge Base](https://signpath.io/knowledge-base). "
                    "If they don't cover a question, it says so.")
        gr.Markdown(database_status)       # a function, so it is re-read every time the page loads
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=620)
                question = gr.Textbox(placeholder="Ask about setup, configuration, crypto providers, build integrations...",
                                      show_label=False, lines=2)
                with gr.Row():
                    ask_button = gr.Button("Ask", variant="primary")
                    clear_button = gr.Button("Clear")
                gr.Examples(EXAMPLES, inputs=question)
            with gr.Column(scale=2):
                sources = gr.Markdown("### Sources\n_Sources for each answer appear here._")
                with gr.Accordion("Retrieval details (ranking)", open=False):
                    details = gr.Markdown()

        outputs = [question, chatbot, sources, details]
        question.submit(respond, [question, chatbot], outputs)
        ask_button.click(respond, [question, chatbot], outputs)
        clear_button.click(clear_chat, None, outputs)

    # allowed_paths lets Gradio serve the downloaded documentation images
    ui.launch(inbrowser=OPEN_BROWSER, allowed_paths=[str(DATA_DIR)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Answer questions about SignPath from the vector database")
    parser.add_argument("--ask", help="answer one question in the terminal instead of opening the UI")
    parser.add_argument("--eval", action="store_true", help="run the evaluation questions (exit code 1 if any fail)")
    parser.add_argument("--eval-retrieval", action="store_true",
                        help="measure search quality (MRR, nDCG@K, Recall@K, Precision@K) on signpath_retrieval_tests.json")
    parser.add_argument("--k", type=int, default=EVAL_K, help="K for --eval-retrieval (default 5)")
    args = parser.parse_args()

    setup()
    if args.eval:
        if not run_eval():
            raise SystemExit(1)
    elif args.eval_retrieval:
        run_retrieval_eval(args.k)
    elif args.ask:
        answer, sources, images, details = final_answer(args.ask)
        print("\n" + answer + "\n\n" + sources + "\n")
        for path, caption in images:
            print(f"Image: {caption}: {path}")
        print("\n" + details)
    else:
        launch_ui()


if __name__ == "__main__":
    main()
