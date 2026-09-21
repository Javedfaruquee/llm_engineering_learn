"""
╔══════════════════════════════════════════════════════════════╗
║         RESUME–JOB DESCRIPTION MATCHING SYSTEM              ║
║         Powered by HuggingFace · Gradio UI                  ╠
╠══════════════════════════════════════════════════════════════╣
║  Models:                                                     ║
║   · sentence-transformers/all-MiniLM-L6-v2  (Embeddings)    ║
║   · dslim/bert-base-NER                     (NER)           ║
╚══════════════════════════════════════════════════════════════╝

Install:
  pip install gradio transformers sentence-transformers torch numpy scikit-learn
"""

import re
import sys
import numpy as np
import gradio as gr

# When output is redirected on Windows, stdout falls back to cp1252, which
# cannot encode the emoji in the log lines below and would crash at import.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
# Models are loaded once at import time and shared by every request, since
# loading takes seconds and would be far too slow to repeat per click.
# The first run downloads them from the Hugging Face Hub and caches them locally.
print("⏳  Loading models …")

# Turns a whole document into one 384-number vector capturing its meaning
EMBED_MODEL   = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# BERT fine-tuned to tag words as person / organisation / location / misc.
# The tokenizer splits text into the sub-word pieces the model understands.
NER_TOKENIZER = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
NER_MODEL     = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# aggregation_strategy="simple" merges sub-word pieces back into whole words,
# so "PyTorch" comes back as one entity instead of "Py", "##Tor", "##ch".
NER_PIPE      = pipeline("ner", model=NER_MODEL, tokenizer=NER_TOKENIZER,
                          aggregation_strategy="simple")
print("✅  Models ready.")

# ─────────────────────────────────────────────
#  TECH SKILLS KEYWORD BANK
# ─────────────────────────────────────────────
TECH_SKILLS = {
    "python","java","javascript","typescript","golang","go","rust","c++","c#",
    "ruby","swift","kotlin","scala","r","matlab","bash","shell","php","perl",
    "dart","elixir","haskell","lua","groovy",
    "react","vue","angular","svelte","nextjs","next.js","nuxt","html","css",
    "sass","tailwind","bootstrap","jquery","webpack","vite","astro",
    "node","node.js","express","fastapi","django","flask","spring","rails",
    "graphql","rest","grpc","websocket","oauth",
    "tensorflow","pytorch","keras","scikit-learn","sklearn","pandas","numpy",
    "spark","hadoop","airflow","mlflow","huggingface","transformers","langchain",
    "openai","llm","nlp","bert","gpt","xgboost","lightgbm","matplotlib",
    "seaborn","plotly","computer vision","cv",
    "sql","mysql","postgresql","postgres","mongodb","redis","elasticsearch",
    "cassandra","dynamodb","neo4j","sqlite","snowflake","bigquery",
    "docker","kubernetes","k8s","aws","gcp","azure","terraform","ansible",
    "jenkins","github actions","ci/cd","linux","nginx","apache","helm",
    "git","github","gitlab","jira","agile","scrum","tdd","microservices",
    "kafka","rabbitmq","celery","prometheus","grafana","dbt","looker","tableau",
    "power bi","figma","sketch","cypress","jest","pytest","selenium",
}

# ─────────────────────────────────────────────
#  5 BUILT-IN JOB DESCRIPTIONS
# ─────────────────────────────────────────────
PRESET_JDS = {

    "🤖  Senior ML Engineer": """\
Senior Machine Learning Engineer — DataDriven Inc.

ABOUT THE ROLE
We are looking for a Senior ML Engineer to design, build, and deploy
machine learning models at scale across our data platform.

REQUIRED SKILLS
Python, PyTorch, scikit-learn, FastAPI, Docker, Kubernetes, AWS,
PostgreSQL, MLflow, Git, CI/CD, pandas, numpy

NICE TO HAVE
Spark, Kafka, Airflow, Terraform, Grafana, Huggingface, LLM

RESPONSIBILITIES
· Develop and maintain ML models for production
· Build scalable REST APIs using FastAPI
· Deploy models to AWS using Docker and Kubernetes
· Monitor model performance with Prometheus and Grafana
· Collaborate via Git and GitHub Actions
""",

    "🌐  Full Stack Web Developer": """\
Full Stack Web Developer — WebNova Agency

ABOUT THE ROLE
Join our team building modern, performant web applications for
enterprise clients across fintech and e-commerce.

REQUIRED SKILLS
JavaScript, TypeScript, React, Node.js, Express, PostgreSQL,
MongoDB, Docker, Git, REST, HTML, CSS, Tailwind

NICE TO HAVE
Next.js, GraphQL, Redis, AWS, CI/CD, Jest, Cypress, Webpack

RESPONSIBILITIES
· Build responsive UI components with React and TypeScript
· Design and maintain RESTful and GraphQL APIs with Node.js
· Manage PostgreSQL and MongoDB databases
· Write unit and integration tests with Jest and Cypress
· Deploy applications using Docker and CI/CD pipelines
""",

    "☁️  DevOps / Cloud Engineer": """\
DevOps / Cloud Infrastructure Engineer — CloudOps Corp

ABOUT THE ROLE
We are scaling our cloud infrastructure and need a DevOps engineer
to own our CI/CD pipelines, container orchestration, and IaC.

REQUIRED SKILLS
Docker, Kubernetes, Terraform, AWS, Linux, Bash, Git,
GitHub Actions, Ansible, Helm, Prometheus, Grafana

NICE TO HAVE
GCP, Azure, Jenkins, Kafka, Python, Go, ELK Stack, Nginx

RESPONSIBILITIES
· Design and manage Kubernetes clusters on AWS EKS
· Write Terraform modules for infrastructure as code
· Build and maintain CI/CD pipelines with GitHub Actions
· Monitor systems with Prometheus and Grafana dashboards
· Automate server provisioning with Ansible
""",

    "🗄️  Data Engineer": """\
Data Engineer — Analytics Hub

ABOUT THE ROLE
We're building a modern data lakehouse and need a skilled Data
Engineer to own our ingestion, transformation, and delivery pipelines.

REQUIRED SKILLS
Python, SQL, Spark, Airflow, dbt, PostgreSQL, Snowflake,
Kafka, AWS, Docker, Git, BigQuery

NICE TO HAVE
Terraform, Kubernetes, Scala, Kafka, Redis, Looker, Tableau,
Great Expectations, Data Quality frameworks

RESPONSIBILITIES
· Design and maintain ELT pipelines using Airflow and dbt
· Build real-time streaming pipelines with Kafka and Spark
· Manage Snowflake data warehouse schemas and optimization
· Integrate data from 10+ external sources via REST APIs
· Write data quality checks and monitor pipeline health
""",

    "🎨  Frontend Developer (React)": """\
Senior Frontend Developer — UX Forward Studio

ABOUT THE ROLE
We are a product studio focused on beautiful, accessible interfaces.
We need a Senior Frontend Developer to lead our React component library.

REQUIRED SKILLS
JavaScript, TypeScript, React, HTML, CSS, Sass, Tailwind,
Git, Webpack, Figma, Jest, Accessibility (a11y)

NICE TO HAVE
Next.js, Vue, Storybook, Cypress, GraphQL, Node.js,
Animation libraries, Three.js, WebGL

RESPONSIBILITIES
· Build and maintain a shared React component library
· Translate Figma designs to pixel-perfect, accessible code
· Write unit and e2e tests with Jest and Cypress
· Collaborate with designers using Figma and Storybook
· Optimize bundle performance with Webpack and Vite
""",
}

PRESET_NAMES  = list(PRESET_JDS.keys())
CUSTOM_LABEL  = "✏️  Custom (type below)"

# ─────────────────────────────────────────────
#  JD SELECTOR HELPERS
# ─────────────────────────────────────────────
def load_preset_jd(preset_name: str) -> str:
    """Return JD text when a preset is chosen; empty string for Custom."""
    if preset_name == CUSTOM_LABEL or preset_name not in PRESET_JDS:
        return ""
    return PRESET_JDS[preset_name]

def save_custom_jd(name: str, text: str, existing: list):
    """Add a new custom JD to the shared dropdown list.

    Returns three values, one per Gradio output wired to this function:
    (dropdown update, new dropdown list for the shared State, status message).
    """
    # "or ''" guards against Gradio passing None for an untouched textbox
    name = (name or "").strip()
    if not name or not (text or "").strip():
        # gr.update() with no arguments means "leave the dropdown unchanged"
        return gr.update(), existing, "⚠️  Please enter both a name and JD text."
    label = f"📌  {name}"
    if label in existing:
        return gr.update(), existing, f"⚠️  '{name}' already exists — use a different name."
    # Stored in the module-level dict so load_preset_jd can find it later.
    # Note this dict is shared by every browser session using the app.
    PRESET_JDS[label] = text.strip()
    new_list = existing + [label]
    # Refresh the dropdown's options and select the newly saved entry
    return gr.update(choices=new_list, value=label), new_list, f"✅  '{name}' saved and selected!"

# ─────────────────────────────────────────────
#  CORE PIPELINE FUNCTIONS
# ─────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Normalise text so both models see consistent input."""
    # Collapse every run of spaces, tabs and newlines into a single space
    text = re.sub(r"\s+", " ", text)
    # Replace anything that isn't a letter, digit, whitespace or one of . , + # / -
    # with a space. Those symbols are kept because skill names need them:
    # "c++", "c#", "node.js", "ci/cd", "scikit-learn".
    text = re.sub(r"[^\w\s\.\,\+\#\/\-]", " ", text)
    return text.strip()

def _skill_pattern(skill: str) -> re.Pattern:
    # \b fails next to symbols ("c++," has no word boundary after the "++"),
    # so require explicitly that no word character, "+" or "#" is adjacent.
    return re.compile(rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])")

# Longest first, so "github actions" claims its text before "github" can,
# and "node.js" before "node" - otherwise one mention counts as two skills.
SKILL_PATTERNS = [(s, _skill_pattern(s)) for s in sorted(TECH_SKILLS, key=len, reverse=True)]

def extract_skills(text: str) -> set:
    """Find every TECH_SKILLS entry mentioned in the text.

    Two passes: a regex scan over the skill bank, then the NER model as a
    backup for anything the regex missed.
    """
    remaining = text.lower()
    found = set()

    # ── Pass 1: regex scan, longest skill names first ──
    for skill, pattern in SKILL_PATTERNS:
        # pattern.subn(replacement, text) replaces every match and returns a
        # pair: (the new text, how many matches were replaced).
        #
        # The replacement here is a lambda, i.e. a small unnamed function
        # written inline. It is exactly equivalent to:
        #
        #     def blank_out(m):
        #         return " " * len(m.group())
        #
        # subn calls it once per match, passing a Match object `m`.
        # m.group() is the matched text (e.g. "github actions"), and
        # " " * len(...) builds a string of that many spaces. So each match is
        # overwritten with blanks of the same length, which "uses up" that text:
        # when the shorter "github" is checked later, it can't match inside
        # "github actions" again. Keeping the length the same (rather than
        # deleting) stops neighbouring words from being glued together.
        remaining, hits = pattern.subn(lambda m: " " * len(m.group()), remaining)
        if hits:
            found.add(skill)

    # ── Pass 2: NER model as a safety net ──
    try:
        # BERT accepts at most 512 tokens, so feed the text in 512-character
        # slices. Stepping by 480 makes consecutive slices overlap by 32
        # characters, so a skill cut in half at one boundary still appears
        # whole in the next slice.
        for chunk in [text[i:i+512] for i in range(0, len(text), 480)]:
            for ent in NER_PIPE(chunk):
                w = ent["word"].lower().strip()
                # Tech names usually get tagged ORG ("Docker") or MISC ("Python").
                # Only accept words already in the skill bank, which filters out
                # company and product names that aren't skills.
                if ent["entity_group"] in ("ORG", "MISC") and len(w) >= 2 and w in TECH_SKILLS:
                    found.add(w)
    except Exception:
        # NER is only a backup; if it fails, the regex results still stand
        pass
    return found

def get_embedding(text: str) -> np.ndarray:
    """Encode text as a 384-dimensional meaning vector."""
    # encode() works on a batch, so wrap the text in a list and take the
    # first (only) row of the result
    return EMBED_MODEL.encode([text])[0]

def compute_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors: near 1.0 = same meaning, near 0 = unrelated."""
    # cosine_similarity compares every row of one matrix with every row of
    # another, so each vector goes in as a one-row matrix and the single
    # score comes out at position [0][0]
    return float(cosine_similarity([a], [b])[0][0])

def skill_gap(resume_skills: set, jd_skills: set) -> dict:
    """Compare the two skill sets using set operations."""
    matched  = resume_skills & jd_skills     # & = in both
    missing  = jd_skills - resume_skills     # - = wanted by the JD, not in the resume
    extra    = resume_skills - jd_skills     # in the resume, not asked for
    # Share of the JD's skills that the resume covers (guarding divide-by-zero)
    coverage = (len(matched) / len(jd_skills) * 100) if jd_skills else 0.0
    return {"matched": sorted(matched), "missing": sorted(missing),
            "extra": sorted(extra), "coverage": round(coverage, 1)}

# ─────────────────────────────────────────────
#  ANALYSIS — SINGLE RESUME
# ─────────────────────────────────────────────
def analyze_single(resume_text: str, jd_text: str):
    """Score one resume against one JD.

    Returns four Markdown strings, one for each output panel in the UI:
    (score, matched skills, missing skills, bonus skills).
    """
    # Every output needs a value, so the error case returns empty strings
    # for the other three panels
    if not (resume_text or "").strip() or not (jd_text or "").strip():
        return "⚠️  Please provide both a Resume and a Job Description.", "", "", ""

    # Same three steps for both documents: r_ = resume, j_ = job description
    r_clean  = clean_text(resume_text);  j_clean  = clean_text(jd_text)
    r_skills = extract_skills(r_clean);  j_skills = extract_skills(j_clean)
    r_vec    = get_embedding(r_clean);   j_vec    = get_embedding(j_clean)
    sim = compute_similarity(r_vec, j_vec)
    gap = skill_gap(r_skills, j_skills)
    pct = round(sim * 100, 1)

    # Chained conditional expression, read left to right: the first test
    # that passes picks the emoji
    emoji = "🔥" if pct >= 80 else "✅" if pct >= 60 else "⚠️" if pct >= 40 else "❌"
    score_md = f"""## {emoji}  Semantic Similarity Score

# `{pct}%`

| Level      | Range      |
|------------|------------|
| Excellent  | 80 – 100 % |
| Good       | 60 – 79 %  |
| Average    | 40 – 59 %  |
| Poor       | 0 – 39 %   |

**Skill Coverage:** `{gap['coverage']}%` of required JD skills found in resume
"""
    matched_md = "## ✅  Matched Skills\n\n"
    matched_md += (" · ".join(f"`{s}`" for s in gap["matched"])
                   if gap["matched"] else "_No matching skills detected._")

    missing_md = "## ❌  Missing Skills\n\n"
    if gap["missing"]:
        missing_md += " · ".join(f"`{s}`" for s in gap["missing"])
        missing_md += f"\n\n> **{len(gap['missing'])} skill(s) to acquire** for a stronger match."
    else:
        missing_md += "_No skill gaps — excellent match!_"

    extra_md = "## ➕  Bonus Skills  (in resume, not required)\n\n"
    extra_md += (" · ".join(f"`{s}`" for s in gap["extra"])
                 if gap["extra"] else "_No extra skills._")

    return score_md, matched_md, missing_md, extra_md

# ─────────────────────────────────────────────
#  ANALYSIS — MULTI RESUME RANKING
# ─────────────────────────────────────────────
def rank_resumes(resumes_text: str, jd_text: str):
    """Score several resumes against one JD and return a ranked Markdown table."""
    if not (resumes_text or "").strip() or not (jd_text or "").strip():
        return "⚠️  Please provide resumes and a job description."

    # The JD is the same for every candidate, so process it once up front
    j_clean  = clean_text(jd_text)
    j_vec    = get_embedding(j_clean)
    j_skills = extract_skills(j_clean)

    # Split the pasted text on each "=== RESUME ===" marker (any case, flexible
    # spacing). Text before the first marker comes out as an empty piece, which
    # the "if b.strip()" filter drops.
    blocks = [b.strip() for b in
              re.split(r"===\s*RESUME\s*===", resumes_text, flags=re.IGNORECASE)
              if b.strip()]
    if not blocks:
        return "⚠️  No resumes found. Separate each block with `=== RESUME ===`."

    results = []
    # enumerate(..., 1) numbers the candidates from 1 instead of 0
    for idx, block in enumerate(blocks, 1):
        # Use the first line as the candidate's name, unless it's too long to
        # plausibly be a name
        first_line = block.split("\n")[0].strip()
        name = first_line if len(first_line) < 60 else f"Candidate {idx}"
        r_clean  = clean_text(block)
        r_vec    = get_embedding(r_clean)
        r_skills = extract_skills(r_clean)
        sim = compute_similarity(r_vec, j_vec)
        gap = skill_gap(r_skills, j_skills)
        results.append({"rank": 0, "name": name, "sim": round(sim * 100, 1),
                         "coverage": gap["coverage"], "matched": len(gap["matched"]),
                         "missing": len(gap["missing"]), "gap_list": gap["missing"]})

    # sort(key=...) needs a function that turns each item into the value to
    # sort by. This lambda is an inline, unnamed version of:
    #
    #     def composite_score(x):
    #         return (x["sim"] + x["coverage"]) / 2
    #
    # For each candidate dict `x` it averages semantic similarity and skill
    # coverage (both percentages) into one score, so a candidate must do well
    # on both. reverse=True puts the highest score first.
    results.sort(key=lambda x: (x["sim"] + x["coverage"]) / 2, reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Medals for the top three; .get() falls back to "#4", "#5", ... for the rest
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    md  = "## 🏆  Resume Ranking\n\n"
    md += "| Rank | Candidate | Similarity | Skill Coverage | Matched | Missing |\n"
    md += "|------|-----------|:----------:|:--------------:|:-------:|:-------:|\n"
    for r in results:
        medal = medals.get(r["rank"], f"#{r['rank']}")
        md += (f"| {medal} | **{r['name']}** | `{r['sim']}%` | "
               f"`{r['coverage']}%` | {r['matched']} | {r['missing']} |\n")
    md += "\n---\n\n### 📋  Skill Gaps per Candidate\n\n"
    for r in results:
        md += f"**{r['name']}** — missing: "
        md += (", ".join(f"`{s}`" for s in r["gap_list"]) if r["gap_list"] else "_none_")
        md += "\n\n"
    return md

# ─────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg:      #08090f;
    --surface: #0f1119;
    --card:    #131825;
    --border:  #1c2236;
    --accent:  #00e5ff;
    --violet:  #7c3aed;
    --green:   #10b981;
    --amber:   #f59e0b;
    --red:     #ef4444;
    --text:    #dde4f0;
    --muted:   #566080;
}
*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* HEADER */
.rh-header {
    background: linear-gradient(160deg, #0a0c16 0%, #10131f 60%, #0a0e18 100%);
    border-bottom: 1px solid var(--border);
    padding: 2.2rem 1rem 1.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.rh-header::after {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 50% -10%,
                rgba(0,229,255,.07) 0%, transparent 70%);
    pointer-events: none;
}
.rh-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(1.6rem, 4vw, 2.4rem);
    font-weight: 700; color: var(--accent);
    letter-spacing: -.04em; margin: 0;
    text-shadow: 0 0 40px rgba(0,229,255,.35);
}
.rh-sub {
    font-size: .82rem; color: var(--muted);
    letter-spacing: .14em; text-transform: uppercase; margin-top: .35rem;
}

/* MODEL LEGEND */
.model-legend {
    display: flex; gap: 1rem; flex-wrap: wrap;
    justify-content: center; padding: 1rem 1rem .5rem;
}
.model-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px; padding: .7rem 1.1rem;
    min-width: 260px; max-width: 400px; flex: 1;
}
.model-card.ner { border-left-color: var(--violet); }
.model-name {
    font-family: 'Space Mono', monospace;
    font-size: .7rem; color: var(--accent);
    font-weight: 700; letter-spacing: .04em;
}
.model-card.ner .model-name { color: #a78bfa; }
.model-role {
    font-size: .8rem; font-weight: 600;
    color: var(--text); margin: .2rem 0 .18rem;
}
.model-reason { font-size: .74rem; color: var(--muted); line-height: 1.5; }
.model-reason strong { color: var(--text); }

/* JD label */
.jd-panel-title {
    font-family: 'Space Mono', monospace;
    font-size: .68rem; letter-spacing: .1em;
    color: var(--muted); text-transform: uppercase; margin-bottom: .4rem;
}

/* TABS */
.tab-nav { border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
    font-family: 'Space Mono', monospace !important;
    font-size: .75rem !important; letter-spacing: .05em !important;
    background: transparent !important; color: var(--muted) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important; padding: .65rem 1.3rem !important;
    transition: all .18s !important;
}
.tab-nav button.selected, .tab-nav button:hover {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
    background: rgba(0,229,255,.04) !important;
}

/* INPUTS */
textarea, input[type="text"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .88rem !important;
    transition: border-color .18s !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(0,229,255,.07) !important;
    outline: none !important;
}
label span {
    font-family: 'Space Mono', monospace !important;
    font-size: .68rem !important; letter-spacing: .09em !important;
    color: var(--muted) !important; text-transform: uppercase !important;
}

/* BUTTONS */
button.primary, .gr-button-primary {
    background: linear-gradient(130deg, var(--violet) 0%, var(--accent) 100%) !important;
    border: none !important; border-radius: 8px !important;
    color: #fff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .82rem !important; font-weight: 700 !important;
    letter-spacing: .04em !important; padding: .72rem 2rem !important;
    cursor: pointer !important; transition: all .22s !important;
    box-shadow: 0 4px 22px rgba(0,229,255,.14) !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(0,229,255,.25) !important;
}
button.secondary {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important; border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important; font-size: .72rem !important;
}

/* MARKDOWN OUTPUT */
.gr-markdown {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important; padding: 1.2rem !important;
    color: var(--text) !important;
}
.gr-markdown h2 {
    font-family: 'Space Mono', monospace !important;
    font-size: .78rem !important; letter-spacing: .12em !important;
    color: var(--accent) !important; text-transform: uppercase;
    border-bottom: 1px solid var(--border) !important;
    padding-bottom: .4rem !important; margin-bottom: .9rem !important;
}
.gr-markdown h1 {
    font-family: 'Space Mono', monospace !important;
    font-size: 2.6rem !important; color: var(--accent) !important;
    text-align: center; text-shadow: 0 0 30px rgba(0,229,255,.4);
}
.gr-markdown code {
    background: rgba(0,229,255,.07) !important;
    border: 1px solid rgba(0,229,255,.16) !important;
    border-radius: 4px !important; color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: .75rem !important; padding: .05rem .35rem !important;
}
.gr-markdown table { width: 100%; border-collapse: collapse; }
.gr-markdown th {
    font-family: 'Space Mono', monospace; font-size: .68rem;
    letter-spacing: .08em; color: var(--muted); text-transform: uppercase;
    border-bottom: 1px solid var(--border); padding: .4rem .6rem;
}
.gr-markdown td {
    font-size: .82rem; padding: .4rem .6rem;
    border-bottom: 1px solid rgba(255,255,255,.04);
}
.gr-markdown blockquote {
    border-left: 3px solid var(--amber) !important;
    background: rgba(245,158,11,.06) !important;
    border-radius: 0 6px 6px 0 !important;
    padding: .5rem .8rem !important; color: var(--amber) !important;
    font-size: .82rem !important;
}
hr { border-color: var(--border) !important; }

/* Score glow */
.score-card .gr-markdown {
    border-color: rgba(0,229,255,.3) !important;
    box-shadow: 0 0 24px rgba(0,229,255,.06) !important;
}

/* JD Library — full-height preformatted blocks */
.gr-markdown pre {
    background: rgba(0,229,255,.04) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
    overflow: visible !important;      /* never clip */
    max-height: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .85rem !important;
    line-height: 1.6 !important;
    color: var(--text) !important;
}
.gr-markdown pre code {
    background: none !important;
    border: none !important;
    padding: 0 !important;
    font-size: inherit !important;
    color: var(--text) !important;
}

/* Footer */
.rh-footer {
    text-align: center; color: var(--muted);
    font-family: 'Space Mono', monospace; font-size: .7rem;
    padding: 1.2rem; border-top: 1px solid var(--border);
    margin-top: 2rem; letter-spacing: .06em;
}
"""

# ─────────────────────────────────────────────
#  SAMPLE DATA
# ─────────────────────────────────────────────
SAMPLE_RESUME = """\
Alex Johnson  |  Senior Software Engineer

SKILLS
Python, FastAPI, Django, PostgreSQL, Redis, Docker, Kubernetes, AWS,
React, TypeScript, Git, GitHub Actions, Terraform, Prometheus, Grafana,
pandas, numpy, scikit-learn, MLflow

EXPERIENCE
Senior Backend Engineer — TechCorp  (2021–Present)
  · Built ML pipelines using Python, pandas, scikit-learn, MLflow
  · Deployed microservices on Kubernetes with Docker, CI/CD via GitHub Actions
  · Designed REST and GraphQL APIs with FastAPI
  · Managed PostgreSQL and Redis; Snowflake for analytics

EDUCATION
B.Sc. Computer Science — State University
"""

SAMPLE_MULTI = """\
=== RESUME ===
Alice Chen
Python, PyTorch, scikit-learn, FastAPI, Docker, Kubernetes, AWS, MLflow, Git, PostgreSQL, Airflow, pandas, numpy

=== RESUME ===
Bob Martinez
Java, Spring, Docker, Kubernetes, AWS, PostgreSQL, Git, React, TypeScript, GraphQL, Redis, CI/CD

=== RESUME ===
Carol White
Python, TensorFlow, Keras, pandas, numpy, scikit-learn, MLflow, AWS, Docker, FastAPI, Git, Spark, Kafka, dbt

=== RESUME ===
David Lee
Python, FastAPI, PostgreSQL, Docker, AWS, Git, pandas, scikit-learn, Airflow, dbt, Snowflake, SQL, Grafana
"""

# ─────────────────────────────────────────────
#  BUILD GRADIO APP
# ─────────────────────────────────────────────
def build_app():
    """Lay out the Gradio UI and connect its buttons to the pipeline functions."""

    all_jd_names = PRESET_NAMES + [CUSTOM_LABEL]

    # Everything created inside this "with" block becomes part of the page,
    # arranged by the nested Tabs / Row / Column blocks. The CSS is applied in
    # launch() below: Gradio 6 moved it off the Blocks constructor.
    with gr.Blocks(title="ResumeMatch · AI Matching System") as demo:

        # HEADER
        gr.HTML("""
        <div class="rh-header">
          <div class="rh-title">◈  RESUMEMATCH</div>
          <div class="rh-sub">AI-Powered Resume · Job Description Intelligence</div>
        </div>
        """)

        # MODEL LEGEND — name + WHY each model is used
        gr.HTML("""
        <div class="model-legend">

          <div class="model-card">
            <div class="model-name">🤗 sentence-transformers/all-MiniLM-L6-v2</div>
            <div class="model-role">Semantic Embedding Model</div>
            <div class="model-reason">
              Converts resume and job-description text into compact
              <strong>384-dimensional vectors</strong>. We then compute
              <strong>cosine similarity</strong> between those vectors — so two
              documents that mean the same thing score high even when they use
              completely different words, far beyond plain keyword matching.
            </div>
          </div>

          <div class="model-card ner">
            <div class="model-name">🤗 dslim/bert-base-NER</div>
            <div class="model-role">Named Entity Recognition Model</div>
            <div class="model-reason">
              A BERT model fine-tuned to label tokens as
              <strong>PER · ORG · LOC · MISC</strong>.
              We harvest <strong>ORG / MISC</strong> entities and cross-reference
              them against our skill bank to surface technologies and frameworks
              that raw keyword scanning would miss — zero manual tagging required.
            </div>
          </div>

        </div>
        """)

        # Shared state: live list of JD dropdown choices
        # gr.State is an invisible value that lives for one browser session and
        # can be passed into and out of event handlers like any component.
        # It holds the current dropdown choices so saved custom JDs persist.
        jd_choices_state = gr.State(value=all_jd_names)

        with gr.Tabs():

            # ═══════════════════════════════════════════════
            #  TAB 1 — Single Resume Analysis
            # ═══════════════════════════════════════════════
            with gr.Tab("🎯  Single Resume Analysis"):

                with gr.Row():
                    # LEFT — resume
                    with gr.Column(scale=1):
                        resume_in = gr.Textbox(
                            label="📄  Resume Text",
                            placeholder="Paste the full resume here …",
                            lines=20, max_lines=30,
                        )
                        gr.Examples(examples=[[SAMPLE_RESUME]], inputs=[resume_in],
                                    label="▶ Load Sample Resume")

                    # RIGHT — JD panel
                    with gr.Column(scale=1):
                        gr.HTML('<div class="jd-panel-title">💼  Job Description</div>')

                        jd_preset_dd = gr.Dropdown(
                            choices=all_jd_names,
                            value=PRESET_NAMES[0],
                            label="Select Preset JD  — or choose ✏️ Custom to type your own",
                            interactive=True,
                        )
                        jd_in = gr.Textbox(
                            label="JD Text  (auto-filled from preset or edit freely)",
                            value=PRESET_JDS[PRESET_NAMES[0]],
                            lines=14, max_lines=22,
                        )

                        with gr.Accordion("📌  Save this JD as a new preset", open=False):
                            custom_name_1  = gr.Textbox(
                                label="Preset Name",
                                placeholder="e.g.  Backend Engineer @ Stripe",
                                lines=1,
                            )
                            save_btn_1    = gr.Button("💾  Add to Preset List", variant="secondary")
                            save_status_1 = gr.Markdown("")

                analyze_btn = gr.Button("⚡  Analyze Match", variant="primary")

                with gr.Row():
                    with gr.Column(elem_classes="score-card"):
                        score_out = gr.Markdown(label="Similarity Score")
                    matched_out = gr.Markdown(label="Matched Skills")
                with gr.Row():
                    missing_out = gr.Markdown(label="Missing Skills")
                    extra_out   = gr.Markdown(label="Bonus Skills")

                # ── events
                # Each line wires a trigger to a function: when it fires, Gradio
                # reads the current values of the `inputs` components, passes
                # them to `fn` as arguments in that order, and writes whatever
                # `fn` returns into the `outputs` components, again in order.
                # Picking a preset fills the JD textbox with that preset's text
                jd_preset_dd.change(fn=load_preset_jd,
                                    inputs=[jd_preset_dd], outputs=[jd_in])
                save_btn_1.click(fn=save_custom_jd,
                                 inputs=[custom_name_1, jd_in, jd_choices_state],
                                 outputs=[jd_preset_dd, jd_choices_state, save_status_1])
                analyze_btn.click(fn=analyze_single,
                                  inputs=[resume_in, jd_in],
                                  outputs=[score_out, matched_out, missing_out, extra_out])

            # ═══════════════════════════════════════════════
            #  TAB 2 — Rank Multiple Resumes
            # ═══════════════════════════════════════════════
            with gr.Tab("🏆  Rank Multiple Resumes"):

                gr.Markdown(
                    "> Separate each resume block with `=== RESUME ===`. "
                    "Candidates are ranked by composite of semantic similarity + skill coverage.",
                    elem_classes="gr-markdown"
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        multi_in = gr.Textbox(
                            label="📋  Multiple Resumes  (delimited by  === RESUME ===)",
                            placeholder="=== RESUME ===\nAlice …\n\n=== RESUME ===\nBob …",
                            lines=20,
                        )
                        gr.Examples(examples=[[SAMPLE_MULTI]], inputs=[multi_in],
                                    label="▶ Load 4 Sample Candidates")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="jd-panel-title">💼  Job Description</div>')

                        jd_preset_dd2 = gr.Dropdown(
                            choices=all_jd_names,
                            value=PRESET_NAMES[0],
                            label="Select Preset JD  — or choose ✏️ Custom",
                            interactive=True,
                        )
                        jd_rank_in = gr.Textbox(
                            label="JD Text",
                            value=PRESET_JDS[PRESET_NAMES[0]],
                            lines=14, max_lines=22,
                        )

                        with gr.Accordion("📌  Save this JD as a new preset", open=False):
                            custom_name_2  = gr.Textbox(
                                label="Preset Name",
                                placeholder="e.g.  Data Scientist @ OpenAI",
                                lines=1,
                            )
                            save_btn_2    = gr.Button("💾  Add to Preset List", variant="secondary")
                            save_status_2 = gr.Markdown("")

                rank_btn    = gr.Button("🏁  Rank Candidates", variant="primary")
                ranking_out = gr.Markdown(label="Ranking Results")

                # ── events  (same wiring as tab 1, see the comment there)
                jd_preset_dd2.change(fn=load_preset_jd,
                                     inputs=[jd_preset_dd2], outputs=[jd_rank_in])
                save_btn_2.click(fn=save_custom_jd,
                                 inputs=[custom_name_2, jd_rank_in, jd_choices_state],
                                 outputs=[jd_preset_dd2, jd_choices_state, save_status_2])
                rank_btn.click(fn=rank_resumes,
                               inputs=[multi_in, jd_rank_in],
                               outputs=[ranking_out])

            # ═══════════════════════════════════════════════
            #  TAB 3 — JD Library (browse all presets)
            # ═══════════════════════════════════════════════
            with gr.Tab("📚  JD Library"):

                gr.HTML("""
                <div class="jd-panel-title" style="padding:.9rem 0 .5rem">
                  5 Built-in Job Descriptions — click any to expand
                </div>
                """)

                # Components can be created in a loop: one collapsible panel
                # per preset. This runs once at build time, so JDs saved
                # later via "Add to Preset List" won't appear in this tab.
                for jd_name, jd_body in PRESET_JDS.items():
                    with gr.Accordion(jd_name, open=False):
                        # Render as fenced code block inside Markdown so the
                        # full text is always visible — no fixed-height clipping.
                        gr.Markdown(f"```\n{jd_body}\n```")

            # ═══════════════════════════════════════════════
            #  TAB 4 — How It Works
            # ═══════════════════════════════════════════════
            with gr.Tab("📖  How It Works"):
                gr.Markdown("""
## Pipeline Architecture

### Step 1 — Text Cleaning
Whitespace collapsed, special characters stripped, text normalised so both
models receive consistent, clean input.

### Step 2 — Skill Extraction  (`dslim/bert-base-NER`)
The NER model labels every token as **PER / ORG / LOC / MISC**.
We harvest **ORG** and **MISC** tags and match them against a curated bank
of 100+ tech skills. A parallel regex scan covers anything NER might miss.

### Step 3 — Semantic Embeddings  (`all-MiniLM-L6-v2`)
Each document is encoded into a **384-dimensional float vector**.
Trained on 1 billion+ sentence pairs, this model understands that
"deploy containers" and "Kubernetes orchestration" are semantically related.

### Step 4 — Cosine Similarity
```
score = cosine(resume_vec, jd_vec)   →   0.0 … 1.0
```
Two semantically equivalent texts score near 1.0, even with different wording.

### Step 5 — Skill Gap Analysis

| Set             | Meaning                          |
|-----------------|----------------------------------|
| `resume ∩ jd`   | Matched (shared) skills          |
| `jd − resume`   | Missing skills — gaps to close   |
| `resume − jd`   | Bonus skills not asked for       |
| `\\|matched\\| / \\|jd\\|` | Skill coverage percentage |

### Step 6 — Ranking  (multi-resume tab)
Composite score = `(similarity % + coverage %) / 2`.
All resumes are sorted descending; top 3 receive 🥇 🥈 🥉.

---

## Score Interpretation

| Range     | Verdict                      |
|-----------|------------------------------|
| 80–100 %  | 🔥 Excellent — strong match  |
| 60–79 %   | ✅ Good — worth interviewing  |
| 40–59 %   | ⚠️ Average — notable gaps    |
| 0–39 %    | ❌ Poor — major misalignment  |
                """)

        # FOOTER
        gr.HTML("""
        <div class="rh-footer">
          RESUMEMATCH &nbsp;·&nbsp; HuggingFace Transformers + Gradio
          &nbsp;·&nbsp; all-MiniLM-L6-v2 &amp; dslim/bert-base-NER
        </div>
        """)

    return demo


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app = build_app()
    app.launch(
        css=CUSTOM_CSS,
        server_name="127.0.0.1",   # local machine only
        server_port=7860,          # open http://127.0.0.1:7860 in a browser
        share=False,   # set True for a public Gradio URL
#        show_api=False,
    )