# SignPath documentation RAG assistant
#
# Build and run (from the week5 folder; the .env file is in the repo root):
#   docker compose --env-file ../../.env up -d --build
# then open http://localhost:7860
#
# Or run the published image directly, with your own key and (optionally) chat model:
#   docker run -d -p 7860:7860 -e OPENAI_API_KEY=sk-... -e SIGNPATH_LLM_MODEL=gpt-4.1-mini \
#     -v signpath-data:/data javed132/signpath-rag
# To answer with Claude or Gemini, add -e SIGNPATH_LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY=...
# (or gemini + GOOGLE_API_KEY). OPENAI_API_KEY is still needed: the search's embeddings come from OpenAI.
#
# The crawled docs, images and vector index live in the /data volume, so they survive restarts
# and upgrades. The first start crawls and indexes the docs (about a minute).

FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/huggingface \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app

# Fonts for rendering the docs' SVG diagrams to PNG (Carlito is metric-compatible with Calibri, which they use)
RUN apt-get update \
 && apt-get install -y --no-install-recommends fonts-crosextra-carlito fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch keeps the image small; the re-ranker is a small model that runs fine on CPU
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
COPY requirements-signpath.txt .
RUN pip install -r requirements-signpath.txt

# Bake the re-ranker model and tokenizer data into the image, so the container needs no downloads at startup
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# The logo is optional: it isn't in the public git repo (see .gitignore), so put hexagon_logo.svg here before
# building internal images. The [g] wildcard lets the build succeed without it; the header then has no logo.
COPY Signpath_RAG.py signpath_retrieval_tests.json hexagon_logo.sv[g] ./

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

# The Evaluation tab is for developers. One build, two tags:
#   docker build -t signpath-rag:latest .                                # users: tab hidden
#   docker build -t signpath-rag:dev --build-arg SHOW_EVALUATION=1 .     # developers: tab shown
# (only this setting differs; either can be overridden with -e SIGNPATH_SHOW_EVALUATION=0/1)
ARG SHOW_EVALUATION=0
ENV SIGNPATH_SHOW_EVALUATION=${SHOW_EVALUATION}

ENV SIGNPATH_DATA_DIR=/data \
    SIGNPATH_OPEN_BROWSER=0 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HUB_OFFLINE=1

VOLUME /data
EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=10s --start-period=300s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/', timeout=5)" || exit 1

CMD ["python", "Signpath_RAG.py"]
