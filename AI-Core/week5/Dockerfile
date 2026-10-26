# SignPath documentation RAG assistant
#
# Build and run (from the week5 folder):
#   docker compose --env-file ../.env up -d --build
# then open http://localhost:7860
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

COPY Signpath_RAG.py signpath_retrieval_tests.json ./

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app

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
