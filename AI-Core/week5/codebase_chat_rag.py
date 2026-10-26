import os
import shutil
import stat
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import gradio as gr

# This tool uses RAG to let you ask questions about a repo in natural language:
# "How does authentication work?"
# "Where are the API routes defined?"
# "What database models exist?"

load_dotenv(override=True)

openai_api_key = os.getenv("OPENAI_API_KEY")

# Fall back to Google Colab secrets if not found in .env
if not openai_api_key:
    try:
        from google.colab import userdata
        openai_api_key = userdata.get("OPENAI_API_KEY")
    except Exception:
        pass

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
else:
    print("OpenAI API Key not set — set it in .env or Colab Secrets")

# --- Configuration ---

CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".rb",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".swift", ".kt",
    ".md", ".txt", ".yaml", ".yml", ".toml", ".json", ".html", ".css",
}

SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".tox", ".mypy_cache", ".pytest_cache", "coverage", ".idea",
}

MAX_FILE_SIZE = 50_000  # Skip files larger than 50KB

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def make_writable_and_retry(function, path, error):
    """git marks its object files read-only, which makes deleting them fail on Windows."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def remove_folder(path: str):
    shutil.rmtree(path, onexc=make_writable_and_retry)


def clone_repo(url: str) -> str:
    """Shallow-clone a GitHub repo into a temp directory."""
    repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")
    target = os.path.join(tempfile.gettempdir(), f"rag_{repo_name}")
    if os.path.exists(target):
        remove_folder(target)
    result = subprocess.run(
        ["git", "clone", "--depth", "1", url, target],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")
    return target


def load_code_files(repo_path: str) -> list[Document]:
    """Walk the repo and load code files as LangChain Documents."""
    documents = []
    repo_root = Path(repo_path)
    for filepath in repo_root.rglob("*"):
        if not filepath.is_file():
            continue
        # Skip excluded directories
        if any(skip in filepath.parts for skip in SKIP_DIRS):
            continue
        # Skip non-code files
        if filepath.suffix.lower() not in CODE_EXTENSIONS:
            continue
        # Skip large files
        if filepath.stat().st_size > MAX_FILE_SIZE:
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            rel_path = str(filepath.relative_to(repo_root))
            documents.append(Document(
                page_content=content,
                metadata={"source": rel_path, "extension": filepath.suffix},
            ))
        except Exception:
            continue
    return documents


print("Repo loading functions ready")

# --- Chunking, Embedding & RAG ---

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
llm = ChatOpenAI(temperature=0, model_name=LLM_MODEL)

# Global state for the loaded repo
vectorstore = None
retriever = None
loaded_repo_name = None

SYSTEM_PROMPT = """You are a senior software engineer helping a developer understand a codebase.
Use the retrieved code snippets below to answer the question.
Always reference the file path when discussing specific code.
If the context doesn't contain enough information, say so honestly.

Retrieved code snippets:
{context}"""


def load_repo(url: str) -> str:
    """Clone repo, chunk files, embed, and build vector store."""
    global vectorstore, retriever, loaded_repo_name

    if not url.strip():
        return "Please enter a GitHub repo URL."

    try:
        repo_path = clone_repo(url)
        repo_name = url.rstrip("/").split("/")[-1].replace(".git", "")

        documents = load_code_files(repo_path)
        if not documents:
            return "No code files found in this repo."

        chunks = text_splitter.split_documents(documents)

        # Build Chroma vectorstore in memory; drop the previous repo's chunks so answers don't mix repos
        if vectorstore is not None:
            vectorstore.delete_collection()
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
        loaded_repo_name = repo_name

        # Clean up cloned files
        remove_folder(repo_path)

        return f"Loaded {repo_name}: {len(documents)} files, {len(chunks)} chunks. Ready to chat!"

    except Exception as e:
        return f"Error: {e}"


def chat(message: str, history: list) -> str:
    """Answer a question about the loaded repo using RAG."""
    if retriever is None:
        return "Please load a repo first using the URL input above."

    docs = retriever.invoke(message)
    context = "\n\n".join(
        f"--- {doc.metadata['source']} ---\n{doc.page_content}" for doc in docs
    )

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
        HumanMessage(content=message),
    ])
    return response.content


print("RAG pipeline ready")

with gr.Blocks(title="GitHub Repo RAG") as ui:
    gr.Markdown("# GitHub Repo RAG")
    gr.Markdown(
        "Paste a public GitHub repo URL, load it, then ask questions about the code."
    )

    with gr.Row():
        url_input = gr.Textbox(
            label="GitHub Repo URL",
            placeholder="https://github.com/user/repo",
            scale=4,
        )
        load_btn = gr.Button("Load Repo", variant="primary", scale=1)

    status = gr.Textbox(label="Status", interactive=False)

    load_btn.click(fn=load_repo, inputs=[url_input], outputs=[status])

    gr.Markdown("---")

    gr.ChatInterface(
        fn=chat,
        examples=[
            "What does this project do?",
            "What are the main files and their purpose?",
            "How is the code structured?",
            "Are there any API routes? Where are they defined?",
        ],
    )

ui.launch(inbrowser=True)