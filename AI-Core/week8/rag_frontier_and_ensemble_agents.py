"""Week 8, Day 2: RAG pricing with a vector database, then the Frontier and Ensemble agents.

Earlier weeks showed that frontier LLMs price products well out of the box, and that a
fine-tuned open-source model can beat them. Today tries an inference-time technique
instead of training: RAG (Retrieval Augmented Generation). Before asking GPT-5.1 for a
price, we look up the 5 most similar products in a vector database and put them, with
their prices, into the prompt.

Steps:
    1. Load the 800k (or 20k lite) summarised products from the HuggingFace Hub.
    2. Embed the training products with all-MiniLM-L6-v2 and store them in Chroma
       (skipped if the database already exists - the full build takes ~30 min on a GPU).
    3. Visualise a sample of the vectors in 2D and 3D with t-SNE, coloured by category.
    4. Show the RAG context for one test item, then evaluate GPT-5.1 with RAG.
    5. Evaluate an ensemble of RAG GPT-5.1 (80%), the fine-tuned Specialist (10%) and
       the Week 6 deep neural network (10%).
    6. Price one product with the Frontier, Neural Network and Ensemble agents.

Needs: HF_TOKEN and OPENAI_API_KEY in .env; the pricer-service deployed on Modal (Day 1);
deep_neural_network.pth from Week 6 in this folder, downloadable from
https://drive.google.com/drive/folders/1uq5C9edPIZ1973dArZiEO-VE13F7m8MK
"""

import logging
import os

import chromadb
import numpy as np
import plotly.graph_objects as go
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from agents.evaluator import evaluate
from agents.frontier_agent import FrontierAgent
from agents.items import Item
from agents.neural_network_agent import NeuralNetworkAgent
from agents.specialist_agent import SpecialistAgent


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

LITE_MODE = False                           # True: 20k products; False: 800k

HF_USERNAME = "ed-donner"
DATASET = f"{HF_USERNAME}/items_lite" if LITE_MODE else f"{HF_USERNAME}/items_full"

DB = "products_vectorstore"                 # Chroma database folder, shared by days 2, 4 and 5
COLLECTION = "products"
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"     # free, local, 384-dimensional vectors
BATCH_SIZE = 1000                           # products embedded and stored per batch

# Plotting all 800k points is fun but can crash the machine; 10k is safe
MAXIMUM_DATAPOINTS = 10_000
CATEGORIES = ["Appliances", "Automotive", "Cell_Phones_and_Accessories", "Electronics",
              "Musical_Instruments", "Office_Products", "Tools_and_Home_Improvement", "Toys_and_Games"]
COLORS = ["cyan", "blue", "brown", "orange", "yellow", "green", "purple", "red"]

# Ensemble weights: (RAG frontier, specialist, neural network)
ENSEMBLE_WEIGHTS = (0.8, 0.1, 0.1)

PRODUCT = "Shure MV7+ professional podcaster microphone with usb-c and XLR outputs"


# ---------------------------------------------------------------------------
# Data and vector store
# ---------------------------------------------------------------------------

def load_items():
    """Log in to HuggingFace and load the summarised train / validation / test splits."""
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    train, val, test = Item.from_hub(DATASET)
    print(f"Loaded {len(train):,} training items, {len(val):,} validation items, {len(test):,} test items")
    return train, val, test


def build_vectorstore(train):
    """Open the Chroma collection of products, creating and filling it on first run.

    Each training product's summary is embedded with ENCODER and stored with its
    category and price as metadata, under the id doc_<index>. Delete the DB folder to
    rebuild it (e.g. after switching LITE_MODE).

    Returns:
        The Chroma collection.
    """
    client = chromadb.PersistentClient(path=DB)
    if COLLECTION not in [collection.name for collection in client.list_collections()]:
        encoder = SentenceTransformer(ENCODER)
        collection = client.create_collection(COLLECTION)
        for start in tqdm(range(0, len(train), BATCH_SIZE)):
            batch = train[start:start + BATCH_SIZE]
            documents = [item.summary for item in batch]
            collection.add(
                ids=[f"doc_{start + offset}" for offset in range(len(batch))],
                documents=documents,
                embeddings=encoder.encode(documents).astype(float).tolist(),
                metadatas=[{"category": item.category, "price": item.price} for item in batch],
            )
    return client.get_or_create_collection(COLLECTION)


def plot_vectors(collection, dimensions):
    """Scatter plot of MAXIMUM_DATAPOINTS stored vectors, reduced to 2 or 3 dimensions.

    t-SNE (t-distributed Stochastic Neighbor Embedding) squeezes the 384-dimensional
    vectors down while keeping similar products close together, so products of the
    same category should form visible clusters. Hover a point to see its text.

    Args:
        collection: The Chroma collection.
        dimensions: 2 or 3.
    """
    result = collection.get(include=["embeddings", "documents", "metadatas"], limit=MAXIMUM_DATAPOINTS)
    vectors = np.array(result["embeddings"])
    documents = result["documents"]
    categories = [metadata["category"] for metadata in result["metadatas"]]
    colors = [COLORS[CATEGORIES.index(category)] for category in categories]

    reduced = TSNE(n_components=dimensions, random_state=42).fit_transform(vectors)
    hover = [f"Category: {category}<br>Text: {document[:50]}..." for category, document in zip(categories, documents)]

    if dimensions == 2:
        trace = go.Scatter(x=reduced[:, 0], y=reduced[:, 1], mode="markers",
                           marker=dict(size=4, color=colors, opacity=0.7), text=hover, hoverinfo="text")
        scene = dict(xaxis_title="x", yaxis_title="y")
    else:
        trace = go.Scatter3d(x=reduced[:, 0], y=reduced[:, 1], z=reduced[:, 2], mode="markers",
                             marker=dict(size=2, color=colors, opacity=0.7), text=hover, hoverinfo="text")
        scene = dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")

    fig = go.Figure(data=[trace])
    fig.update_layout(title=f"{dimensions}D Chroma Vectorstore Visualization", scene=scene,
                      width=1200, height=800, margin=dict(r=20, b=10, l=10, t=40))
    fig.show()


# ---------------------------------------------------------------------------
# Pricers for the evaluator
# ---------------------------------------------------------------------------

def show_rag_prompt(frontier, item):
    """Print the RAG prompt for one item: its summary plus 5 similar products and their prices."""
    documents, prices = frontier.find_similars(item.summary)
    print(frontier.messages_for(item.summary, documents, prices)[0]["content"])
    print(f"Actual price: ${item.price:,.2f}")


def make_rag_pricer(frontier):
    """Build the RAG pricer: GPT-5.1 with 5 similar products as context (via FrontierAgent).

    The function name becomes the chart title: "GPT 5.1 Rag".
    """
    def gpt_5__1_rag(item):
        """Price the item with GPT-5.1, given similar products from the vector store."""
        return frontier.price(item.summary)

    return gpt_5__1_rag


def make_ensemble_pricer(frontier, specialist, neural_network):
    """Build a pricer that blends three models with ENSEMBLE_WEIGHTS.

    The frontier model gets most of the weight because it is the most accurate; the
    other two pull its estimate slightly towards their own.
    """
    frontier_weight, specialist_weight, network_weight = ENSEMBLE_WEIGHTS

    def ensemble(item):
        """Weighted average of the RAG, fine-tuned and neural-network prices."""
        return (frontier.price(item.summary) * frontier_weight
                + specialist.price(item.summary) * specialist_weight
                + neural_network.price(item.summary) * network_weight)

    return ensemble


def price_with_agents(collection, description):
    """Price one product with each pricing agent in turn and print the results.

    The Ensemble Agent also rewrites the description with the Preprocessor first, so
    its inputs differ slightly from the other two agents'.
    """
    from agents.ensemble_agent import EnsembleAgent

    for agent in (FrontierAgent(collection), NeuralNetworkAgent(), EnsembleAgent(collection)):
        print(f"{agent.name}: ${agent.price(description):,.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Build the vector store, visualise it, and evaluate RAG and the ensemble."""
    load_dotenv(override=True)

    train, _, test = load_items()
    collection = build_vectorstore(train)
    plot_vectors(collection, dimensions=2)
    plot_vectors(collection, dimensions=3)

    frontier = FrontierAgent(collection)
    show_rag_prompt(frontier, test[0])
    evaluate(make_rag_pricer(frontier), test)

    specialist = SpecialistAgent()
    neural_network = NeuralNetworkAgent()
    evaluate(make_ensemble_pricer(frontier, specialist, neural_network), test)

    # Show the agents' log lines only now; during the evaluations they'd flood the output
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    price_with_agents(collection, PRODUCT)


if __name__ == "__main__":
    main()
