# ============================================================
# indexer.py — One-time script that embeds the question bank
#              and stores vectors in ChromaDB on disk.
#
# Run once before using the retriever:
#   python -m module4_interview_generator.indexer
#
# After this runs, a folder called chroma_db/ appears inside
# module4_interview_generator/. The retriever loads from there.
# Re-running this script is safe — it clears and rebuilds.
# ============================================================

import os
import sys

import chromadb
from sentence_transformers import SentenceTransformer

from .question_bank import QUESTION_BANK, get_all_ids, get_all_texts, get_all_metadatas

# ── Paths ─────────────────────────────────────────────────────
_MODULE_DIR = os.path.dirname(__file__)
CHROMA_PATH = os.path.join(_MODULE_DIR, "chroma_db")
COLLECTION_NAME = "interview_questions"

# ── Embedding model (downloads ~80 MB on first use) ──────────
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def build_index(force: bool = False) -> None:
    """
    Embed all questions in QUESTION_BANK and store them in ChromaDB.

    Args:
        force: If True, drops and rebuilds the collection even if it
               already has documents. Default False (skip if already built).
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Check if already indexed
    existing = client.get_or_create_collection(COLLECTION_NAME)
    count = existing.count()
    if count > 0 and not force:
        print(f"[indexer] Index already built ({count} questions). Skipping.")
        print(f"          Pass force=True or run with --force to rebuild.")
        return

    # Delete and recreate for a clean rebuild
    client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    print(f"[indexer] Loading embedding model '{EMBED_MODEL_NAME}'...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = get_all_texts()
    ids = get_all_ids()
    metadatas = get_all_metadatas()

    print(f"[indexer] Embedding {len(texts)} questions...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"[indexer] Storing in ChromaDB at {CHROMA_PATH} ...")
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[indexer] Done. {collection.count()} questions indexed.")


def get_collection():
    """
    Returns the live ChromaDB collection. Called by retriever.py.
    Raises RuntimeError if the index has not been built yet.
    """
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        raise RuntimeError(
            "ChromaDB index is empty. Run `python -m module4_interview_generator.indexer` first."
        )
    return collection


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    force_rebuild = "--force" in sys.argv
    build_index(force=force_rebuild)
