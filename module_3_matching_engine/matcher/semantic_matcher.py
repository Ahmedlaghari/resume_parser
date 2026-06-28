"""
semantic_matcher.py
--------------------
Two jobs:
1. Own the sentence-transformers model (loaded once, reused everywhere)
   so skills_matcher.py can also use it for fuzzy skill matching.
2. Compute the overall "resume text vs JD text" similarity score
   (the semantic_weight piece of the final formula).

We use a small, fast model (all-MiniLM-L6-v2) since this only needs to
compare short-to-medium text, not do heavy NLP.
"""

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=1)
def get_embedder():
    """
    Loads the sentence-transformers model exactly once per process
    (lru_cache memoizes it). Importing sentence_transformers is done
    lazily inside this function so the rest of the module can be
    imported/tested without the (large) dependency installed.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Standard cosine similarity, clipped to [0, 1] for our scoring scale."""
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    sim = float(np.dot(vec_a, vec_b) / denom)
    # Cosine similarity is technically [-1, 1]; for short skill/text
    # phrases negative values essentially never happen, but clip
    # defensively so the score never goes below 0.
    return max(0.0, min(1.0, sim))


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embeds a batch of strings at once (cheaper than one-by-one calls)."""
    model = get_embedder()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def semantic_similarity_score(candidate_summary: str, jd_text: str) -> float:
    """
    Returns a 0-100 score for how semantically close the candidate's
    summary/experience text is to the JD's responsibilities +
    qualifications text. This is the "overall fit beyond keywords" signal.
    """
    candidate_summary = (candidate_summary or "").strip()
    jd_text = (jd_text or "").strip()

    if not candidate_summary or not jd_text:
        # Nothing to compare -- don't penalize, don't reward. Treat as neutral.
        return 50.0

    embeddings = embed_texts([candidate_summary, jd_text])
    similarity = cosine_similarity(embeddings[0], embeddings[1])
    return round(similarity * 100, 2)
