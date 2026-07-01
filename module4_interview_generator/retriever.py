# ============================================================
# retriever.py — Multi-query retrieval with category routing.
#
# Instead of one big query, we fire four targeted queries —
# one per category — each filtered to only search within that
# category's questions. Results are merged and deduplicated.
#
# This is the RAG retrieval step. No LLM here — pure vector
# similarity search using sentence-transformers + ChromaDB.
# ============================================================

from sentence_transformers import SentenceTransformer
from .indexer import get_collection

_embed_model = None  # lazy-loaded on first call


def _get_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text).tolist()


# ============================================================
# STEP 1 — QUERY TRANSLATION
# Convert the structured candidate/JD/match data into four
# plain-English search strings, one per category.
# ============================================================

def _technical_query(candidate: dict, match_result: dict) -> str:
    matched = match_result.get("matched_skills", [])[:6]
    skills = ", ".join(matched) if matched else "software engineering and programming"
    exp_years = candidate.get("total_experience_years", "")
    level = f"{exp_years} years experience" if exp_years else ""
    return f"Technical depth interview questions for {skills} {level}".strip()


def _experience_query(candidate: dict, jd: dict) -> str:
    role = jd.get("job_title", "the role")
    companies = [
        e.get("company", "")
        for e in candidate.get("experience", [])
        if e.get("company")
    ][:2]
    projects = [
        p.get("name", "")
        for p in candidate.get("projects", [])
        if p.get("name")
    ][:2]
    company_str = " and ".join(filter(None, companies)) or "previous companies"
    project_str = " and ".join(filter(None, projects))
    base = f"Experience probe questions about work history and projects for {role} at {company_str}"
    if project_str:
        base += f", including projects: {project_str}"
    return base


def _gap_query(match_result: dict) -> str:
    missing = match_result.get("missing_skills", [])[:6]
    if not missing:
        return "questions about skill gaps and areas for professional development"
    return f"Gap questions probing missing skills: {', '.join(missing)}"


def _behavioral_query(jd: dict) -> str:
    role = jd.get("job_title", "software engineering")
    responsibilities = jd.get("responsibilities", [])[:2]
    resp_str = "; ".join(responsibilities) if responsibilities else ""
    base = f"Behavioral and situational interview questions for {role}"
    if resp_str:
        base += f" involving {resp_str}"
    return base


# ============================================================
# STEP 2 — ROUTING + RETRIEVAL
# Each query is routed to its own category bucket inside
# ChromaDB via the `where` filter. This means a gap query
# only competes against gap questions — much more precise.
# ============================================================

def _query_category(
    collection,
    query_text: str,
    category: str,
    n: int,
) -> list[dict]:
    """Run one vector search against a single category."""
    embedding = _embed(query_text)
    results = collection.query(
        query_embeddings=[embedding],
        where={"category": category},
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        meta = results["metadatas"][0][i]
        hits.append({
            "id": doc_id,
            "text": results["documents"][0][i],
            "category": meta["category"],
            "skill_tags": meta.get("skill_tags", ""),
            "experience_level": meta.get("experience_level", "any"),
            "what_to_listen_for": meta.get("what_to_listen_for", ""),
            "distance": results["distances"][0][i],
        })
    return hits


# ============================================================
# STEP 3 — MERGE + DEDUPLICATE
# Combine results from all four queries, drop duplicates by id.
# ============================================================

def retrieve(
    candidate: dict,
    jd: dict,
    match_result: dict,
    n_per_category: int = 6,
) -> list[dict]:
    """
    Run four category-routed queries and return a merged, deduplicated
    pool of the most relevant questions.

    Args:
        candidate:      Module 1 resume JSON (dict)
        jd:             Module 2 JD JSON (dict)
        match_result:   Module 3 result for this candidate (dict)
        n_per_category: How many questions to retrieve per category.

    Returns:
        List of question dicts ready to be personalized by the LLM.
        Typically 4 × n_per_category items before deduplication.
    """
    collection = get_collection()

    queries = {
        "technical": _technical_query(candidate, match_result),
        "experience": _experience_query(candidate, jd),
        "gap": _gap_query(match_result),
        "behavioral": _behavioral_query(jd),
    }

    seen_ids: set[str] = set()
    pool: list[dict] = []

    for category, query_text in queries.items():
        hits = _query_category(collection, query_text, category, n_per_category)
        for hit in hits:
            if hit["id"] not in seen_ids:
                seen_ids.add(hit["id"])
                pool.append(hit)

    return pool


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    import json

    sample_candidate = {
        "name": "Ali Khan",
        "total_experience_years": 4,
        "skills": ["Python", "TensorFlow", "NLP", "FastAPI", "PostgreSQL"],
        "experience": [
            {"company": "ABC Corp", "title": "ML Engineer"},
            {"company": "DataStart", "title": "Data Scientist"},
        ],
        "projects": [
            {"name": "Sentiment Pipeline"},
            {"name": "Text Classification API"},
        ],
    }
    sample_jd = {
        "job_title": "Machine Learning Engineer",
        "responsibilities": ["deploy models", "build NLP pipelines"],
    }
    sample_match = {
        "matched_skills": ["Python", "TensorFlow", "NLP"],
        "missing_skills": ["Docker", "PyTorch", "Kubernetes"],
    }

    results = retrieve(sample_candidate, sample_jd, sample_match)
    print(f"Retrieved {len(results)} questions:\n")
    for r in results:
        print(f"  [{r['category'].upper()}] {r['id']}  (dist={r['distance']:.3f})")
        print(f"    {r['text'][:80]}...")
        print()
