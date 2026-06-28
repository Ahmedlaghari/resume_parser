# ============================================================
# pipeline.py — connects Module 1 + Module 2 → Module 3
#
# Usage:
#   python pipeline.py resume.json jd.json
#   python pipeline.py resume.json jd.json --weights weights.json
#
# Or import and call directly:
#   from pipeline import run
#   result = run(module1_json, module2_json)
# ============================================================

import sys
import json
import pathlib
import argparse

# Add project root to path so imports work from anywhere
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from integration.adapters import adapt_resume, adapt_jd
import numpy as np

def _setup_embedder():
    """
    Tries the real sentence-transformers model first.
    Falls back to a fake embedder if unavailable (no internet, sandbox).
    On your own machine this does nothing — real model loads automatically.
    """
    try:
        import matcher.semantic_matcher as sm
        sm.get_embedder()
        print("Using real sentence-transformers model (all-MiniLM-L6-v2)")
    except Exception:
        import matcher.semantic_matcher as sm
        import matcher.skills_matcher as sk
        def _fake(texts):
            vecs = []
            for t in texts:
                seed = abs(hash(t.strip().lower())) % (2**32)
                rng  = np.random.default_rng(seed)
                v    = rng.standard_normal(64)
                vecs.append(v / np.linalg.norm(v))
            return np.array(vecs)
        sm.embed_texts = _fake
        sk.embed_texts = _fake
        print("WARNING: real model unavailable, using fake embedder (semantic scores not meaningful)")

_setup_embedder()


def run(
    module1_json: dict,
    module2_json: dict,
    skill_weights: dict = None,
    category_weights: dict = None,
    generate_explanations: bool = True,
) -> dict:
    """
    Takes raw Module 1 + Module 2 outputs, adapts them, and returns
    the full Module 3 match result.

    Args:
        module1_json:          Raw output dict from Module 1 (resume parser).
        module2_json:          Raw output dict from Module 2 (JD analyzer).
        skill_weights:         Optional. e.g. {"Python": 1.0, "Docker": 0.6}
        category_weights:      Optional. e.g. {"skills_weight": 0.5, ...}
        generate_explanations: Set False to skip the Groq LLM call (faster).

    Returns:
        The full Module 3 response dict (summary_list + detailed_results).
    """
    # Step 1: transform both outputs into Module 3's expected shapes
    adapted_resume = adapt_resume(module1_json)
    adapted_jd     = adapt_jd(module2_json)

    # Step 2: build the request body Module 3's API expects
    request_body = {
        "job_description": adapted_jd,
        "candidates": [adapted_resume],
        "generate_explanations": generate_explanations,
    }
    if skill_weights:
        request_body["skill_weights"] = skill_weights
    if category_weights:
        request_body["category_weights"] = category_weights

    # Step 3: call Module 3
    from fastapi.testclient import TestClient
    from main import app

    client = TestClient(app)
    response = client.post("/match-candidates", json=request_body)
    response.raise_for_status()
    return response.json()


def _print_result(result: dict):
    """Pretty-prints the match result to the terminal."""
    print(f"\nJob: {result['job_title']}")
    print(f"Candidates evaluated: {result['total_candidates_evaluated']}")
    print("=" * 60)

    for c in result["summary_list"]:
        print(f"\nRank {c['rank']}: {c['candidate_name']}")
        print(f"  Score:   {c['final_score']}/100")
        print(f"  Verdict: {c['verdict']}")
        print(f"  Reason:  {c['one_line_reason']}")

    print("\n--- Detailed breakdown ---")
    for d in result["detailed_results"]:
        print(f"\n{d['candidate_name']}")
        b = d["breakdown"]
        print(f"  Skills:     {b['skills_score']}/100")
        print(f"  Experience: {b['experience_score']}/100")
        print(f"  Education:  {b['education_score']}/100")
        print(f"  Semantic:   {b['semantic_similarity_score']}/100")

        print(f"\n  Skill matches:")
        for s in d["skill_match_detail"]:
            icon = "✓" if s["matched"] else "✗"
            has  = f"  (via '{s['candidate_has']}')" if s["candidate_has"] != s["skill"] and s["candidate_has"] else ""
            print(f"    {icon} {s['skill']:<20} weight={s['weight']}  score={s['match_score']}{has}")

        if d["missing_critical_skills"]:
            print(f"\n  Missing critical skills: {', '.join(d['missing_critical_skills'])}")

        print(f"\n  Explanation: {d['explanation']}")

    print(f"\n{'='*60}")
    print("Weights used:")
    cw = result["weights_used"]["category_weights"]
    print(f"  skills={cw['skills_weight']}  experience={cw['experience_weight']}  education={cw['education_weight']}  semantic={cw['semantic_weight']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Module 1 + 2 output through Module 3")
    parser.add_argument("resume", help="Path to Module 1 output JSON file")
    parser.add_argument("jd",     help="Path to Module 2 output JSON file")
    parser.add_argument("--weights", help="Optional path to weights JSON file", default=None)
    parser.add_argument("--no-explain", action="store_true", help="Skip LLM explanation (faster)")
    args = parser.parse_args()

    module1_json = json.loads(pathlib.Path(args.resume).read_text())
    module2_json = json.loads(pathlib.Path(args.jd).read_text())

    skill_weights    = None
    category_weights = None
    if args.weights:
        w = json.loads(pathlib.Path(args.weights).read_text())
        skill_weights    = w.get("skill_weights")
        category_weights = w.get("category_weights")

    result = run(
        module1_json,
        module2_json,
        skill_weights=skill_weights,
        category_weights=category_weights,
        generate_explanations=not args.no_explain,
    )
    _print_result(result)
