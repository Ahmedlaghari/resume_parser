"""
test_logic.py
---------------
Tests the deterministic parts of the engine end-to-end, and the
skills matcher's fuzzy-matching *logic* using a fake embedder (so this
file runs without downloading the real ~90MB sentence-transformers
model -- useful for quick CI / sandboxed runs). Swap FAKE_SIMILARITIES
for the real get_embedder() in your own environment to test actual
semantic quality.

Run with: python -m pytest tests/test_logic.py -v
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import matcher.semantic_matcher as semantic_matcher
import matcher.skills_matcher as skills_matcher

# ----------------------------------------------------------------
# Fake embedder: hand-crafted so known synonym pairs are "similar"
# and unrelated terms are not, without needing the real model.
# ----------------------------------------------------------------
FAKE_VOCAB = {
    "ml": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
    "machine learning": np.array([0.95, 0.05, 0.0, 0.0, 0.0]),   # close to "ml"
    "python": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
    "tensorflow": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    "tf": np.array([0.0, 0.05, 0.95, 0.0, 0.0]),                  # close to "tensorflow"
    "docker": np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
}


def fake_embed_texts(texts: list[str]) -> np.ndarray:
    """Known synonym pairs come from FAKE_VOCAB above. Anything else gets a
    deterministic-but-distinct vector derived from a hash of the word, so
    two unrelated unknown skills (e.g. "Java" and "AWS") don't accidentally
    collide the way they would with a single shared default vector."""
    vectors = []
    for t in texts:
        key = t.strip().lower()
        if key in FAKE_VOCAB:
            vectors.append(FAKE_VOCAB[key])
        else:
            seed = int.from_bytes(key.encode(), "little") % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.random(5)
            vectors.append(v / np.linalg.norm(v))
    return np.array(vectors)


def setup_module(module):
    # Patch the embed_texts function everywhere it's been imported into.
    semantic_matcher.embed_texts = fake_embed_texts
    skills_matcher.embed_texts = fake_embed_texts


# ----------------------------------------------------------------
# experience_matcher tests
# ----------------------------------------------------------------
from matcher.experience_matcher import experience_score


def test_experience_within_range_is_100():
    assert experience_score(4, min_years=3, max_years=5) == 100.0


def test_experience_below_min_partial_credit():
    score = experience_score(2, min_years=3, max_years=5)  # 1 year short
    assert 0 < score < 100


def test_experience_far_below_min_is_low():
    score = experience_score(0, min_years=5, max_years=8)
    assert score < 20


def test_experience_overqualified_floor():
    score = experience_score(20, min_years=3, max_years=5)
    assert score >= 70  # never craters just for being overqualified


# ----------------------------------------------------------------
# education_matcher tests
# ----------------------------------------------------------------
from matcher.education_matcher import education_score


def test_education_meets_requirement():
    assert education_score("Masters", "Bachelors") == 100.0


def test_education_no_requirement_defaults_100():
    assert education_score(None, None) == 100.0


def test_education_below_requirement_penalized():
    score = education_score("Associate", "Masters")
    assert score < 100.0


# ----------------------------------------------------------------
# skills_matcher tests (using the fake embedder above)
# ----------------------------------------------------------------
def test_exact_skill_match():
    score, details = skills_matcher.match_skills(
        required_skills=["Python"],
        candidate_skills=["Python"],
        skill_weights={"Python": 1.0},
    )
    assert score == 100.0
    assert details[0].matched is True
    assert details[0].match_score == 1.0


def test_semantic_synonym_match_above_threshold():
    # JD wants "Machine Learning", candidate only listed "ML" -> should match via embedding
    score, details = skills_matcher.match_skills(
        required_skills=["Machine Learning"],
        candidate_skills=["ML"],
        skill_weights={"Machine Learning": 1.0},
    )
    assert details[0].matched is True
    assert details[0].candidate_has == "ML"


def test_unrelated_skill_does_not_match():
    score, details = skills_matcher.match_skills(
        required_skills=["Python"],
        candidate_skills=["Docker"],
        skill_weights={"Python": 1.0},
    )
    assert details[0].matched is False
    assert score == 0.0


def test_weighted_formula_matches_hand_calculation():
    # This is the exact example from the spec walkthrough:
    # Python=1.0(have), Java=0.5(missing), Docker=0.7(have), AWS=0.6(have)
    # expected = (1.0*1 + 0.5*0 + 0.7*1 + 0.6*1) / 2.8 * 100 = 82.14
    score, details = skills_matcher.match_skills(
        required_skills=["Python", "Java", "Docker", "AWS"],
        candidate_skills=["Python", "Docker", "AWS"],
        skill_weights={"Python": 1.0, "Java": 0.5, "Docker": 0.7, "AWS": 0.6},
    )
    assert abs(score - 82.14) < 0.1


def test_missing_high_weight_skill_hurts_more_than_low_weight():
    # Same candidate skill set, but flip which missing skill carries the weight
    candidate_skills = ["Java"]  # has Java, missing Python

    score_python_high, _ = skills_matcher.match_skills(
        required_skills=["Python", "Java"],
        candidate_skills=candidate_skills,
        skill_weights={"Python": 1.0, "Java": 0.2},
    )
    score_python_low, _ = skills_matcher.match_skills(
        required_skills=["Python", "Java"],
        candidate_skills=candidate_skills,
        skill_weights={"Python": 0.2, "Java": 1.0},
    )
    # Missing Python should hurt more when Python is weighted higher
    assert score_python_high < score_python_low


def test_no_required_skills_returns_perfect_score():
    score, details = skills_matcher.match_skills([], ["Python"], {})
    assert score == 100.0
    assert details == []


# ----------------------------------------------------------------
# scorer tests
# ----------------------------------------------------------------
from matcher.scorer import combine_scores, verdict_for_score, normalize_skill_weights
from matcher.models import CategoryWeights


def test_category_weights_auto_normalize():
    weights = CategoryWeights(skills_weight=2, experience_weight=2, education_weight=0, semantic_weight=0)
    normalized = weights.normalized()
    total = (
        normalized.skills_weight
        + normalized.experience_weight
        + normalized.education_weight
        + normalized.semantic_weight
    )
    assert abs(total - 1.0) < 1e-9
    assert normalized.skills_weight == normalized.experience_weight == 0.5


def test_combine_scores_matches_hand_calculation():
    # Reuses the full worked example from the top of the conversation
    weights = CategoryWeights(
        skills_weight=0.5, experience_weight=0.25, education_weight=0.15, semantic_weight=0.10
    )
    final, breakdown, used = combine_scores(
        skills_score=82.14, experience_score=85, education_score=80, semantic_score=88,
        category_weights=weights,
    )
    assert abs(final - 83.12) < 0.05


def test_verdict_buckets():
    assert verdict_for_score(95) == "Excellent match"
    assert verdict_for_score(75) == "Strong match"
    assert verdict_for_score(55) == "Good match"
    assert verdict_for_score(35) == "Partial match"
    assert verdict_for_score(10) == "Weak match"
    assert verdict_for_score(0) == "Weak match"
    assert verdict_for_score(100) == "Excellent match"


def test_skill_weight_clipping():
    resolved = normalize_skill_weights(["Python", "Java"], {"Python": 1.5, "Java": -0.3})
    assert resolved["Python"] == 1.0  # clipped down
    assert resolved["Java"] == 0.0    # clipped up to 0
