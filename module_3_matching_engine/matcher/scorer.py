"""
scorer.py
----------
Layer 1 of the scoring system: combines the four sub-scores
(skills, experience, education, semantic) into one final_score
using the category weights, and assigns a verdict bucket.

This file deliberately contains NO LLM calls and NO embedding calls --
everything here is plain arithmetic, so given the same inputs and
weights it always produces the same output. That determinism is the
whole point: it's what makes the score auditable.
"""

from .models import CategoryWeights, ScoreBreakdown

VERDICT_BUCKETS = [
    (90, 100, "Excellent match"),
    (70, 89, "Strong match"),
    (50, 69, "Good match"),
    (30, 49, "Partial match"),
    (0, 29, "Weak match"),
]


def verdict_for_score(score: float) -> str:
    for low, high, label in VERDICT_BUCKETS:
        if low <= score <= high:
            return label
    # Defensive fallback (shouldn't be reachable since buckets cover 0-100)
    return "Weak match"


def normalize_skill_weights(
    required_skills: list[str], skill_weights: dict[str, float] | None
) -> dict[str, float]:
    """
    Fills in default weight 1.0 for any required skill not explicitly
    weighted, and clips any provided weight into [0, 1] (per the
    'outside 0-1 range should clip or normalize' acceptance criterion).
    """
    skill_weights = skill_weights or {}
    resolved = {}
    for skill in required_skills:
        raw_weight = skill_weights.get(skill, 1.0)
        resolved[skill] = max(0.0, min(1.0, raw_weight))
    return resolved


def combine_scores(
    skills_score: float,
    experience_score: float,
    education_score: float,
    semantic_score: float,
    category_weights: CategoryWeights | None,
) -> tuple[float, ScoreBreakdown, CategoryWeights]:
    """
    Returns (final_score, breakdown, normalized_weights_actually_used).
    """
    weights = (category_weights or CategoryWeights()).normalized()

    final_score = (
        weights.skills_weight * skills_score
        + weights.experience_weight * experience_score
        + weights.education_weight * education_score
        + weights.semantic_weight * semantic_score
    )

    breakdown = ScoreBreakdown(
        skills_score=skills_score,
        experience_score=experience_score,
        education_score=education_score,
        semantic_similarity_score=semantic_score,
    )

    return round(final_score, 2), breakdown, weights
