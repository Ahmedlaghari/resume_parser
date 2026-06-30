"""
skills_matcher.py
-------------------
Implements Layer 2 of the scoring system: weighted skill matching.

For each required skill in the JD:
  1. Try an exact (case-insensitive) match against the candidate's skills.
  2. If no exact match, embed the JD skill and every candidate skill and
     take the best cosine similarity. If it clears SIMILARITY_THRESHOLD,
     count it as a partial match using the similarity score itself.
  3. Otherwise the skill is unmatched (match_i = 0).

Then apply the weighted-average formula from the spec:
    skills_score = sum(weight_i * match_i) / sum(weight_i)
"""

import numpy as np

from .semantic_matcher import embed_texts
from .models import SkillMatchDetail

SIMILARITY_THRESHOLD = 0.75  # tune this if matches feel too loose/strict


def _exact_match(jd_skill: str, candidate_skills: list[str]) -> str | None:
    """Case-insensitive exact match. Returns the matching candidate skill string, or None."""
    jd_skill_lower = jd_skill.strip().lower()
    for cand_skill in candidate_skills:
        if cand_skill.strip().lower() == jd_skill_lower:
            return cand_skill
    return None


def match_skills(
    required_skills: list[str],
    candidate_skills: list[str],
    skill_weights: dict[str, float],
) -> tuple[float, list[SkillMatchDetail]]:
    """
    Returns (skills_score_0_to_100, list_of_SkillMatchDetail).

    skill_weights: dict mapping skill name -> weight. Any required skill
    not present in this dict defaults to weight 1.0 (per spec).
    """
    if not required_skills:
        return 100.0, []

    # Embed all JD skills and all candidate skills in two batches up front,
    # so we don't re-embed the candidate list once per unmatched JD skill.
    jd_vecs = embed_texts(required_skills)
    cand_vecs = embed_texts(candidate_skills) if candidate_skills else None

    weighted_sum = 0.0
    weight_total = 0.0
    details: list[SkillMatchDetail] = []

    for i, jd_skill in enumerate(required_skills):
        weight = skill_weights.get(jd_skill, 1.0)
        weight_total += weight

        exact = _exact_match(jd_skill, candidate_skills)
        if exact is not None:
            match_score = 1.0
            matched = True
            candidate_has = exact
        elif cand_vecs is not None:
            # Embeddings are already L2-normalised, so dot product == cosine similarity.
            # One matrix-vector multiply replaces the per-skill Python loop.
            scores = np.clip(cand_vecs @ jd_vecs[i], 0.0, 1.0)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            best_skill = candidate_skills[best_idx]
            if best_score >= SIMILARITY_THRESHOLD:
                match_score = best_score
                matched = True
                candidate_has = best_skill
            else:
                match_score = 0.0
                matched = False
                candidate_has = None
        else:
            match_score = 0.0
            matched = False
            candidate_has = None

        weighted_sum += weight * match_score
        details.append(
            SkillMatchDetail(
                skill=jd_skill,
                weight=round(weight, 3),
                matched=matched,
                match_score=round(match_score, 3),
                candidate_has=candidate_has,
            )
        )

    skills_score = (weighted_sum / weight_total) * 100 if weight_total > 0 else 100.0
    return round(skills_score, 2), details
