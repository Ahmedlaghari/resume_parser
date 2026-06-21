"""
education_matcher.py
----------------------
Compares candidate's highest degree against the JD's required education
using a simple hierarchy. If the JD doesn't specify a requirement, we
don't penalize anyone (default to 100), per the spec.
"""

DEGREE_RANK = {
    "phd": 4,
    "doctorate": 4,
    "masters": 3,
    "master": 3,
    "bachelors": 2,
    "bachelor": 2,
    "associate": 1,
    "associates": 1,
    "none": 0,
}


def _rank(degree: str | None) -> int:
    if not degree:
        return 0
    return DEGREE_RANK.get(degree.strip().lower(), 0)


def education_score(candidate_degree: str | None, required_degree: str | None) -> float:
    if not required_degree:
        # JD didn't ask for a specific education level -> don't penalize.
        return 100.0

    candidate_rank = _rank(candidate_degree)
    required_rank = _rank(required_degree)

    if candidate_rank >= required_rank:
        return 100.0

    # Below requirement: lose 30 points per missing degree tier, floor at 0
    tiers_short = required_rank - candidate_rank
    return round(max(0.0, 100 - tiers_short * 30), 2)
