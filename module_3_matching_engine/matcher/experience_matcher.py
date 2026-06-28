"""
experience_matcher.py
-----------------------
Compares a candidate's total years of experience against the JD's
required range and produces a 0-100 score.

Scoring logic:
  - Inside [min, max]               -> 100
  - Below min, but within 2 years   -> linear ramp from 100 down to 40
  - Below min by more than 2 years  -> linear ramp from 40 down to 0
  - Above max (overqualified)       -> gentle penalty, never below 70
    (overqualified is usually a much smaller problem than underqualified,
    so it shouldn't crater the score the way being underqualified does)
"""

NEAR_MISS_YEARS = 2.0          # "slightly below" window
FAR_MISS_FLOOR_YEARS = 5.0     # beyond this far below min, score bottoms out at 0
OVERQUALIFIED_FLOOR = 70.0     # being overqualified never drops below this


def experience_score(
    candidate_years: float,
    min_years: float,
    max_years: float | None,
) -> float:
    candidate_years = max(0.0, candidate_years)
    min_years = max(0.0, min_years)

    # No max specified -> treat as "no upper bound"
    effective_max = max_years if max_years is not None else float("inf")

    if min_years <= candidate_years <= effective_max:
        return 100.0

    if candidate_years < min_years:
        shortfall = min_years - candidate_years
        if shortfall <= NEAR_MISS_YEARS:
            # Linear ramp: 0 years short -> 100, NEAR_MISS_YEARS short -> 40
            return round(100 - (shortfall / NEAR_MISS_YEARS) * 60, 2)
        else:
            # Linear ramp: NEAR_MISS_YEARS short -> 40, FAR_MISS_FLOOR_YEARS+ short -> 0
            extra_shortfall = shortfall - NEAR_MISS_YEARS
            span = FAR_MISS_FLOOR_YEARS - NEAR_MISS_YEARS
            score = 40 - (extra_shortfall / span) * 40
            return round(max(0.0, score), 2)

    # candidate_years > effective_max: overqualified
    excess = candidate_years - effective_max
    # Gentle penalty: lose 5 points per year over, floor at OVERQUALIFIED_FLOOR
    score = 100 - (excess * 5)
    return round(max(OVERQUALIFIED_FLOOR, score), 2)
