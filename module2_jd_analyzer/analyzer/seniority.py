"""
seniority.py — Detect the seniority level of a role from JD text.

Strategy (two layers):
  1. Keyword scan — fast, covers 90% of cases.
  2. Experience-year fallback — if no keyword matched, infer from
     the years-of-experience figure extracted elsewhere.

Returns one of: "Junior" | "Mid" | "Senior" | "Lead" | "Principal"
Returns None if there is genuinely no signal.
"""

import re


# --------------------------------------------------------------------------
# Keyword → seniority mapping.
# Checked in order — first match wins.  More specific terms go first.
# --------------------------------------------------------------------------
SENIORITY_KEYWORDS: list[tuple[str, list[str]]] = [
    ("Principal", ["principal", "staff engineer", "distinguished", "fellow"]),
    ("Lead",      ["tech lead", "team lead", "engineering lead", "lead engineer",
                   "lead developer"]),
    ("Senior",    ["senior", "sr.", "sr ", "5+ years", "6+ years", "7+ years",
                   "8+ years", "10+ years"]),
    ("Mid",       ["mid-level", "mid level", "intermediate", "2-5 years",
                   "3-5 years", "3+ years", "4+ years"]),
    ("Junior",    ["junior", "entry level", "entry-level", "associate",
                   "graduate", "0-2 years", "1-2 years", "fresh"]),
]


def _years_to_seniority(years_str: str) -> str | None:
    """
    Given a string like '3-5 years' or '7+ years', return a seniority label.
    Only called when keyword matching found nothing.
    """
    # Pull out all numbers from the string
    nums = list(map(int, re.findall(r"\d+", years_str)))
    if not nums:
        return None

    # Use the highest number mentioned (upper bound or the single figure)
    max_years = max(nums)

    if max_years <= 2:
        return "Junior"
    elif max_years <= 5:
        return "Mid"
    elif max_years <= 8:
        return "Senior"
    else:
        return "Lead"


def detect_seniority(text: str, experience_required: str | None = None) -> str | None:
    """
    Main entry point.

    Args:
        text:                Full cleaned JD text (lowercased inside this fn).
        experience_required: Optional string like "3-5 years" already extracted
                             by extractor.py — used as fallback.

    Returns:
        Seniority string or None.

    Usage:
        from analyzer.seniority import detect_seniority
        level = detect_seniority(clean_text, experience_str)
    """
    lower = text.lower()

    # Layer 1 — keyword scan
    for level, keywords in SENIORITY_KEYWORDS:
        for kw in keywords:
            if kw in lower:
                return level

    # Layer 2 — fall back to experience years
    if experience_required:
        return _years_to_seniority(experience_required)

    return None
