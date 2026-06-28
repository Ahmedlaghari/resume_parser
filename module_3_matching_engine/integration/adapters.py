# ============================================================
# adapters.py — bridges Module 1 + Module 2 outputs to Module 3 inputs
#
# Module 3's Pydantic models expect specific field names and types.
# Module 1 and Module 2 use slightly different field names and shapes.
# This file transforms the real outputs without touching any module's code.
#
# Two functions:
#   adapt_resume(module1_json)  → dict that matches CandidateResume schema
#   adapt_jd(module2_json)      → dict that matches JobDescription schema
# ============================================================

import re


# ============================================================
# SECTION 1 — EDUCATION MAPPING
# ============================================================

# Maps degree strings from Module 1 into the simple labels
# that education_matcher.py understands.
# Add more entries here as you see new degree formats come through.
DEGREE_MAP = {
    # Bachelors
    "bs": "Bachelors",
    "bsc": "Bachelors",
    "b.s.": "Bachelors",
    "bachelor": "Bachelors",
    "bachelors": "Bachelors",
    "b.e.": "Bachelors",
    "be": "Bachelors",
    "beng": "Bachelors",
    # Masters
    "ms": "Masters",
    "msc": "Masters",
    "m.s.": "Masters",
    "master": "Masters",
    "masters": "Masters",
    "mba": "Masters",
    # PhD
    "phd": "PhD",
    "ph.d.": "PhD",
    "doctorate": "PhD",
    "doctoral": "PhD",
    # Associate
    "associate": "Associate",
    "a-levels": "Associate",  # treating A-levels as closest equivalent
    "as": "Associate",
}

DEGREE_RANK = {"PhD": 4, "Masters": 3, "Bachelors": 2, "Associate": 1}


def _map_degree(raw_degree: str) -> str | None:
    """
    Takes a raw degree string like "BS Computer Science" or "Master of Science"
    and returns the simple label Module 3 needs: "Bachelors", "Masters", etc.

    Strategy: check if any key from DEGREE_MAP appears as a word in the
    degree string. Takes the highest-ranked match found, in case the
    string somehow contains multiple keywords.
    """
    if not raw_degree:
        return None

    raw_lower = raw_degree.strip().lower()
    best_label = None
    best_rank = 0

    for keyword, label in DEGREE_MAP.items():
        # Match as a whole word or at start of string to avoid false matches
        # e.g. "mba" shouldn't match inside "combinatorics"
        pattern = r"(^|\s|\.)" + re.escape(keyword) + r"($|\s|\.)"
        if re.search(pattern, raw_lower):
            rank = DEGREE_RANK.get(label, 0)
            if rank > best_rank:
                best_rank = rank
                best_label = label

    return best_label


def _highest_degree(education_list: list[dict]) -> str | None:
    """
    Module 1 outputs a list of education objects. This finds the highest
    degree from all of them and returns its simple label.
    """
    if not education_list:
        return None

    best_label = None
    best_rank = 0

    for edu in education_list:
        raw = edu.get("degree", "")
        label = _map_degree(raw)
        if label:
            rank = DEGREE_RANK.get(label, 0)
            if rank > best_rank:
                best_rank = rank
                best_label = label

    return best_label


# ============================================================
# SECTION 2 — EXPERIENCE PARSING
# ============================================================

def _parse_experience_string(exp_str: str) -> tuple[float, float | None]:
    """
    Parses experience strings from Module 2 into (min_years, max_years).

    Handles formats like:
        "5+ years"     → (5.0, None)
        "3-5 years"    → (3.0, 5.0)
        "2 to 4 years" → (2.0, 4.0)
        "5 years"      → (5.0, None)
        ""             → (0.0, None)
    """
    if not exp_str:
        return 0.0, None

    # "5+" or "5+ years"
    match = re.search(r"(\d+)\s*\+", exp_str)
    if match:
        return float(match.group(1)), None

    # "3-5 years" or "3 to 5 years"
    match = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)", exp_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    # "5 years" — single number
    match = re.search(r"(\d+)", exp_str)
    if match:
        return float(match.group(1)), None

    return 0.0, None


def _calc_years_experience(experience_list: list[dict]) -> float:
    """
    Module 1 outputs experience as a list of job objects, each with
    start/end dates. This calculates total years.

    If the list is empty (like for Ahmed who is a student), returns 0.
    If dates are present, tries to sum durations. Falls back to counting
    entries × 1.5 years as a rough estimate if dates aren't parseable.
    """
    if not experience_list:
        return 0.0

    total = 0.0
    for job in experience_list:
        # Try to extract years from duration strings like "2020 - 2022"
        start = job.get("start_year") or job.get("start") or ""
        end = job.get("end_year") or job.get("end") or "Present"

        start_match = re.search(r"(\d{4})", str(start))
        end_match = re.search(r"(\d{4})", str(end))

        if start_match and end_match:
            total += float(end_match.group(1)) - float(start_match.group(1))
        else:
            # Can't parse dates — count each job as ~1.5 years
            total += 1.5

    return round(total, 1)


# ============================================================
# SECTION 3 — MAIN ADAPTER FUNCTIONS
# ============================================================

def adapt_resume(module1_json: dict) -> dict:
    """
    Transforms a Module 1 (Resume Parser) output dict into the shape
    that Module 3's CandidateResume Pydantic model expects.

    Args:
        module1_json: The raw dict from Module 1's API response.

    Returns:
        A dict ready to be passed to CandidateResume(**result).
    """
    return {
        # "name" in Module 1 → "candidate_name" in Module 3
        "candidate_name": module1_json.get("name", "Unknown"),

        # "skills" matches directly — already a list of strings
        "skills": module1_json.get("skills", []),

        # "experience" is a list of job objects → calculate total years
        "total_years_experience": _calc_years_experience(
            module1_json.get("experience", [])
        ),

        # "education" is a list of objects → extract highest degree label
        "highest_degree": _highest_degree(
            module1_json.get("education", [])
        ),

        # "summary" in Module 1 → "summary_text" in Module 3
        # Also append project descriptions to give semantic matcher more signal
        "summary_text": _build_summary_text(module1_json),
    }


def adapt_jd(module2_json: dict) -> dict:
    """
    Transforms a Module 2 (JD Analyzer) output dict into the shape
    that Module 3's JobDescription Pydantic model expects.

    Args:
        module2_json: The raw dict from Module 2's API response.

    Returns:
        A dict ready to be passed to JobDescription(**result).
    """
    min_exp, max_exp = _parse_experience_string(
        module2_json.get("experience_required", "")
    )

    # responsibilities and qualifications are lists in Module 2 → join to strings
    responsibilities = module2_json.get("responsibilities", [])
    qualifications = module2_json.get("qualifications", [])

    return {
        "job_title": module2_json.get("job_title", ""),

        # required_skills already a list of strings ✓
        "required_skills": module2_json.get("required_skills", []),
        "preferred_skills": module2_json.get("nice_to_have_skills", []),

        "min_years_experience": min_exp,
        "max_years_experience": max_exp,

        # Module 2 doesn't extract required_education explicitly
        # so default to None (education_matcher returns 100 when None)
        "required_education": None,

        # Join lists into paragraph text for semantic matching
        "responsibilities_text": "\n".join(responsibilities),
        "qualifications_text": "\n".join(qualifications),
    }


# ============================================================
# SECTION 4 — HELPERS
# ============================================================

def _build_summary_text(module1_json: dict) -> str:
    """
    Combines the candidate's summary + project descriptions into one
    text block for semantic matching. The more meaningful text the
    semantic matcher has, the better it captures overall fit.
    """
    parts = []

    summary = module1_json.get("summary", "")
    if summary:
        parts.append(summary)

    projects = module1_json.get("projects", [])
    for project in projects:
        desc = project.get("description", "")
        if desc:
            parts.append(desc)

    return " ".join(parts)
