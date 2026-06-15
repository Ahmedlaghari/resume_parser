"""
extractor.py — Field-by-field extraction from cleaned JD text.

Each function is responsible for exactly one field.
They are all pure functions: text in, value out.
None is returned when a field genuinely cannot be found — never crash.

Techniques used here:
  - Regex for structured patterns (salary, experience, employment type)
  - Section-header detection for bullets (responsibilities, qualifications…)
  - spaCy NER for location (GPE = geo-political entity) and job title
  - Simple heuristics for company name and industry
"""

import re
import spacy

# --------------------------------------------------------------------------
# Load spaCy model once at import time (not on every request).
# Run this first if you haven't: python -m spacy download en_core_web_sm
# --------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run:  python -m spacy download en_core_web_sm"
    )


# ==========================================================================
# HELPERS
# ==========================================================================

def _extract_section_bullets(text: str, headers: list[str]) -> list[str]:
    """
    Find a section by its header keyword(s) and return its bullet points.

    Example — for headers=["responsibilities", "what you'll do"]:
      Finds the line that contains any of those words, then collects
      every subsequent line that starts with '-', '•', a number, etc.
      Stops when it hits the next ALL-CAPS or Title-Case section header
      or runs out of lines.

    Returns a list of clean strings (one per bullet).
    """
    lines = text.split("\n")
    bullets: list[str] = []
    inside_section = False

    # Regex: a line is a bullet if it starts with -, *, •, or a digit+dot
    bullet_re = re.compile(r"^\s*[-*\u2022\d\.]+\s+(.+)")
    # Regex: a new section header (a short line that doesn't look like a bullet)
    header_re = re.compile(r"^[A-Z][A-Za-z &/]{2,50}:?\s*$")

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # Detect section header
        if any(h in lower for h in headers):
            inside_section = True
            continue

        if inside_section:
            # Stop if we hit another section header
            if header_re.match(stripped) and stripped not in ("", "-"):
                # Check it's not one of our target headers again
                if not any(h in stripped.lower() for h in headers):
                    inside_section = False
                    continue

            # Collect bullet lines
            m = bullet_re.match(stripped)
            if m:
                bullets.append(m.group(1).strip())
            elif stripped and not header_re.match(stripped):
                # Plain line inside the section (no bullet prefix) — keep it
                # but only if it's reasonably long (not a header)
                if len(stripped) > 20:
                    bullets.append(stripped)

    return bullets


# ==========================================================================
# INDIVIDUAL FIELD EXTRACTORS
# ==========================================================================

def extract_job_title(text: str) -> str | None:
    """
    Strategy:
      1. Look for hiring phrases on the first 10 lines.
      2. Fall back to the very first non-blank line.
      3. Use spaCy to confirm it looks like an entity (optional).
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Common hiring phrase patterns
    hiring_patterns = [
        r"(?:we are|we're)\s+(?:hiring|looking for|seeking)\s+(?:a|an)\s+(.+?)(?:\.|,|$)",
        r"(?:position|role|job title)\s*[:\-]\s*(.+?)(?:\n|$)",
        r"^(.+?)\s*(?:-|–|at|@)\s*.+$",    # "ML Engineer - TechCorp"
    ]
    for line in lines[:10]:
        for pattern in hiring_patterns:
            m = re.search(pattern, line, re.IGNORECASE)
            if m:
                title = m.group(1).strip().rstrip(".,")
                if 3 < len(title) < 80:
                    return title

    # Last resort: first non-blank line (usually the job title)
    if lines:
        first = lines[0]
        if len(first) < 80:   # titles are short
            return first

    return None


def extract_company(text: str) -> str | None:
    """
    Look for 'at <Company>', 'with <Company>', or spaCy ORG entities
    near the top of the document.
    """
    # Pattern: "Engineer at Acme Corp" or "join us at Acme"
    m = re.search(
        r"\bat\s+([A-Z][A-Za-z0-9 &.,'-]{1,50}?)(?:\s*[\.,\n]|$)",
        text[:600]
    )
    if m:
        return m.group(1).strip()

    # spaCy ORG fallback (look only in first 300 chars — intro paragraph)
    doc = nlp(text[:300])
    for ent in doc.ents:
        if ent.label_ == "ORG":
            return ent.text

    return None


def extract_location(text: str) -> str | None:
    """
    Use spaCy GPE (Geo-Political Entity) detection.
    Also look for explicit remote/hybrid keywords.
    """
    # Check for remote / hybrid keywords first
    lower = text[:500].lower()
    if "fully remote" in lower:
        location_prefix = "Remote"
    elif "remote" in lower and "hybrid" in lower:
        location_prefix = "Remote / Hybrid"
    elif "remote" in lower:
        location_prefix = "Remote"
    elif "hybrid" in lower:
        location_prefix = "Hybrid"
    else:
        location_prefix = None

    # spaCy GPE detection
    doc = nlp(text[:600])
    gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

    if gpe_entities and location_prefix:
        return f"{location_prefix} / {', '.join(gpe_entities[:2])}"
    elif gpe_entities:
        return ", ".join(gpe_entities[:2])
    elif location_prefix:
        return location_prefix

    # Regex fallback: "Location: Karachi, Pakistan"
    m = re.search(r"location\s*[:\-]\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    return None


def extract_employment_type(text: str) -> str | None:
    """Keyword match for employment type."""
    lower = text.lower()
    # Order matters — check more specific terms first
    types = [
        ("Contract",   ["contract", "freelance", "consulting"]),
        ("Part-time",  ["part-time", "part time"]),
        ("Internship", ["internship", "intern"]),
        ("Full-time",  ["full-time", "full time", "permanent"]),
    ]
    for label, keywords in types:
        if any(kw in lower for kw in keywords):
            return label

    return None


def extract_experience(text: str) -> str | None:
    """
    Regex patterns for experience:
      - "3+ years"
      - "3-5 years of experience"
      - "minimum 5 years"
    """
    patterns = [
        r"(\d+\s*[-–to]+\s*\d+)\s*(?:\+)?\s*years?",   # "3-5 years"
        r"(\d+\+)\s*years?",                              # "5+ years"
        r"minimum\s+(\d+)\s*years?",                      # "minimum 3 years"
        r"at\s+least\s+(\d+)\s*years?",                  # "at least 2 years"
        r"(\d+)\s*years?\s+of\s+experience",              # "5 years of experience"
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    return None


def extract_salary(text: str) -> str | None:
    """
    Regex for common salary formats:
      - $80,000 - $100,000
      - £50k - £70k
      - PKR 150,000/month
      - 80k - 100k USD
    """
    patterns = [
        # Dollar / Pound / Euro range
        r"[\$£€]\s*\d[\d,k]+\s*(?:-|–|to)\s*[\$£€]?\s*\d[\d,k]+(?:\s*(?:USD|GBP|EUR|PKR|per\s+\w+))?",
        # PKR or currency code first
        r"(?:PKR|USD|GBP|EUR)\s*\d[\d,k]+\s*(?:-|–|to)\s*(?:PKR|USD|GBP|EUR)?\s*\d[\d,k]+",
        # "80k - 100k USD"
        r"\d+k\s*(?:-|–|to)\s*\d+k(?:\s*(?:USD|GBP|EUR|PKR))?",
        # Single figure: "$90,000/year"
        r"[\$£€]\s*\d[\d,k]+(?:\s*per\s+(?:year|annum|month))?",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip()

    return None


def extract_responsibilities(text: str) -> list[str]:
    """Pull bullets from Responsibilities / What You'll Do sections."""
    headers = [
        "responsibilities", "what you'll do", "what you will do",
        "your role", "role overview", "duties", "you will",
        "in this role", "key responsibilities",
    ]
    return _extract_section_bullets(text, headers)


def extract_qualifications(text: str) -> list[str]:
    """Pull bullets from Requirements / Qualifications sections."""
    headers = [
        "requirements", "qualifications", "what we need",
        "what we're looking for", "what we look for",
        "must have", "you have", "you bring", "minimum qualifications",
        "basic qualifications", "required qualifications",
    ]
    return _extract_section_bullets(text, headers)


def extract_benefits(text: str) -> list[str]:
    """Pull bullets from Benefits / Perks sections."""
    headers = [
        "benefits", "perks", "what we offer", "why join us",
        "compensation", "we provide", "you'll get",
    ]
    return _extract_section_bullets(text, headers)


def extract_industry(text: str) -> str | None:
    """
    Simple keyword-to-industry mapping.
    Returns the first match or None.
    """
    industry_map: dict[str, list[str]] = {
        "Technology":       ["software", "saas", "cloud", "ai", "machine learning",
                             "startup", "tech", "platform"],
        "Finance":          ["fintech", "banking", "finance", "investment",
                             "trading", "payments"],
        "Healthcare":       ["healthcare", "health tech", "medical", "pharma",
                             "clinical", "hospital"],
        "E-commerce":       ["e-commerce", "ecommerce", "retail", "marketplace"],
        "Telecommunications": ["telecom", "telco", "5g", "networking"],
        "Education":        ["edtech", "education", "e-learning", "learning platform"],
        "Cybersecurity":    ["cybersecurity", "security", "infosec", "soc"],
        "Gaming":           ["gaming", "game development", "unity", "unreal"],
    }
    lower = text.lower()
    for industry, keywords in industry_map.items():
        if any(kw in lower for kw in keywords):
            return industry

    return None
