# ============================================================
# extractor.py — Extract Structured Fields from Raw Resume Text
# ============================================================
# This is the brain of the parser. Given raw unstructured text,
# it finds and returns structured data.
#
# Strategy:
#   - Regex  → precise patterns (email, phone, URLs)
#   - spaCy  → NER for person name and location
#   - Section matching → split resume into labeled blocks
#     (Skills, Experience, Education, etc.) then parse each
# ============================================================

import re
import spacy
from typing import Optional
from parser.models import ResumeData, ExperienceEntry, EducationEntry, ProjectEntry

# ── Load spaCy model once at import time (not per-request) ──────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


# ============================================================
# SECTION 1 — REGEX EXTRACTORS
# ============================================================

def extract_email(text: str) -> Optional[str]:
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
    matches = re.findall(pattern, text)
    for match in matches:
        digits_only = re.sub(r"\D", "", match)
        if len(digits_only) >= 7:
            return match.strip()
    return None


def extract_linkedin(text: str) -> Optional[str]:
    pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_github(text: str) -> Optional[str]:
    pattern = r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


# ============================================================
# SECTION 2 — SPACY NER EXTRACTORS
# ============================================================

def extract_name(text: str) -> Optional[str]:
    top_text = text[:500]
    doc = nlp(top_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), None)
    if first_line and len(first_line.split()) <= 4 and not re.search(r"[@/]", first_line):
        return first_line
    return None


def extract_location(text: str) -> Optional[str]:
    doc = nlp(text[:1000])
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.strip()
    return None


# ============================================================
# SECTION 3 — SECTION SPLITTER
# ============================================================

SECTION_HEADERS = {
    "summary":          ["summary", "objective", "profile", "about me", "about"],
    "skills":           ["skills", "technical skills", "core competencies",
                         "technologies", "tools", "expertise"],
    "experience":       ["experience", "work experience", "employment",
                         "work history", "professional experience"],
    "education":        ["education", "academic background", "qualifications",
                         "academics"],
    "projects":         ["projects", "personal projects", "key projects",
                         "portfolio"],
    "certifications":   ["certifications", "certificates", "courses",
                         "training", "achievements"],
}


def split_into_sections(text: str) -> dict:
    sections = {key: "" for key in SECTION_HEADERS}
    current_section = "summary"
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip().lower()
        matched_section = None
        for section_name, keywords in SECTION_HEADERS.items():
            if any(stripped == kw or stripped.startswith(kw + ":") for kw in keywords):
                matched_section = section_name
                break
        if matched_section:
            current_section = matched_section
        else:
            sections[current_section] += line + "\n"

    return sections


# ============================================================
# SECTION 4 — FIELD PARSERS FOR EACH SECTION
# ============================================================

def parse_skills(skills_text: str) -> list:
    """
    Extract a list of skills from the skills section.

    Handles any reasonable format:
      - Comma-separated:  "Python, Java, SQL"
      - Bullet/newline:   each skill on its own line
      - Pipes:            "Python | Java | SQL"
      - Semicolons:       "Python; Java; SQL"
      - Colons used as category headers: "Languages: Python, Java"
        → strips the category label and keeps the skills
      - Slash-separated:  "Python/Django"  → kept as-is (one skill)
      - Numbered lists:   "1. Python  2. Java" → strips the number
      - Inline paragraphs / free-form sentences → each token that
        looks like a recognisable skill word is kept

    No hard length cap is enforced so that compound skill names
    (e.g. "Machine Learning & Deep Learning") are not silently dropped.
    Very long lines that look like prose sentences are split further.
    """
    if not skills_text.strip():
        return []

    text = skills_text

    # ── 1. Strip category-header prefixes like "Languages: " ──────────────
    # e.g. "Languages: Python, Java" → "Python, Java"
    text = re.sub(r"^\s*[\w &/]+\s*:\s*", "", text, flags=re.MULTILINE)

    # ── 2. Normalise separators ────────────────────────────────────────────
    # Bullets, pipes, semicolons → commas
    text = re.sub(r"[•●▪\-\|;]", ",", text)
    # Numbered-list prefixes like "1." "2)" → comma
    text = re.sub(r"^\s*\d+[\.\)]\s*", ",", text, flags=re.MULTILINE)

    # ── 3. Split on commas and newlines ───────────────────────────────────
    raw_tokens = re.split(r"[,\n]", text)

    skills = []
    for token in raw_tokens:
        token = token.strip().strip(".")

        if not token:
            continue

        # If the token is a long prose sentence (> ~60 chars and has spaces),
        # try to break it up further on " and ", " & ", or multiple spaces.
        if len(token) > 60 and " " in token:
            sub_tokens = re.split(r"\s{2,}|\band\b|&", token, flags=re.IGNORECASE)
            for sub in sub_tokens:
                sub = sub.strip().strip(".")
                if sub and len(sub) >= 2:
                    skills.append(sub)
        else:
            if len(token) >= 2:
                skills.append(token)

    # ── 4. Deduplicate (case-insensitive), preserve first-seen order ───────
    seen = set()
    unique = []
    for s in skills:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def parse_experience(experience_text: str) -> list:
    if not experience_text.strip():
        return []

    blocks = re.split(r"\n{2,}", experience_text.strip())
    entries = []

    duration_pattern = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4})"
        r"\s*[-–—to]+\s*"
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\.?\s?\d{4}|Present|Current|Now)",
        re.IGNORECASE
    )

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        title = None
        company = None
        duration = None
        description_lines = []

        for i, line in enumerate(lines):
            dur_match = duration_pattern.search(line)
            if dur_match and not duration:
                duration = dur_match.group(0).strip()
                remainder = line[:dur_match.start()].strip(" |,–-")
                if remainder and not company:
                    company = remainder
            elif i == 0 and not title:
                title = line
            elif i == 1 and not company:
                company = line
            else:
                description_lines.append(line)

        if title or company:
            entries.append(ExperienceEntry(
                title=title,
                company=company,
                duration=duration,
                description=" ".join(description_lines) or None
            ))

    return entries


def parse_education(education_text: str) -> list:
    if not education_text.strip():
        return []

    blocks = re.split(r"\n{2,}", education_text.strip())
    entries = []
    year_pattern = re.compile(r"\b(19|20)\d{2}\b")
    degree_keywords = [
        "bs", "be", "bsc", "ba", "ms", "msc", "ma", "mba", "phd",
        "bachelor", "master", "doctorate", "diploma", "associate",
        "b.s.", "m.s.", "b.e.", "m.e."
    ]

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        degree = None
        institution = None
        year = None

        for line in lines:
            year_match = year_pattern.search(line)
            if year_match and not year:
                year = year_match.group(0)
            line_lower = line.lower()
            if any(kw in line_lower for kw in degree_keywords) and not degree:
                degree = line
            elif not institution and line not in (degree or ""):
                institution = line

        if degree or institution:
            entries.append(EducationEntry(
                degree=degree,
                institution=institution,
                year=year
            ))

    return entries


def parse_projects(projects_text: str) -> list:
    """
    Extract structured project entries from free-form text.

    Handles all common resume project formats:

    1. Blank-line-separated blocks (classic):
         Project Name
         Description line
         Tech: Python, Flask

    2. Bullet-prefixed entries (no blank lines between projects):
         • Project Name — description here
         • Another Project: did X using Python and React

    3. Numbered lists:
         1. Project Name
            Description
         2. Next Project

    4. Inline tech — finds the tech stack even when not on a
       dedicated "Tech:" line, by scanning for parenthesised
       lists, "using X, Y, Z", "built with X", or slash/comma
       clusters that look like technology names.

    5. Single-line entries:
         Project Name – short description (Python, Flask)

    Tech stack detection is expanded to recognise:
      - "tech:", "technologies:", "tools:", "built with:",
        "stack:", "frameworks:", "libraries:", "languages:"
      - Parenthesised tech at end of line: "My App (Python, React)"
      - "using X, Y and Z" anywhere in description
    """
    if not projects_text.strip():
        return []

    # ── Patterns ──────────────────────────────────────────────────────────
    # Explicit tech-label line
    tech_label_re = re.compile(
        r"^(?:tech(?:nologies)?(?:\s*used)?|stack|tools?|built\s+with"
        r"|frameworks?|libraries|languages?)\s*[:\-]\s*(.*)",
        re.IGNORECASE
    )
    # Parenthesised tech at end: "My App (Python, React, SQL)"
    paren_tech_re = re.compile(r"\(([^)]{3,})\)\s*$")
    # "using X, Y, Z" or "built with X"
    using_re = re.compile(
        r"(?:using|built\s+with|written\s+in|developed\s+(?:in|with))\s+"
        r"([\w\s,/\-\.+#]+)",
        re.IGNORECASE
    )
    # Bullet prefixes: •, ●, ▪, -, *, or numbered "1." "2)"
    bullet_re = re.compile(r"^\s*(?:[•●▪\-\*]|\d+[\.\)])\s+")
    # Em-dash or colon used as name–description separator on one line
    separator_re = re.compile(r"\s+[–—:]\s+", re.UNICODE)

    def _extract_tech(text_fragment: str) -> list:
        """Pull a tech list from any fragment of text."""
        # Try "using / built with …"
        m = using_re.search(text_fragment)
        if m:
            raw = m.group(1)
            items = [t.strip() for t in re.split(r"[,/]|\band\b", raw, flags=re.IGNORECASE)
                     if t.strip() and len(t.strip()) > 1]
            if items:
                return items
        # Try parenthesised list
        m = paren_tech_re.search(text_fragment)
        if m:
            raw = m.group(1)
            items = [t.strip() for t in re.split(r"[,|;/]", raw) if t.strip()]
            if items:
                return items
        return []

    # ── Step 1: decide splitting strategy ─────────────────────────────────
    # If bullet-prefixed lines exist, split on bullets (ignoring blank lines).
    # Otherwise fall back to blank-line block splitting.
    lines = projects_text.splitlines()
    has_bullets = sum(1 for l in lines if bullet_re.match(l)) >= 2

    if has_bullets:
        # Regroup: each bullet starts a new project; continuation lines
        # (non-bullet, non-blank) belong to the current project.
        raw_blocks = []
        current: list = []
        for line in lines:
            if bullet_re.match(line):
                if current:
                    raw_blocks.append("\n".join(current))
                # Strip the bullet prefix before storing
                current = [bullet_re.sub("", line).strip()]
            elif line.strip():
                current.append(line.strip())
        if current:
            raw_blocks.append("\n".join(current))
    else:
        # Classic blank-line separation
        raw_blocks = re.split(r"\n{2,}", projects_text.strip())

    entries = []

    for block in raw_blocks:
        block_lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not block_lines:
            continue

        name = None
        description_lines = []
        tech_stack = []

        for i, line in enumerate(block_lines):
            # ── Check for explicit tech-label line ──────────────────────
            tech_match = tech_label_re.match(line)
            if tech_match:
                raw_tech = tech_match.group(1)
                tech_stack = [t.strip() for t in re.split(r"[,|;/]", raw_tech) if t.strip()]
                continue  # don't add to description

            # ── First line is the project name ──────────────────────────
            if i == 0:
                # Handle "Name – description" or "Name: description" on one line
                parts = separator_re.split(line, maxsplit=1)
                name = parts[0].strip()
                # Strip parenthesised tech from name if present
                name_clean = paren_tech_re.sub("", name).strip()
                if name_clean:
                    name = name_clean
                if len(parts) > 1:
                    rest = parts[1].strip()
                    # Try to pull tech from this inline description part
                    inline_tech = _extract_tech(rest)
                    if inline_tech and not tech_stack:
                        tech_stack = inline_tech
                    # Strip the "using …" clause before saving as description
                    rest_clean = using_re.sub("", rest).strip().strip(".")
                    if rest_clean:
                        description_lines.append(rest_clean)
            else:
                # Try to pull tech from description lines
                inline_tech = _extract_tech(line)
                if inline_tech and not tech_stack:
                    tech_stack = inline_tech
                # Keep line as description (minus any "using …" clause)
                desc_line = using_re.sub("", line).strip().strip(".")
                # Also strip trailing parenthesised tech
                desc_line = paren_tech_re.sub("", desc_line).strip()
                if desc_line:
                    description_lines.append(desc_line)

        # ── Also scan the whole block for tech if still empty ──────────
        if not tech_stack:
            tech_stack = _extract_tech(block)

        if name:
            entries.append(ProjectEntry(
                name=name,
                description=" ".join(description_lines) or None,
                tech_stack=tech_stack,
            ))

    return entries


def parse_certifications(cert_text: str) -> list:
    if not cert_text.strip():
        return []

    cleaned = re.sub(r"[•\-\|]", "\n", cert_text)
    certs = []
    for line in cleaned.splitlines():
        line = line.strip()
        if line and len(line) > 5:
            certs.append(line)
    return certs


def parse_summary(summary_text: str) -> Optional[str]:
    if not summary_text.strip():
        return None
    cleaned = re.sub(r"\s+", " ", summary_text).strip()
    return cleaned if cleaned else None


# ============================================================
# SECTION 5 — MASTER EXTRACTOR
# ============================================================

def extract_resume_data(text: str) -> ResumeData:
    email = extract_email(text)
    phone = extract_phone(text)
    linkedin = extract_linkedin(text)
    github = extract_github(text)

    name = extract_name(text)
    location = extract_location(text)

    sections = split_into_sections(text)

    summary = parse_summary(sections["summary"])
    skills = parse_skills(sections["skills"])
    experience = parse_experience(sections["experience"])
    education = parse_education(sections["education"])
    projects = parse_projects(sections["projects"])
    certifications = parse_certifications(sections["certifications"])

    return ResumeData(
        name=name,
        email=email,
        phone=phone,
        location=location,
        linkedin=linkedin,
        github=github,
        summary=summary,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        certifications=certifications,
    )