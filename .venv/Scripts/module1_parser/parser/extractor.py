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
# en_core_web_sm is a small English model that can detect:
#   PERSON, ORG (organization), GPE (city/country), DATE, etc.
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "spaCy model not found. Run: python -m spacy download en_core_web_sm"
    )


# ============================================================
# SECTION 1 — REGEX EXTRACTORS
# These use regular expression patterns to find specific data.
# ============================================================

def extract_email(text: str) -> Optional[str]:
    """
    Find an email address using regex.
    Pattern breakdown:
      [a-zA-Z0-9._%+-]+   → username (letters, digits, dots, etc.)
      @                   → literal @ sign
      [a-zA-Z0-9.-]+      → domain name
      \.[a-zA-Z]{2,}      → TLD like .com, .pk, .io
    """
    pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    match = re.search(pattern, text)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    """
    Find a phone number. Handles many formats:
      +92-300-1234567   (Pakistani)
      +1 (234) 567-8900 (US)
      0300 1234567      (local)
      03001234567       (no spaces)
    """
    pattern = r"(\+?\d[\d\s\-().]{7,}\d)"
    matches = re.findall(pattern, text)
    # Return the first match that's at least 7 digits long
    for match in matches:
        digits_only = re.sub(r"\D", "", match)
        if len(digits_only) >= 7:
            return match.strip()
    return None


def extract_linkedin(text: str) -> Optional[str]:
    """Find a LinkedIn profile URL."""
    pattern = r"(?:https?://)?(?:www\.)?linkedin\.com/in/[\w\-]+"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


def extract_github(text: str) -> Optional[str]:
    """Find a GitHub profile URL."""
    pattern = r"(?:https?://)?(?:www\.)?github\.com/[\w\-]+"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else None


# ============================================================
# SECTION 2 — SPACY NER EXTRACTORS
# spaCy reads the text and labels entities like PERSON, GPE.
# ============================================================

def extract_name(text: str) -> Optional[str]:
    """
    Use spaCy Named Entity Recognition to find the person's name.

    How it works:
    1. spaCy reads the first ~500 characters (name is usually at top)
    2. It labels each entity it finds: PERSON, ORG, DATE, etc.
    3. We return the first PERSON entity found.

    Limitation: spaCy sometimes misses names or tags them as ORG.
    The first 500 chars heuristic works for standard resume formats.
    """
    # Only look at the top of the resume for the name
    top_text = text[:500]
    doc = nlp(top_text)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text.strip()

    # Fallback: first non-empty line is often the name
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), None)
    # Only use if it looks like a name (2-4 words, no special chars)
    if first_line and len(first_line.split()) <= 4 and not re.search(r"[@/]", first_line):
        return first_line

    return None


def extract_location(text: str) -> Optional[str]:
    """
    Use spaCy NER to find a city or country (GPE = Geo-Political Entity).
    We scan the whole text and return the first GPE found.
    """
    doc = nlp(text[:1000])  # Location usually appears near the top
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.strip()
    return None


# ============================================================
# SECTION 3 — SECTION SPLITTER
# Resumes are divided into sections (SKILLS, EXPERIENCE, etc.)
# We split the text by these headers to isolate each block.
# ============================================================

# Common section header keywords. We match them case-insensitively.
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
    """
    Split resume text into labeled sections.

    Approach:
    1. Go line by line.
    2. Check if a line matches any known section header.
    3. Collect all lines until the next header into that section.

    Returns:
        A dict like: {"skills": "Python\nJava\n...", "experience": "..."}
    """
    sections = {key: "" for key in SECTION_HEADERS}
    current_section = "summary"  # Assume text before first header is summary
    lines = text.splitlines()

    for line in lines:
        stripped = line.strip().lower()

        # Check if this line is a section header
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
# Each function takes a section's raw text and returns clean data.
# ============================================================

def parse_skills(skills_text: str) -> list:
    """
    Extract a list of skills from the skills section.

    Skills are typically separated by:
      - Commas: "Python, Java, SQL"
      - Bullet points / newlines: each skill on its own line
      - Pipes: "Python | Java | SQL"

    We split on all of these and clean up the results.
    """
    if not skills_text.strip():
        return []

    # Replace bullets, pipes, semicolons with commas, then split
    cleaned = re.sub(r"[•\-\|;]", ",", skills_text)
    raw_skills = re.split(r"[,\n]", cleaned)

    skills = []
    for skill in raw_skills:
        skill = skill.strip()
        # Skip very short (noise) or very long (sentences) items
        if skill and 2 <= len(skill) <= 50:
            skills.append(skill)

    # Deduplicate while preserving order
    seen = set()
    unique_skills = []
    for s in skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_skills.append(s)

    return unique_skills


def parse_experience(experience_text: str) -> list:
    """
    Extract structured experience entries.

    This is the hardest part of resume parsing because the format
    varies so much. We use a heuristic approach:
    1. Split text into blocks separated by blank lines.
    2. For each block, try to detect title, company, duration.
    3. The rest becomes the description.

    Heuristics used:
    - Duration: looks for year ranges like "2020 - 2023" or "Jan 2021 – Present"
    - Title/Company: usually the first 1-2 lines of the block
    """
    if not experience_text.strip():
        return []

    # Split into blocks (separated by 1+ blank lines)
    blocks = re.split(r"\n{2,}", experience_text.strip())
    entries = []

    # Regex to detect date ranges
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
            # Check if this line contains a date range
            dur_match = duration_pattern.search(line)
            if dur_match and not duration:
                duration = dur_match.group(0).strip()
                # The rest of the line (minus the date) might be company or title
                remainder = line[:dur_match.start()].strip(" |,–-")
                if remainder and not company:
                    company = remainder
            elif i == 0 and not title:
                title = line
            elif i == 1 and not company:
                company = line
            else:
                description_lines.append(line)

        # Only add if we found at least one meaningful field
        if title or company:
            entries.append(ExperienceEntry(
                title=title,
                company=company,
                duration=duration,
                description=" ".join(description_lines) or None
            ))

    return entries


def parse_education(education_text: str) -> list:
    """
    Extract structured education entries.

    Format varies, but usually looks like:
      BS Computer Science
      University of Karachi
      2021

    Or: "BS Computer Science, University of Karachi, 2021"

    We split into blocks and try to detect degree, institution, year.
    """
    if not education_text.strip():
        return []

    blocks = re.split(r"\n{2,}", education_text.strip())
    entries = []

    year_pattern = re.compile(r"\b(19|20)\d{2}\b")

    # Keywords that suggest a degree
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
            # Check for year
            year_match = year_pattern.search(line)
            if year_match and not year:
                year = year_match.group(0)

            # Check for degree keywords
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
    Extract structured project entries.

    Typical format:
      Project Name
      Description of what it does
      Tech: Python, BERT, Flask

    We split by blank lines and extract name, description, tech stack.
    """
    if not projects_text.strip():
        return []

    blocks = re.split(r"\n{2,}", projects_text.strip())
    entries = []

    tech_pattern = re.compile(
        r"(?:tech(?:nologies)?(?:\s*used)?|stack|tools?|built with)\s*[:\-]?\s*(.*)",
        re.IGNORECASE
    )

    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if not lines:
            continue

        name = lines[0] if lines else None
        description_lines = []
        tech_stack = []

        for line in lines[1:]:
            tech_match = tech_pattern.match(line)
            if tech_match:
                # Parse the tech list from the matched portion
                tech_raw = tech_match.group(1)
                tech_stack = [t.strip() for t in re.split(r"[,|;]", tech_raw) if t.strip()]
            else:
                description_lines.append(line)

        # If no explicit tech line, try to find inline tech from description
        if not tech_stack and description_lines:
            combined = " ".join(description_lines)
            # Look for patterns like "using Python, Flask, and SQL"
            using_match = re.search(r"using\s+([\w\s,/\-\.]+)", combined, re.IGNORECASE)
            if using_match:
                tech_raw = using_match.group(1)
                tech_stack = [t.strip() for t in re.split(r"[,and ]+", tech_raw)
                               if t.strip() and len(t.strip()) > 1]

        if name:
            entries.append(ProjectEntry(
                name=name,
                description=" ".join(description_lines) or None,
                tech_stack=tech_stack
            ))

    return entries


def parse_certifications(cert_text: str) -> list:
    """
    Extract a list of certification names.
    Each line (or comma-separated item) is one certification.
    """
    if not cert_text.strip():
        return []

    cleaned = re.sub(r"[•\-\|]", "\n", cert_text)
    certs = []

    for line in cleaned.splitlines():
        line = line.strip()
        if line and len(line) > 5:  # Skip very short noise
            certs.append(line)

    return certs


def parse_summary(summary_text: str) -> Optional[str]:
    """
    Clean and return the summary/objective section.
    Just strip whitespace and return as a single paragraph.
    """
    if not summary_text.strip():
        return None
    # Collapse multiple spaces/newlines into single spaces
    cleaned = re.sub(r"\s+", " ", summary_text).strip()
    return cleaned if cleaned else None


# ============================================================
# SECTION 5 — MASTER EXTRACTOR
# The single function called by main.py
# ============================================================

def extract_resume_data(text: str) -> ResumeData:
    """
    Master function: takes raw resume text, returns a ResumeData object.

    Process:
    1. Run regex extractors on the full text (email, phone, URLs)
    2. Run spaCy extractors on the full text (name, location)
    3. Split text into sections
    4. Run section parsers on each block
    5. Assemble everything into a ResumeData object
    """
    # ── Step 1: Regex fields (scan full text) ───────────────
    email = extract_email(text)
    phone = extract_phone(text)
    linkedin = extract_linkedin(text)
    github = extract_github(text)

    # ── Step 2: NER fields (spaCy on full text) ─────────────
    name = extract_name(text)
    location = extract_location(text)

    # ── Step 3: Split text into labeled sections ─────────────
    sections = split_into_sections(text)

    # ── Step 4: Parse each section ───────────────────────────
    summary = parse_summary(sections["summary"])
    skills = parse_skills(sections["skills"])
    experience = parse_experience(sections["experience"])
    education = parse_education(sections["education"])
    projects = parse_projects(sections["projects"])
    certifications = parse_certifications(sections["certifications"])

    # ── Step 5: Assemble and return ──────────────────────────
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
