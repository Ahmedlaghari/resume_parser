# ============================================================
# extractor_llm.py — LLM-powered structured resume extraction
# ============================================================
# Replaces all regex, spaCy NER, and section-splitting logic
# from the old extractor.py with a single LLM API call.
#
# Dependencies:
#   pip install anthropic instructor pydantic
#
# Usage:
#   from parser.extractor_llm import extract_resume_data
#   data = extract_resume_data(raw_text)
# ============================================================

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env and puts variables into os.environ

from groq import Groq
import re
import instructor

_client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON
)

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ============================================================
# SECTION 1 — PYDANTIC MODELS
# ============================================================
# These replace the old dataclass/TypedDict models in models.py.
# Field descriptions are load-bearing — instructor injects them
# into the LLM prompt as part of the JSON schema, so the model
# knows exactly what to populate.
# ============================================================

class ExperienceEntry(BaseModel):
    title: Optional[str] = Field(None, description="Job title or role, e.g. 'Senior Software Engineer'")
    company: Optional[str] = Field(None, description="Employer or organisation name")
    duration: Optional[str] = Field(None, description="Date range, e.g. 'Jan 2020 – Present' or '2018–2021'")
    description: Optional[str] = Field(None, description="Responsibilities, achievements, and technologies used")


class EducationEntry(BaseModel):
    degree: Optional[str] = Field(None, description="Degree and field, e.g. 'B.S. Computer Science'")
    institution: Optional[str] = Field(None, description="University or school name")
    year: Optional[str] = Field(None, description="Graduation year or date range")


class ProjectEntry(BaseModel):
    name: Optional[str] = Field(None, description="Project name or title")
    description: Optional[str] = Field(None, description="What the project does and what problem it solves")
    tech_stack: list[str] = Field(
        default_factory=list,
        description="Technologies, frameworks, and tools used, e.g. ['Python', 'FastAPI', 'PostgreSQL']"
    )


class ResumeData(BaseModel):
    # ── Contact & identity ─────────────────────────────────────────────────
    name: Optional[str] = Field(None, description="Full name of the candidate")
    email: Optional[str] = Field(None, description="Primary email address")
    phone: Optional[str] = Field(None, description="Phone number, preserving original formatting")
    location: Optional[str] = Field(None, description="City, state, or country. Not the full mailing address.")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL or handle")
    github: Optional[str] = Field(None, description="GitHub profile URL or handle")

    # ── Content sections ───────────────────────────────────────────────────
    summary: Optional[str] = Field(None, description="Professional summary or objective statement")
    skills: list[str] = Field(
        default_factory=list,
        description="Flat list of individual skills. Split 'Python, Django, React' into ['Python', 'Django', 'React']."
    )
    experience: list[ExperienceEntry] = Field(
        default_factory=list,
        description="Work experience entries, most recent first"
    )
    education: list[EducationEntry] = Field(
        default_factory=list,
        description="Education entries, most recent first"
    )
    projects: list[ProjectEntry] = Field(
        default_factory=list,
        description="Personal, academic, or open-source projects"
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Certification names, e.g. 'AWS Certified Solutions Architect – 2022'"
    )

    # ── Field-level validators ─────────────────────────────────────────────
    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Discard values that don't look like email addresses."""
        if v and not re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", v):
            return None
        return v

    @field_validator("skills", mode="before")
    @classmethod
    def coerce_skills_to_list(cls, v):
        """
        Handle the case where the LLM returns skills as a single
        comma-separated string instead of a list.
        """
        if isinstance(v, str):
            return [s.strip() for s in re.split(r"[,;|]", v) if s.strip()]
        return v


# ============================================================
# SECTION 2 — TEXT PREPARATION
# ============================================================

def prepare_resume_text(raw: str) -> str:
    """
    Minimal cleanup before sending to the LLM.
    Does NOT parse, split, or extract — just removes noise
    that wastes tokens without adding meaning.
    """
    # Collapse 3+ consecutive blank lines down to 2
    text = re.sub(r"\n{3,}", "\n\n", raw)

    # Strip null bytes and non-printable control characters
    # that sometimes appear after PDF extraction
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


# ============================================================
# SECTION 3 — PROMPT
# ============================================================

SYSTEM_PROMPT = """\
You are a precise resume parser. Extract all available information from the \
resume text provided and return it as a structured JSON object matching the \
schema exactly.

Rules:
- If a field is not present in the resume, return null or an empty list.
- For skills, return individual skill names as separate list items — do not \
  return a single comma-separated string.
- For experience and education, preserve the original wording; do not rephrase.
- For duration/dates, keep the original format from the resume.
- Do not infer or fabricate values that are not explicitly stated.
"""


# ============================================================
# SECTION 4 — MASTER EXTRACTOR
# ============================================================

# Build the instructor-patched Anthropic client once at import time.
# instructor handles:
#   1. Converting ResumeData's JSON schema into a tool definition
#   2. Parsing the model's JSON response back into a ResumeData instance
#   3. Retrying with the validation error if Pydantic rejects the output



def extract_resume_datallm(
    raw_text: str,
    *,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 2048,
    max_retries: int = 2,
) -> ResumeData:
    """
    Parse a raw resume string into a validated ResumeData object.

    Args:
        raw_text:    The full resume as plain text (from PDF/DOCX extraction).
        model:       Anthropic model to use.
        max_tokens:  Max tokens for the LLM response.
        max_retries: How many times instructor will retry on validation error.

    Returns:
        A fully validated ResumeData instance. Fields not found in the resume
        will be None or empty lists — never missing keys.

    Raises:
        instructor.exceptions.InstructorRetryException: if the model still
            returns invalid JSON after max_retries attempts.
        anthropic.APIError: on network or authentication failures.
    """
    text = prepare_resume_text(raw_text)

    resume: ResumeData = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        messages=[
            {
                "role": "system",  # ← system goes here as a message role
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Parse this resume:\n\n{text}",
            }
        ],
        response_model=ResumeData,
    )
    print("\n========== LLM EXTRACTOR OUTPUT ==========")
    print(resume.model_dump_json(indent=2))
    print("==========================================\n")

    return resume


# ============================================================
# SECTION 5 — QUICK TEST (run this file directly to verify)
# ============================================================

if __name__ == "__main__":
    SAMPLE = """
Jane Smith
jane.smith@email.com | +92 300 1234567 | linkedin.com/in/janesmith
Karachi, Pakistan

SUMMARY
Backend engineer with 4 years of experience building APIs and data pipelines.

SKILLS
Python; FastAPI; PostgreSQL; Redis | Docker, Kubernetes
AWS (Lambda, S3, RDS)

EXPERIENCE

Senior Backend Engineer | TechCo | Mar 2022 – Present
  - Designed REST APIs serving 2M+ requests/day
  - Reduced DB query latency by 35% via query optimisation and caching

Backend Engineer
StartupXYZ, Karachi
2020 – 2022
Built internal tooling using Flask and Celery. Integrated with Stripe and Twilio.

EDUCATION
B.E. Software Engineering
NED University, Karachi — 2020

PROJECTSS
Real-time Chat App
Built with FastAPI + WebSockets + Redis pub/sub. Supports 500 concurrent users.

Inventory Management System — Django, PostgreSQL, Celery
Automated purchase order generation, reducing manual work by 60%.

CERTIFICATIONS
AWS Certified Developer – Associate, 2023
    """

    result = extract_resume_datallm(SAMPLE)
    print(result.model_dump_json(indent=2))
