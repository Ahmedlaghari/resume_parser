# ============================================================
# extractor.py — LLM-powered field-by-field extraction from JD text.
#
# Replaces all regex, spaCy NER, and section-splitting logic with
# a single structured LLM call via instructor + Groq.
#
# JobDescription is defined in models.py (the Module 2 ↔ 3 contract).
# This module only populates the fields it owns — seniority_level,
# required_skills, nice_to_have_skills, and keywords are left as
# defaults for seniority.py, skill_classifier.py, and KeyBERT.
#
# Dependencies:
#   pip install groq instructor pydantic python-dotenv
#
# Usage:
#   from analyzer.extractor import extract_jd_data
#   data = extract_jd_data(raw_jd_text)
# ============================================================

import os
import re

from dotenv import load_dotenv
from groq import Groq
import instructor

from .models import JobDescription

load_dotenv()

# ── Instructor-patched Groq client (created once at import time) ──────────
_client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON,
)


# ============================================================
# SECTION 1 — TEXT PREPARATION
# ============================================================

def _prepare_text(raw: str) -> str:
    """Minimal cleanup — removes noise without altering meaning."""
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text.strip()


# ============================================================
# SECTION 2 — PROMPT
# ============================================================

_SYSTEM_PROMPT = """\
You are a precise job-description parser. Extract all available fields from \
the job posting provided and return them as a structured JSON object matching \
the schema exactly.

Rules:
- Only populate these fields — leave everything else as null or empty list:
  job_title, company, location, employment_type, experience_required,
  salary_range, responsibilities, qualifications, benefits, industry.
- If a field is not present in the JD, return null (strings) or [] (lists).
- For list fields, return each bullet point as a separate string.
- Preserve original wording; do not rephrase or summarise.
- Do not infer or fabricate values not explicitly stated in the text.
"""


# ============================================================
# SECTION 3 — MAIN EXTRACTOR
# ============================================================

def extract_jd_data(
    raw_text: str,
    *,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 2048,
    max_retries: int = 2,
) -> JobDescription:
    """
    Parse a raw job description string into a validated JobDescription object.

    Only populates extractor-owned fields. The caller is responsible for
    filling seniority_level, required_skills, nice_to_have_skills, and
    keywords via the respective downstream modules.

    Args:
        raw_text:    Full JD as plain text.
        model:       Groq model to use.
        max_tokens:  Max tokens for the LLM response.
        max_retries: Retry attempts if Pydantic validation fails.

    Returns:
        A partially populated JobDescription instance.

    Raises:
        instructor.exceptions.InstructorRetryException: on repeated invalid JSON.
        groq.APIError: on network or authentication failures.
    """
    text = _prepare_text(raw_text)

    result: JobDescription = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Parse this job description:\n\n{text}"},
        ],
        response_model=JobDescription,
    )

    print("\n========== JD EXTRACTOR OUTPUT ==========")
    print(result.model_dump_json(indent=2))
    print("=========================================\n")

    return result


# ============================================================
# SECTION 4 — QUICK TEST
# ============================================================

if __name__ == "__main__":
    SAMPLE = """
    Senior ML Engineer — AI Platform Team
    at DataSphere (Remote / Karachi, Pakistan)

    We are hiring a Senior ML Engineer to join our AI Platform team.
    Full-time | PKR 250,000 – 350,000/month

    About DataSphere
    DataSphere is a SaaS company building next-gen analytics and AI tooling.

    Responsibilities
    - Design and deploy production ML pipelines serving millions of requests.
    - Mentor junior engineers on MLOps best practices.
    - Collaborate with product and data teams to define model requirements.

    Requirements
    - 5+ years of experience in Machine Learning or Data Science.
    - Strong proficiency in Python and PyTorch or TensorFlow.
    - Experience with Kubernetes and MLflow.
    - BSc/MSc in Computer Science or related field.

    Nice to Have
    - Familiarity with LangChain or other LLM frameworks.
    - Experience with AWS SageMaker or Vertex AI.

    Benefits
    - Competitive salary + equity.
    - Fully remote with quarterly team meet-ups.
    - Health insurance and 20 days paid leave.
    """

    data = extract_jd_data(SAMPLE)
    print(data.model_dump_json(indent=2))