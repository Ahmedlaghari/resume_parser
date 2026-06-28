# ============================================================
# seniority.py — LLM-powered seniority level detection from JD text.
#
# Replaces the keyword-scan + experience-year fallback heuristics
# with a single structured LLM call via instructor + Groq.
#
# Returns one of: "Junior" | "Mid" | "Senior" | "Lead" | "Principal" | None
#
# Dependencies:
#   pip install groq instructor pydantic python-dotenv
#
# Usage:
#   from analyzer.seniority import detect_seniority
#   level = detect_seniority(clean_text)
# ============================================================

import os
import re
import logging
from typing import Literal, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
from groq import Groq
import instructor
from pydantic import BaseModel, Field

load_dotenv()

# ── Instructor-patched Groq client ────────────────────────────────────────
_client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON,
)


# ============================================================
# SECTION 1 — PYDANTIC MODEL
# ============================================================

SeniorityLevel = Literal["Junior", "Mid", "Senior", "Lead", "Principal"]


class SeniorityResult(BaseModel):
    level: Optional[SeniorityLevel] = Field(
        None,
        description=(
            "The seniority level of the role. Choose exactly one of: "
            "'Junior', 'Mid', 'Senior', 'Lead', 'Principal'. "
            "Return null if there is genuinely no signal. "
            "\n\nGuidelines:"
            "\n- Junior:    entry-level, graduate, associate, 0–2 years exp."
            "\n- Mid:       intermediate, mid-level, 2–5 years exp."
            "\n- Senior:    'senior', 'sr.', 5+ years exp., owns significant scope."
            "\n- Lead:      tech lead, team lead, engineering lead, lead engineer — "
            "manages or coordinates other engineers."
            "\n- Principal: principal engineer, staff engineer, distinguished, "
            "fellow — org-wide technical leadership."
            "\n\nPriority: explicit title keywords > years of experience > "
            "implied responsibility level."
        ),
    )
    reasoning: str = Field(
        description=(
            "One sentence explaining which signal (title keyword, experience "
            "years, or responsibility scope) determined the level."
        )
    )


# ============================================================
# SECTION 2 — PROMPT
# ============================================================

_SYSTEM_PROMPT = """\
You are an expert at analysing job descriptions and determining the seniority \
level of the role being advertised.

Given a job description, identify the seniority level using these signals \
(in order of priority):
1. Explicit title keywords: "junior", "senior", "lead", "principal", "staff", etc.
2. Years of experience required: 0–2 → Junior, 2–5 → Mid, 5–8 → Senior, 8+ → Lead.
3. Implied scope: managing teams → Lead; org-wide decisions → Principal.

Return a structured JSON with the level and a brief reasoning sentence.
Return null for level only if there is absolutely no signal in the text.
"""


# ============================================================
# SECTION 3 — MAIN FUNCTION
# ============================================================

def detect_seniority(
    text: str,
    experience_required: Optional[str] = None,
    *,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 256,
    max_retries: int = 2,
) -> Optional[SeniorityLevel]:
    """
    Detect the seniority level of a role from a job description.

    Args:
        text:                Full cleaned JD text.
        experience_required: Optional pre-extracted experience string
                             (e.g. "3-5 years") to include as a hint.
                             Saves the LLM from re-parsing it.
        model:               Groq model to use.
        max_tokens:          Max tokens for the LLM response.
        max_retries:         Retry attempts if Pydantic validation fails.

    Returns:
        One of "Junior" | "Mid" | "Senior" | "Lead" | "Principal", or None.

    Usage:
        from analyzer.seniority import detect_seniority
        level = detect_seniority(clean_text, experience_str)
    """
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()

    # Optionally inject the pre-extracted experience string as a hint
    hint = (
        f"\n\n[Pre-extracted experience requirement: {experience_required}]"
        if experience_required
        else ""
    )

    result: SeniorityResult = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Determine the seniority level of this role:{hint}\n\n{clean}",
            },
        ],
        response_model=SeniorityResult,
    )

    logger.debug("Seniority detector: level=%s  reasoning=%s", result.level, result.reasoning)

    return result.level


# ============================================================
# SECTION 4 — QUICK TEST
# ============================================================

if __name__ == "__main__":
    tests = [
        (
            "Junior Python Developer\nWe are hiring a fresh graduate or someone "
            "with 0-2 years of experience to join our team.",
            None,
        ),
        (
            "Senior ML Engineer\nRequirements: 5+ years of experience in machine "
            "learning. Strong Python skills. Experience with PyTorch.",
            "5+ years",
        ),
        (
            "Tech Lead – Platform Engineering\nYou will lead a team of 6 engineers, "
            "define technical roadmap, and own cross-team architecture decisions.",
            None,
        ),
        (
            "Principal Engineer\nWe are looking for a distinguished engineer to set "
            "org-wide technical strategy across all product lines.",
            None,
        ),
        (
            "Software Engineer\nWe need someone with 3-5 years of backend experience "
            "to build APIs and maintain our data pipelines.",
            "3-5 years",
        ),
    ]

    for jd_text, exp in tests:
        level = detect_seniority(jd_text, exp)
        print(f"Input snippet: {jd_text[:60]}…")
        print(f"Result: {level}\n")
