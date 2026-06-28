# ============================================================
# skill_classifier.py — LLM-powered skill extraction + classification.
#
# Replaces the SKILLS_DB keyword scan and context-window heuristics
# with a single structured LLM call via instructor + Groq.
#
# The model is asked to:
#   1. Identify every technical and process skill mentioned in the JD.
#   2. Classify each as "required" or "nice_to_have" based on context.
#
# Dependencies:
#   pip install groq instructor pydantic python-dotenv
#
# Usage:
#   from analyzer.skill_classifier import extract_and_classify_skills
#   req, nice = extract_and_classify_skills(clean_text)
# ============================================================

import os
import re
import logging
from typing import Literal

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
# SECTION 1 — PYDANTIC MODELS
# ============================================================

class SkillEntry(BaseModel):
    name: str = Field(
        description=(
            "The skill name exactly as it should be displayed, properly cased. "
            "Examples: 'Python', 'PyTorch', 'REST API', 'CI/CD', 'AWS', 'Docker'."
        )
    )
    classification: Literal["required", "nice_to_have"] = Field(
        description=(
            "Whether the skill is required or nice-to-have. "
            "Mark as 'required' if the surrounding text uses words like: "
            "'must', 'required', 'essential', 'mandatory', 'minimum', "
            "'you have', 'you must', 'we require'. "
            "Mark as 'nice_to_have' if the surrounding text uses words like: "
            "'preferred', 'nice to have', 'bonus', 'plus', 'advantage', "
            "'desirable', 'ideally', 'familiarity with', 'exposure to', "
            "'beneficial', 'good to have'. "
            "Default to 'required' when context is ambiguous."
        )
    )


class SkillsResult(BaseModel):
    skills: list[SkillEntry] = Field(
        default_factory=list,
        description=(
            "Complete list of all technical and process skills found in the JD. "
            "Include: programming languages, frameworks, libraries, cloud platforms, "
            "DevOps tools, databases, ML/AI tools, methodologies (Agile, Scrum), "
            "and soft/process skills explicitly listed as requirements. "
            "Do NOT include vague traits like 'communication skills' or "
            "'problem-solving' unless they are listed as explicit requirements."
        ),
    )


# ============================================================
# SECTION 2 — PROMPT
# ============================================================

_SYSTEM_PROMPT = """\
You are a precise technical skill extractor for job descriptions.

Your task:
1. Read the job description carefully.
2. Identify every concrete technical and process skill mentioned.
3. For each skill, decide whether it is "required" or "nice_to_have" \
   based on the surrounding language.
4. Return the results as a structured JSON object.

Rules:
- Include each unique skill only once (deduplicate).
- Use consistent, properly-cased display names (e.g. 'PyTorch', not 'pytorch').
- Do not include soft traits like 'teamwork' or 'communication' \
  unless the JD explicitly lists them as requirements.
- Do not infer skills not mentioned in the text.
"""


# ============================================================
# SECTION 3 — MAIN FUNCTION
# ============================================================

def extract_and_classify_skills(
    text: str,
    *,
    model: str = "llama-3.3-70b-versatile",
    max_tokens: int = 1024,
    max_retries: int = 2,
) -> tuple[list[str], list[str]]:
    """
    Extract and classify all skills mentioned in a job description.

    Args:
        text:        Full cleaned JD text.
        model:       Groq model to use.
        max_tokens:  Max tokens for the LLM response.
        max_retries: Retry attempts if Pydantic validation fails.

    Returns:
        (required_skills, nice_to_have_skills) — two lists of skill name strings.

    Usage:
        from analyzer.skill_classifier import extract_and_classify_skills
        req, nice = extract_and_classify_skills(clean_text)
    """
    # Strip control chars before sending
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).strip()

    result: SkillsResult = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract and classify skills from this JD:\n\n{clean}"},
        ],
        response_model=SkillsResult,
    )

    required = [s.name for s in result.skills if s.classification == "required"]
    nice_to_have = [s.name for s in result.skills if s.classification == "nice_to_have"]

    logger.debug("Skill classifier: required=%s  nice_to_have=%s", required, nice_to_have)

    return required, nice_to_have


# ============================================================
# SECTION 4 — QUICK TEST
# ============================================================

if __name__ == "__main__":
    SAMPLE = """
    Senior ML Engineer — AI Platform

    Requirements
    - Must have 5+ years of Python experience.
    - Strong proficiency in PyTorch or TensorFlow is required.
    - Experience with Kubernetes and Docker is essential.
    - Familiarity with SQL and NoSQL databases.
    - AWS or GCP certification mandatory.

    Nice to Have
    - Exposure to LangChain or LlamaIndex is a bonus.
    - Familiarity with Airflow preferred.
    - Knowledge of React for internal tooling dashboards.
    - Experience with Agile / Scrum is desirable.
    """

    req, nice = extract_and_classify_skills(SAMPLE)
    print("Required:", req)
    print("Nice-to-have:", nice)
