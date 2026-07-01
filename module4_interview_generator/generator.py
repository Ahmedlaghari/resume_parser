# ============================================================
# generator.py — LLM personalization step.
#
# Takes the raw question pool from retriever.py + full candidate
# context and asks the LLM to:
#   1. Select the best 10-12 questions from the pool
#   2. Personalize each — replace [SKILL], [COMPANY], [PROJECT]
#      with the candidate's actual details
#   3. Return structured JSON via instructor
#
# Uses the same instructor + Groq pattern as Modules 1-3.
# ============================================================

import os
import logging
from typing import Literal

from dotenv import load_dotenv
from groq import Groq
import instructor
from pydantic import BaseModel, Field

load_dotenv()
logger = logging.getLogger(__name__)

_client = instructor.from_groq(
    Groq(api_key=os.getenv("GROQ_API_KEY")),
    mode=instructor.Mode.JSON,
)

DEFAULT_MODEL = "llama-3.3-70b-versatile"


# ============================================================
# SECTION 1 — OUTPUT SCHEMA
# ============================================================

class InterviewQuestion(BaseModel):
    category: Literal["technical", "experience", "gap", "behavioral"] = Field(
        description="The question category"
    )
    question: str = Field(
        description=(
            "The personalized interview question. Replace every [SKILL], [COMPANY], "
            "[PROJECT], [MATCHED_SKILL], [MISSING_SKILL] placeholder with the "
            "candidate's actual details from their resume."
        )
    )
    what_to_listen_for: str = Field(
        description=(
            "What the recruiter should listen for in the candidate's answer. "
            "Keep this concise and specific to this candidate."
        )
    )


class InterviewQuestionSet(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate")
    job_title: str = Field(description="The job title from the job description")
    questions: list[InterviewQuestion] = Field(
        description=(
            "10 to 12 personalized interview questions. Must include at least "
            "2 from each category: technical, experience, gap, behavioral."
        ),
        min_length=8,
        max_length=14,
    )


# ============================================================
# SECTION 2 — CONTEXT BUILDER
# ============================================================

def _build_candidate_summary(candidate: dict, match_result: dict) -> str:
    """Compress candidate data into a dense text block for the prompt."""
    name = candidate.get("name", "Candidate")
    years = candidate.get("total_experience_years", "unknown")
    skills = candidate.get("skills", [])
    experience = candidate.get("experience", [])
    projects = candidate.get("projects", [])
    matched = match_result.get("matched_skills", [])
    missing = match_result.get("missing_skills", [])
    score = match_result.get("final_score", match_result.get("score", "N/A"))

    exp_lines = []
    for e in experience[:3]:
        title = e.get("title", "")
        company = e.get("company", "")
        duration = e.get("duration", "")
        exp_lines.append(f"  - {title} at {company} ({duration})")

    proj_lines = []
    for p in projects[:3]:
        pname = p.get("name", "")
        stack = ", ".join(p.get("tech_stack", []))
        proj_lines.append(f"  - {pname}" + (f" [{stack}]" if stack else ""))

    return f"""CANDIDATE: {name}
Experience: {years} years
Skills: {', '.join(skills[:15])}
Match score: {score}/100
Matched skills (they HAVE these): {', '.join(matched)}
Missing skills (they DON'T have these): {', '.join(missing)}

Work History:
{chr(10).join(exp_lines) if exp_lines else '  (none listed)'}

Projects:
{chr(10).join(proj_lines) if proj_lines else '  (none listed)'}"""


def _build_prompt(
    candidate: dict,
    jd: dict,
    match_result: dict,
    question_pool: list[dict],
) -> str:
    candidate_block = _build_candidate_summary(candidate, match_result)
    role = jd.get("job_title", "the role")
    jd_skills = jd.get("required_skills", jd.get("qualifications", []))[:8]
    jd_skills_str = ", ".join(jd_skills) if jd_skills else "see job description"

    pool_lines = []
    for i, q in enumerate(question_pool, 1):
        pool_lines.append(
            f"{i}. [{q['category'].upper()}] (id={q['id']})\n"
            f"   Q: {q['text']}\n"
            f"   Listen for: {q['what_to_listen_for']}"
        )
    pool_block = "\n\n".join(pool_lines)

    return f"""You are preparing a recruiter for an interview with the candidate below.

{candidate_block}

ROLE THEY ARE INTERVIEWING FOR: {role}
KEY ROLE REQUIREMENTS: {jd_skills_str}

---

You have been given a pool of {len(question_pool)} candidate questions retrieved from a question bank.
Your job is to:
1. Select the best 10-12 questions from the pool that are most relevant to THIS candidate.
2. Personalize each selected question — replace every placeholder like [SKILL], [COMPANY],
   [PROJECT], [MATCHED_SKILL], [MISSING_SKILL], [CLOUD_PROVIDER] with the candidate's
   actual details (their real company names, real skills, real project names from above).
3. Make sure your final set has at least 2 questions from each category:
   technical, experience, gap, behavioral.
4. Update the "what_to_listen_for" to be specific to this candidate where relevant.

QUESTION POOL:
{pool_block}

Return the final personalized question set as structured JSON."""


# ============================================================
# SECTION 3 — MAIN GENERATOR
# ============================================================

def generate(
    candidate: dict,
    jd: dict,
    match_result: dict,
    question_pool: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 3000,
    max_retries: int = 2,
) -> InterviewQuestionSet:
    """
    Personalize the retrieved question pool for this specific candidate.

    Args:
        candidate:      Module 1 resume JSON
        jd:             Module 2 JD JSON
        match_result:   Module 3 result for this candidate
        question_pool:  Output of retriever.retrieve()

    Returns:
        InterviewQuestionSet — validated Pydantic object
    """
    prompt = _build_prompt(candidate, jd, match_result, question_pool)

    result: InterviewQuestionSet = _client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        max_retries=max_retries,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert technical recruiter. You create highly personalized, "
                    "specific interview questions based on a candidate's actual resume details. "
                    "Never leave placeholder text like [SKILL] or [COMPANY] in your output — "
                    "always replace them with the candidate's real information."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_model=InterviewQuestionSet,
    )

    logger.debug("Generator output:\n%s", result.model_dump_json(indent=2))
    return result
