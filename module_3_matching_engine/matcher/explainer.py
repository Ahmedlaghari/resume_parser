# ============================================================
# explainer.py — LLM-powered plain-English explanation of a score.
#
# The ONLY place this module calls an LLM. Everything that determines
# the actual final_score happens in scorer.py/skills_matcher.py/etc.
# and is fully deterministic. This file just turns an already-computed
# score breakdown into a couple of human-readable sentences.
#
# Why split it like this: if a candidate (or a regulator) asks "why did
# this person score 48.5?", you can answer with the real arithmetic,
# not "the AI said so." The LLM here is just translating numbers you
# already trust into English -- it can't change the ranking.
#
# Uses the same instructor + Groq pattern as analyzer/extractor.py:
# response_model gives back a validated Pydantic object directly, so
# there's no manual json.loads / fence-stripping to maintain.
#
# Dependencies:
#   pip install groq instructor pydantic python-dotenv
# ============================================================

import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq
import instructor
from pydantic import BaseModel, Field, field_validator

from .models import ScoreBreakdown, SkillMatchDetail

load_dotenv()

DEFAULT_MODEL = "llama-3.3-70b-versatile"  # same model extractor.py uses; swap to a smaller Groq model to cut latency/cost


@lru_cache(maxsize=1)
def _get_client():
    """
    Builds the instructor-patched Groq client lazily, on first use, and
    memoizes it (same one-time-init idea as extractor.py's module-level
    _client, but deferred).

    Why deferred and not built at import time like extractor.py does:
    Groq() raises immediately if GROQ_API_KEY is missing/empty -- it
    doesn't wait until you actually make a call. If this module built
    its client eagerly at import time, `from matcher.explainer import
    generate_explanation` would crash on startup for anyone who hasn't
    set a key yet, which would take down the whole FastAPI app (main.py
    imports this at module level) even for requests that don't need an
    explanation. Building it lazily means the error only surfaces inside
    generate_explanation's try/except below, where it's caught and
    routed to the deterministic fallback instead.
    """
    return instructor.from_groq(
        Groq(api_key=os.getenv("GROQ_API_KEY")),
        mode=instructor.Mode.JSON,
    )


# ============================================================
# SECTION 1 — OUTPUT SCHEMA
# ============================================================

class ExplanationResult(BaseModel):
    """What we ask the LLM to return -- instructor validates this directly,
    so a malformed response triggers an automatic retry instead of a
    json.loads crash."""

    one_line_reason: str = Field(
        description="One sentence, <=20 words, mentions the most impactful "
        "matched and/or missing skill by weight."
    )
    explanation: str = Field(
        description="2-3 sentences, plain English, references the highest-weight "
        "matches/gaps and the weakest sub-score."
    )

    @field_validator("one_line_reason")
    @classmethod
    def _keep_it_short(cls, v: str) -> str:
        words = v.split()
        if len(words) > 30:  # hard ceiling beyond the requested 20, in case the model overshoots
            v = " ".join(words[:30]) + "..."
        return v


# ============================================================
# SECTION 2 — PROMPT
# ============================================================

_SYSTEM_PROMPT = """\
You are writing for a recruiter screening dashboard. Be concise and factual -- \
do not invent any information beyond what's given in the candidate's score \
breakdown and skill match details. Always mention the highest-weight matched \
and/or missing skills, since those drove the score the most.
"""


def _build_user_message(
    candidate_name: str,
    job_title: str,
    final_score: float,
    breakdown: ScoreBreakdown,
    skill_match_detail: list[SkillMatchDetail],
) -> str:
    missing = [s for s in skill_match_detail if not s.matched]
    matched = [s for s in skill_match_detail if s.matched]

    return f"""Candidate: {candidate_name}
Job: {job_title}
Final score: {final_score}/100

Score breakdown:
- Skills: {breakdown.skills_score}/100
- Experience: {breakdown.experience_score}/100
- Education: {breakdown.education_score}/100
- Semantic/overall fit: {breakdown.semantic_similarity_score}/100

Matched skills (weight, skill): {[(s.weight, s.skill) for s in matched]}
Missing skills (weight, skill): {[(s.weight, s.skill) for s in missing]}
(Higher weight = more important to this role)"""


# ============================================================
# SECTION 3 — MAIN EXPLAINER
# ============================================================

def generate_explanation(
    candidate_name: str,
    job_title: str,
    final_score: float,
    breakdown: ScoreBreakdown,
    skill_match_detail: list[SkillMatchDetail],
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 300,
    max_retries: int = 2,
) -> tuple[str, str]:
    """
    Returns (one_line_reason, explanation). Falls back to a template-based
    explanation if the API call fails (no key configured, network error,
    repeated validation failure, etc.) so the rest of the pipeline never
    breaks because of this optional step.
    """
    try:
        client = _get_client()
        result: ExplanationResult = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            max_retries=max_retries,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_message(
                        candidate_name, job_title, final_score, breakdown, skill_match_detail
                    ),
                },
            ],
            response_model=ExplanationResult,
        )
        return result.one_line_reason, result.explanation

    except Exception as e:
        return _fallback_explanation(candidate_name, breakdown, skill_match_detail, error=e)


# ============================================================
# SECTION 4 — DETERMINISTIC FALLBACK
# ============================================================

def _fallback_explanation(
    candidate_name: str,
    breakdown: ScoreBreakdown,
    skill_match_detail: list[SkillMatchDetail],
    error: Exception | None = None,
) -> tuple[str, str]:
    """Deterministic, template-based fallback -- used if the LLM call isn't available."""
    missing = sorted([s for s in skill_match_detail if not s.matched], key=lambda s: -s.weight)
    matched = sorted([s for s in skill_match_detail if s.matched], key=lambda s: -s.weight)

    top_matched = matched[0].skill if matched else None
    top_missing = missing[0].skill if missing else None

    if top_missing:
        one_liner = f"Missing {top_missing} (weight {missing[0].weight}); strongest area: skills={breakdown.skills_score}."
    elif top_matched:
        one_liner = f"Has {top_matched} and all required skills; skills_score={breakdown.skills_score}."
    else:
        one_liner = f"No required skills specified; overall score driven by experience/education."

    explanation = (
        f"{candidate_name} scored {breakdown.skills_score}/100 on skills, "
        f"{breakdown.experience_score}/100 on experience, {breakdown.education_score}/100 on education, "
        f"and {breakdown.semantic_similarity_score}/100 on overall semantic fit. "
        + (f"Note: explanation generated via fallback template, not LLM ({type(error).__name__})." if error else "")
    )
    return one_liner, explanation.strip()


# ============================================================
# SECTION 5 — QUICK TEST
# ============================================================

if __name__ == "__main__":
    from .models import SkillMatchDetail as _SMD

    sample_breakdown = ScoreBreakdown(
        skills_score=82.14, experience_score=85, education_score=80, semantic_similarity_score=88
    )
    sample_details = [
        _SMD(skill="Python", weight=1.0, matched=True, match_score=1.0, candidate_has="Python"),
        _SMD(skill="Java", weight=0.5, matched=False, match_score=0.0, candidate_has=None),
        _SMD(skill="Docker", weight=0.7, matched=True, match_score=1.0, candidate_has="Docker"),
    ]

    one_liner, explanation = generate_explanation(
        candidate_name="Jane Doe",
        job_title="Machine Learning Engineer",
        final_score=87.5,
        breakdown=sample_breakdown,
        skill_match_detail=sample_details,
    )
    print("ONE LINE REASON:", one_liner)
    print("EXPLANATION:", explanation)
