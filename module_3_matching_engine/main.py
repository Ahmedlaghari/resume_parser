"""
main.py
--------
FastAPI app exposing POST /match-candidates.

Run locally with:
    uvicorn main:app --reload

Then test with the sample data:
    curl -X POST http://127.0.0.1:8000/match-candidates \\
      -H "Content-Type: application/json" \\
      -d @sample_data/full_request_example.json

Or just open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from fastapi import FastAPI
from dotenv import load_dotenv

from matcher.models import (
    MatchRequest,
    MatchResponse,
    SummaryEntry,
    DetailedResult,
    WeightsUsed,
)
from matcher.skills_matcher import match_skills
from matcher.semantic_matcher import semantic_similarity_score
from matcher.experience_matcher import experience_score
from matcher.education_matcher import education_score
from matcher.scorer import combine_scores, normalize_skill_weights, verdict_for_score
from matcher.explainer import generate_explanation

load_dotenv()

app = FastAPI(
    title="Module 3 — Candidate-Job Matching Engine",
    description="Weighted, semantically-aware candidate ranking against a job description.",
)


def _score_one_candidate(candidate, jd, skill_weights, category_weights):
    """Runs one candidate through all four sub-matchers and combines them."""
    skills_score, skill_match_detail = match_skills(
        required_skills=jd.required_skills,
        candidate_skills=candidate.skills,
        skill_weights=skill_weights,
    )

    exp_score = experience_score(
        candidate_years=candidate.total_years_experience,
        min_years=jd.min_years_experience,
        max_years=jd.max_years_experience,
    )

    edu_score = education_score(
        candidate_degree=candidate.highest_degree,
        required_degree=jd.required_education,
    )

    jd_text = f"{jd.responsibilities_text}\n{jd.qualifications_text}".strip()
    sem_score = semantic_similarity_score(candidate.summary_text, jd_text)

    final_score, breakdown, weights_used = combine_scores(
        skills_score=skills_score,
        experience_score=exp_score,
        education_score=edu_score,
        semantic_score=sem_score,
        category_weights=category_weights,
    )

    missing_critical = [d.skill for d in skill_match_detail if not d.matched and d.weight >= 0.7]

    return final_score, breakdown, skill_match_detail, missing_critical, weights_used


@app.post("/match-candidates", response_model=MatchResponse)
def match_candidates(request: MatchRequest) -> MatchResponse:
    jd = request.job_description

    # Layer 2 weights: fill defaults (1.0) for any unweighted required skill, clip to [0,1]
    skill_weights = normalize_skill_weights(jd.required_skills, request.skill_weights)

    scored_candidates = []
    weights_used_normalized = None

    for candidate in request.candidates:
        final_score, breakdown, skill_match_detail, missing_critical, weights_used_normalized = (
            _score_one_candidate(candidate, jd, skill_weights, request.category_weights)
        )

        if request.generate_explanations:
            one_liner, explanation = generate_explanation(
                candidate_name=candidate.candidate_name,
                job_title=jd.job_title,
                final_score=final_score,
                breakdown=breakdown,
                skill_match_detail=skill_match_detail,
            )
        else:
            one_liner = f"Scored {final_score}/100 — explanations disabled for this request."
            explanation = one_liner

        scored_candidates.append(
            {
                "candidate_name": candidate.candidate_name,
                "final_score": final_score,
                "breakdown": breakdown,
                "skill_match_detail": skill_match_detail,
                "missing_critical": missing_critical,
                "one_liner": one_liner,
                "explanation": explanation,
            }
        )

    # Sort best-to-worst, then assign ranks shared by both summary_list and detailed_results
    scored_candidates.sort(key=lambda c: c["final_score"], reverse=True)

    summary_list = []
    detailed_results = []
    for i, c in enumerate(scored_candidates, start=1):
        verdict = verdict_for_score(c["final_score"])

        summary_list.append(
            SummaryEntry(
                rank=i,
                candidate_name=c["candidate_name"],
                final_score=c["final_score"],
                verdict=verdict,
                one_line_reason=c["one_liner"],
            )
        )
        detailed_results.append(
            DetailedResult(
                rank=i,
                candidate_name=c["candidate_name"],
                final_score=c["final_score"],
                breakdown=c["breakdown"],
                skill_match_detail=c["skill_match_detail"],
                missing_critical_skills=c["missing_critical"],
                explanation=c["explanation"],
            )
        )

    return MatchResponse(
        job_title=jd.job_title,
        total_candidates_evaluated=len(request.candidates),
        summary_list=summary_list,
        detailed_results=detailed_results,
        weights_used=WeightsUsed(
            category_weights=weights_used_normalized,
            skill_weights=skill_weights,
        ),
    )


@app.get("/")
def health_check():
    return {"status": "ok", "service": "candidate-matching-engine"}
