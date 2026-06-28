"""
models.py
---------
Pydantic schemas for Module 3's API. These define exactly what the
/match-candidates endpoint expects as input and what it returns.

Why this file exists separately: FastAPI uses these classes to
auto-validate incoming JSON (reject bad requests before your logic
even runs) and to auto-generate the API docs at /docs.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ----------------------------------------------------------------
# INPUT SCHEMAS (what Module 1 and Module 2 hand us)
# ----------------------------------------------------------------

class JobDescription(BaseModel):
    """
    Expected shape of Module 2's (JD Analyzer) output.
    Adjust field names here if your real Module 2 output differs --
    this is the ONE place you'd need to change to stay in sync.
    """
    job_title: str
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_years_experience: float = 0
    max_years_experience: Optional[float] = None
    required_education: Optional[str] = None  # e.g. "Bachelors", "Masters", or None
    responsibilities_text: str = ""           # used for semantic similarity
    qualifications_text: str = ""             # used for semantic similarity


class CandidateResume(BaseModel):
    """
    Expected shape of Module 1's (Resume Parser) output, one per candidate.
    """
    candidate_name: str
    skills: list[str] = Field(default_factory=list)
    total_years_experience: float = 0
    highest_degree: Optional[str] = None      # e.g. "Bachelors", "Masters", "PhD", None
    summary_text: str = ""                    # used for semantic similarity


# ----------------------------------------------------------------
# WEIGHT SCHEMAS (the configurable part -- Layer 1 and Layer 2)
# ----------------------------------------------------------------

class CategoryWeights(BaseModel):
    """Layer 1: how much each broad category contributes to final_score."""
    skills_weight: float = 0.25
    experience_weight: float = 0.25
    education_weight: float = 0.25
    semantic_weight: float = 0.25

    def normalized(self) -> "CategoryWeights":
        """Returns a copy rescaled so all four weights sum to 1.0."""
        total = (
            self.skills_weight
            + self.experience_weight
            + self.education_weight
            + self.semantic_weight
        )
        if total <= 0:
            # Degenerate case: fall back to equal weighting rather than divide by zero
            return CategoryWeights()
        return CategoryWeights(
            skills_weight=self.skills_weight / total,
            experience_weight=self.experience_weight / total,
            education_weight=self.education_weight / total,
            semantic_weight=self.semantic_weight / total,
        )


# ----------------------------------------------------------------
# REQUEST SCHEMA
# ----------------------------------------------------------------

class MatchRequest(BaseModel):
    job_description: JobDescription
    candidates: list[CandidateResume]
    category_weights: Optional[CategoryWeights] = None
    # skill_weights is optional; any skill not mentioned defaults to 1.0
    skill_weights: Optional[dict[str, float]] = None
    # Set False to skip the LLM explanation call (faster, no API cost)
    generate_explanations: bool = True

    @field_validator("candidates")
    @classmethod
    def must_have_at_least_one_candidate(cls, v):
        if len(v) == 0:
            raise ValueError("candidates list cannot be empty")
        return v


# ----------------------------------------------------------------
# RESPONSE SCHEMAS
# ----------------------------------------------------------------

class SkillMatchDetail(BaseModel):
    skill: str
    weight: float
    matched: bool
    match_score: float          # 1.0 for exact match, 0-1 similarity for partial, 0.0 for none
    candidate_has: Optional[str]  # the candidate's skill string that matched it, if any


class ScoreBreakdown(BaseModel):
    skills_score: float
    experience_score: float
    education_score: float
    semantic_similarity_score: float


class SummaryEntry(BaseModel):
    rank: int
    candidate_name: str
    final_score: float
    verdict: str
    one_line_reason: str


class DetailedResult(BaseModel):
    rank: int
    candidate_name: str
    final_score: float
    breakdown: ScoreBreakdown
    skill_match_detail: list[SkillMatchDetail]
    missing_critical_skills: list[str]
    explanation: str


class WeightsUsed(BaseModel):
    category_weights: CategoryWeights
    skill_weights: dict[str, float]


class MatchResponse(BaseModel):
    job_title: str
    total_candidates_evaluated: int
    summary_list: list[SummaryEntry]
    detailed_results: list[DetailedResult]
    weights_used: WeightsUsed
