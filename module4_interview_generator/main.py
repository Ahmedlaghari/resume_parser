# ============================================================
# main.py — Module 4 entry point.
#
# Orchestrates the full RAG pipeline:
#   1. Auto-build the ChromaDB index on first run (indexer)
#   2. Retrieve relevant questions for this candidate (retriever)
#   3. Personalize them with the LLM (generator)
#   4. Return the final structured output
#
# Usage:
#   from module4_interview_generator.main import generate_interview_questions
#   result = generate_interview_questions(candidate, jd, match_result)
# ============================================================

import json
import logging

from .indexer import build_index, get_collection
from .retriever import retrieve
from .generator import generate

logger = logging.getLogger(__name__)


def _ensure_index() -> None:
    """Build the ChromaDB index if it doesn't exist yet."""
    try:
        get_collection()
    except RuntimeError:
        logger.info("Index not found — building now...")
        build_index()


def generate_interview_questions(
    candidate: dict,
    jd: dict,
    match_result: dict,
    n_per_category: int = 6,
) -> dict:
    """
    Generate a personalized interview question set for one candidate.

    Args:
        candidate:      Module 1 output (parsed resume dict)
        jd:             Module 2 output (parsed JD dict)
        match_result:   Module 3 output for this specific candidate
        n_per_category: Questions retrieved per category before LLM selection

    Returns:
        Dict matching the Module 4 output schema:
        {
            "candidate_name": str,
            "job_title": str,
            "questions": [
                {
                    "category": str,
                    "question": str,
                    "what_to_listen_for": str
                },
                ...
            ]
        }
    """
    _ensure_index()

    pool = retrieve(candidate, jd, match_result, n_per_category=n_per_category)
    logger.info("Retrieved %d questions from vector DB", len(pool))

    result = generate(candidate, jd, match_result, pool)

    return result.model_dump()


# ============================================================
# QUICK TEST — run with:
#   python -m module4_interview_generator.main
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    sample_candidate = {
        "name": "Ali Khan",
        "email": "ali.khan@example.com",
        "total_experience_years": 4,
        "skills": [
            "Python", "TensorFlow", "NLP", "FastAPI",
            "PostgreSQL", "Git", "Pandas", "Scikit-learn"
        ],
        "experience": [
            {
                "title": "ML Engineer",
                "company": "ABC Corp",
                "duration": "Jan 2022 – Present",
                "description": "Built NLP pipelines for text classification. Deployed models via FastAPI."
            },
            {
                "title": "Data Scientist",
                "company": "DataStart",
                "duration": "2020 – 2021",
                "description": "Developed sentiment analysis models on social media data."
            },
        ],
        "projects": [
            {
                "name": "Sentiment Pipeline",
                "description": "End-to-end NLP pipeline for Twitter sentiment",
                "tech_stack": ["Python", "TensorFlow", "FastAPI"]
            },
            {
                "name": "Resume Classifier",
                "description": "Multi-label resume classification system",
                "tech_stack": ["Python", "Scikit-learn", "PostgreSQL"]
            },
        ],
        "education": [
            {
                "degree": "B.S. Computer Science",
                "institution": "FAST-NUCES",
                "year": "2020"
            }
        ],
        "certifications": [],
    }

    sample_jd = {
        "job_title": "Machine Learning Engineer",
        "company": "TechVision",
        "required_skills": ["Python", "PyTorch", "Docker", "NLP", "Kubernetes", "MLflow"],
        "nice_to_have_skills": ["AWS", "FastAPI"],
        "experience_required": "3-5 years",
        "responsibilities": [
            "Build and deploy production ML pipelines",
            "Fine-tune transformer models for NLP tasks",
            "Containerize models with Docker and orchestrate with Kubernetes",
        ],
        "qualifications": [
            "Strong Python skills",
            "Experience with deep learning frameworks",
            "MLOps and deployment experience",
        ],
    }

    sample_match = {
        "candidate_name": "Ali Khan",
        "final_score": 71.5,
        "matched_skills": ["Python", "NLP", "FastAPI", "TensorFlow"],
        "missing_skills": ["PyTorch", "Docker", "Kubernetes", "MLflow"],
        "explanation": "Strong NLP background but missing key deployment and MLOps skills.",
    }

    print("Generating interview questions...\n")
    output = generate_interview_questions(sample_candidate, sample_jd, sample_match)

    print(json.dumps(output, indent=2))
