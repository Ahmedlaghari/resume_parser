# ============================================================
# test_integration.py — runs all three modules together using
# real output from Module 1 and Module 2.
#
# Run with:
#   cd module_3_matching_engine
#   python integration/test_integration.py
# ============================================================

import sys
import json
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Patch embedder before any matcher imports ─────────────────
# Remove these 10 lines once you have sentence-transformers installed.
# They replace the real model with a fake one so this test runs
# without downloading the 90MB model.
import matcher.semantic_matcher as _sm
import matcher.skills_matcher as _sk

def _fake_embed(texts):
    vectors = []
    for t in texts:
        seed = abs(hash(t.strip().lower())) % (2**32)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(64)
        vectors.append(v / np.linalg.norm(v))
    return np.array(vectors)

_sm.embed_texts = _fake_embed
_sk.embed_texts = _fake_embed
# ─────────────────────────────────────────────────────────────

from integration.adapters import adapt_resume, adapt_jd
from matcher.models import CandidateResume, JobDescription, CategoryWeights
from matcher.skills_matcher import match_skills
from matcher.semantic_matcher import semantic_similarity_score
from matcher.experience_matcher import experience_score
from matcher.education_matcher import education_score
from matcher.scorer import combine_scores, normalize_skill_weights, verdict_for_score
from matcher.explainer import generate_explanation


# ============================================================
# REAL MODULE 1 OUTPUT — Ahmed's resume
# ============================================================

MODULE1_OUTPUT = {
    "name": "Ahmed Ali Laghari",
    "email": "alaghari082@gmail.com",
    "phone": "+92 343 216 3386",
    "location": None,
    "linkedin": None,
    "github": "github.com/Ahmedlaghari",
    "summary": "First-year BSCS student at NUST (H-12, Batch 2029) with a strong self-directed track record in machine learning and data analysis.",
    "skills": [
        "Python", "Bash", "scikit-learn", "NumPy", "Pandas",
        "Custom Neural Networks", "Feature Engineering", "Backpropagation",
        "TF-IDF", "K-Means", "Random Forest", "Git", "GitHub",
        "Jupyter Notebook", "Google Colab"
    ],
    "experience": [],
    "education": [
        {"degree": "BS Computer Science", "institution": "NUST", "year": "2025-2029"},
        {"degree": "A-Levels", "institution": "Credo College, Karachi", "year": "2023-2025"},
        {"degree": "O-Levels", "institution": "Karachi Public School", "year": "2021-2023"},
    ],
    "projects": [
        {"name": "Seismic Event Detection", "description": "Applied the PickNet model to estimate P-wave arrival times from planetary seismometer data for the NASA Space Apps Challenge.", "tech_stack": ["Python", "Jupyter Notebook", "PickNet"]},
        {"name": "Neural Network for Sin-Curve Prediction", "description": "Designed a feedforward neural network entirely from scratch to approximate the sine function, implementing forward pass, backpropagation, and gradient descent manually without any ML framework.", "tech_stack": ["Python", "NumPy", "Custom Backpropagation"]},
        {"name": "Fake News / Spam Mail Detector", "description": "Developed a text classification pipeline using TF-IDF vectorisation with Logistic Regression and Naive Bayes.", "tech_stack": ["Python", "scikit-learn", "TF-IDF"]},
        {"name": "Cardio Disease Prediction", "description": "Classified cardiovascular disease risk from clinical patient data with feature preprocessing and precision/recall evaluation.", "tech_stack": ["Python", "scikit-learn"]},
        {"name": "Gold Price Prediction", "description": "Predicted gold prices from financial indicators using a RandomForest Regressor with correlation-based feature selection.", "tech_stack": ["Python", "scikit-learn", "Random Forest Regressor"]},
        {"name": "Customer Segmentation using K-Means", "description": "Segmented mall customers by spending behaviour and income using K-Means clustering.", "tech_stack": ["Python", "scikit-learn", "K-Means"]},
    ],
    "certifications": []
}


# ============================================================
# REAL MODULE 2 OUTPUT — Senior ML Engineer JD
# ============================================================

MODULE2_OUTPUT = {
    "job_title": "Senior Machine Learning Engineer - AI Platform",
    "company": "NeuralWorks",
    "location": "Remote / San Francisco, CA",
    "employment_type": "Full-time",
    "experience_required": "5+ years",
    "salary_range": "$140,000 - $180,000/year + equity",
    "responsibilities": [
        "Design, train, and deploy large-scale ML models for real-time recommendation systems.",
        "Build and maintain end-to-end MLOps pipelines from experimentation to production.",
        "Collaborate with data engineers to design feature stores and data pipelines.",
        "Mentor junior ML engineers and conduct code reviews.",
        "Define model evaluation frameworks and champion ML best practices across the team.",
        "Work closely with product and engineering teams to translate business goals into ML solutions."
    ],
    "qualifications": [
        "5+ years of hands-on experience in machine learning or data science.",
        "Strong proficiency in Python; experience with PyTorch or TensorFlow is mandatory.",
        "Solid understanding of ML fundamentals: supervised/unsupervised learning, model evaluation, regularization.",
        "Experience deploying models to production using Docker and Kubernetes.",
        "Proficiency with MLflow or a similar experiment tracking tool is required.",
        "Familiarity with distributed data processing using Spark or Kafka.",
        "Experience with AWS (SageMaker, S3, EC2) or GCP (Vertex AI, BigQuery).",
        "Strong communication skills and ability to work in an Agile environment."
    ],
    "benefits": ["Competitive salary + equity package."],
    "industry": "Technology",
    "seniority_level": "Senior",
    "required_skills": [
        "Python", "PyTorch", "TensorFlow", "Docker", "Kubernetes",
        "MLflow", "Apache Spark", "Kafka", "AWS", "SageMaker",
        "S3", "EC2", "GCP", "Vertex AI", "BigQuery", "Agile"
    ],
    "nice_to_have_skills": ["LLMs", "LangChain", "Hugging Face Transformers", "Rust", "Go"],
    "keywords": ["Python", "PyTorch", "TensorFlow", "Docker", "Kubernetes", "MLflow"]
}


# ============================================================
# STEP 1 — ADAPT (transform real outputs into Module 3 shapes)
# ============================================================

def step1_adapt():
    print("\n" + "="*60)
    print("STEP 1 — ADAPTING MODULE 1 + 2 OUTPUTS")
    print("="*60)

    adapted_resume = adapt_resume(MODULE1_OUTPUT)
    adapted_jd     = adapt_jd(MODULE2_OUTPUT)

    print("\nAdapted resume fields:")
    print(f"  candidate_name:          {adapted_resume['candidate_name']}")
    print(f"  skills (count):          {len(adapted_resume['skills'])}")
    print(f"  total_years_experience:  {adapted_resume['total_years_experience']}")
    print(f"  highest_degree:          {adapted_resume['highest_degree']}")
    print(f"  summary_text (first 80): {adapted_resume['summary_text'][:80]}...")

    print("\nAdapted JD fields:")
    print(f"  job_title:               {adapted_jd['job_title']}")
    print(f"  required_skills (count): {len(adapted_jd['required_skills'])}")
    print(f"  min_years_experience:    {adapted_jd['min_years_experience']}")
    print(f"  max_years_experience:    {adapted_jd['max_years_experience']}")
    print(f"  required_education:      {adapted_jd['required_education']}")

    return adapted_resume, adapted_jd


# ============================================================
# STEP 2 — VALIDATE (confirm Pydantic accepts the adapted data)
# ============================================================

def step2_validate(adapted_resume, adapted_jd):
    print("\n" + "="*60)
    print("STEP 2 — PYDANTIC VALIDATION")
    print("="*60)

    candidate = CandidateResume(**adapted_resume)
    jd        = JobDescription(**adapted_jd)

    print(f"\n✓ CandidateResume validated: {candidate.candidate_name}")
    print(f"✓ JobDescription validated:  {jd.job_title}")

    return candidate, jd


# ============================================================
# STEP 3 — SCORE (run all four sub-matchers)
# ============================================================

def step3_score(candidate, jd):
    print("\n" + "="*60)
    print("STEP 3 — SCORING")
    print("="*60)

    # Custom weights — heavily favour ML skills over cloud infrastructure
    # since Ahmed is an ML student, not a cloud engineer
    skill_weights = {
        "Python":       1.0,   # must-have
        "PyTorch":      0.9,   # very important
        "TensorFlow":   0.8,   # important (Ahmed has custom neural nets)
        "Docker":       0.6,
        "Kubernetes":   0.5,
        "MLflow":       0.5,
        "Apache Spark": 0.4,
        "Kafka":        0.3,
        "AWS":          0.4,
        "SageMaker":    0.4,
        "S3":           0.2,
        "EC2":          0.2,
        "GCP":          0.3,
        "Vertex AI":    0.3,
        "BigQuery":     0.2,
        "Agile":        0.3,
    }

    category_weights = CategoryWeights(
        skills_weight=0.45,
        experience_weight=0.25,
        education_weight=0.15,
        semantic_weight=0.15,
    )

    # Normalize skill weights to [0,1] and fill defaults
    resolved_skill_weights = normalize_skill_weights(jd.required_skills, skill_weights)

    # Run each sub-matcher
    skills_sc, skill_detail = match_skills(
        required_skills=jd.required_skills,
        candidate_skills=candidate.skills,
        skill_weights=resolved_skill_weights,
    )

    exp_sc = experience_score(
        candidate_years=candidate.total_years_experience,
        min_years=jd.min_years_experience,
        max_years=jd.max_years_experience,
    )

    edu_sc = education_score(
        candidate_degree=candidate.highest_degree,
        required_degree=jd.required_education,
    )

    jd_text = f"{jd.responsibilities_text}\n{jd.qualifications_text}"
    sem_sc = semantic_similarity_score(candidate.summary_text, jd_text)

    final_sc, breakdown, weights_used = combine_scores(
        skills_score=skills_sc,
        experience_score=exp_sc,
        education_score=edu_sc,
        semantic_score=sem_sc,
        category_weights=category_weights,
    )

    print(f"\n  Skills score:     {skills_sc:6.2f}/100")
    print(f"  Experience score: {exp_sc:6.2f}/100  (candidate has {candidate.total_years_experience} yrs, JD needs {jd.min_years_experience}+)")
    print(f"  Education score:  {edu_sc:6.2f}/100  (candidate: {candidate.highest_degree}, required: {jd.required_education or 'not specified'})")
    print(f"  Semantic score:   {sem_sc:6.2f}/100")
    print(f"  ─────────────────────────")
    print(f"  FINAL SCORE:      {final_sc:6.2f}/100  → {verdict_for_score(final_sc)}")

    print("\n  Skill-by-skill breakdown:")
    print(f"  {'Skill':<20} {'Weight':>6}  {'Matched':>7}  {'Score':>6}  {'Candidate Has'}")
    print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*20}")
    for d in skill_detail:
        matched_str = "✓" if d.matched else "✗"
        has = d.candidate_has or "—"
        print(f"  {d.skill:<20} {d.weight:>6.2f}  {matched_str:>7}  {d.match_score:>6.3f}  {has}")

    return final_sc, breakdown, skill_detail, weights_used


# ============================================================
# STEP 4 — EXPLAIN
# ============================================================

def step4_explain(candidate, jd, final_sc, breakdown, skill_detail):
    print("\n" + "="*60)
    print("STEP 4 — EXPLANATION (Groq/fallback)")
    print("="*60)

    one_liner, explanation = generate_explanation(
        candidate_name=candidate.candidate_name,
        job_title=jd.job_title,
        final_score=final_sc,
        breakdown=breakdown,
        skill_match_detail=skill_detail,
    )

    print(f"\n  One-liner: {one_liner}")
    print(f"\n  Full explanation:\n  {explanation}")


# ============================================================
# STEP 5 — FULL API CALL (tests main.py end-to-end)
# ============================================================

def step5_api_call(adapted_resume, adapted_jd):
    print("\n" + "="*60)
    print("STEP 5 — FULL API CALL VIA /match-candidates")
    print("="*60)

    try:
        from fastapi.testclient import TestClient
        from main import app

        client = TestClient(app)

        request_body = {
            "job_description": adapted_jd,
            "candidates": [adapted_resume],
            "category_weights": {
                "skills_weight": 0.45,
                "experience_weight": 0.25,
                "education_weight": 0.15,
                "semantic_weight": 0.15,
            },
            "skill_weights": {
                "Python": 1.0, "PyTorch": 0.9, "TensorFlow": 0.8,
                "Docker": 0.6, "Kubernetes": 0.5, "MLflow": 0.5,
                "Apache Spark": 0.4, "Kafka": 0.3, "AWS": 0.4,
                "SageMaker": 0.4, "Agile": 0.3,
            },
            "generate_explanations": True,
        }

        resp = client.post("/match-candidates", json=request_body)
        data = resp.json()

        print(f"\n  HTTP status: {resp.status_code}")
        print(f"\n  Summary list:")
        for c in data["summary_list"]:
            print(f"    Rank {c['rank']}: {c['candidate_name']}")
            print(f"    Score:   {c['final_score']}")
            print(f"    Verdict: {c['verdict']}")
            print(f"    Reason:  {c['one_line_reason']}")

        print(f"\n  Full explanation from detailed_results:")
        print(f"    {data['detailed_results'][0]['explanation']}")

    except Exception as e:
        print(f"\n  API call failed: {e}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n🔗 MODULE 1 + 2 + 3 INTEGRATION TEST")
    print("   Candidate: Ahmed Ali Laghari")
    print("   JD: Senior ML Engineer — NeuralWorks")

    adapted_resume, adapted_jd = step1_adapt()
    candidate, jd              = step2_validate(adapted_resume, adapted_jd)
    final_sc, breakdown, skill_detail, weights_used = step3_score(candidate, jd)
    step4_explain(candidate, jd, final_sc, breakdown, skill_detail)
    step5_api_call(adapted_resume, adapted_jd)

    print("\n" + "="*60)
    print("✓ Integration test complete")
    print("="*60)
