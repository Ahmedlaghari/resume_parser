"""
skill_classifier.py — Extract and classify skills from a JD.

Two things happen here:
  1. DETECTION  — scan the JD for known tech skills using a master list.
  2. CLASSIFICATION — decide if each skill is "required" or "nice-to-have"
     by looking at the surrounding sentence/line for signal words.

Method A (rule-based) is implemented here — it handles the vast majority
of real JDs correctly and needs no model download or GPU.

If accuracy on your real JDs turns out to be < 80%, the next upgrade is
to add Method B (HuggingFace zero-shot classification) as a second pass.
"""

import re

# --------------------------------------------------------------------------
# MASTER SKILLS LIST
# Extend this list freely — it's just Python strings.
# Lower-case everything here; matching is done case-insensitively.
# --------------------------------------------------------------------------
SKILLS_DB: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "scala", "kotlin", "swift", "r", "matlab", "bash", "shell", "perl", "ruby",

    # ML / AI
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "reinforcement learning", "mlops", "llm", "generative ai",
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "hugging face", "transformers", "langchain", "openai", "stable diffusion",
    "yolo", "bert", "gpt",

    # Data
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "cassandra", "bigquery", "snowflake", "dbt", "spark", "hadoop", "kafka",
    "airflow", "pandas", "numpy", "matplotlib", "seaborn", "plotly", "tableau",
    "power bi", "looker",

    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github actions", "ci/cd", "mlflow", "kubeflow",
    "sagemaker", "vertex ai", "azure ml", "databricks",

    # Backend / APIs
    "fastapi", "flask", "django", "rest api", "graphql", "grpc", "microservices",
    "rabbitmq", "celery", "nginx", "linux", "git", "github", "gitlab",

    # Frontend (to avoid false positives on Backend JDs)
    "react", "angular", "vue", "next.js", "html", "css",

    # Soft / Process skills (often listed as requirements)
    "agile", "scrum", "jira", "confluence",
]

# --------------------------------------------------------------------------
# CONTEXT WORDS that indicate a skill is REQUIRED
# --------------------------------------------------------------------------
REQUIRED_SIGNALS = [
    "must", "required", "requirement", "essential", "mandatory",
    "minimum", "necessary", "expected", "need to have", "needs to have",
    "you have", "you must", "you will have", "you should have",
    "we require", "we expect",
]

# --------------------------------------------------------------------------
# CONTEXT WORDS that indicate a skill is NICE-TO-HAVE
# --------------------------------------------------------------------------
PREFERRED_SIGNALS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "plus",
    "advantage", "desirable", "ideally", "familiarity with",
    "exposure to", "knowledge of", "experience with", "not required",
    "beneficial", "good to have",
]


def _get_context_window(text: str, skill: str, window: int = 120) -> str:
    """
    Return up to `window` characters surrounding the first occurrence
    of `skill` in `text`.  We use this snippet to look for signal words.
    """
    idx = text.lower().find(skill.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(skill) + window)
    return text[start:end].lower()


def _classify_skill(context: str) -> str:
    """
    Given a context snippet, return 'required' or 'nice_to_have'.
    Default to 'required' — most skills in JDs are required unless
    explicitly marked otherwise.
    """
    for signal in PREFERRED_SIGNALS:
        if signal in context:
            return "nice_to_have"
    # No preferred signal found → treat as required
    return "required"


def extract_and_classify_skills(
    text: str,
) -> tuple[list[str], list[str]]:
    """
    Main entry point.

    Scans the full JD text for known skills, then classifies each one.

    Returns:
        (required_skills, nice_to_have_skills) — two lists of strings.

    Usage:
        from analyzer.skill_classifier import extract_and_classify_skills
        req, nice = extract_and_classify_skills(clean_text)
    """
    required: list[str] = []
    nice_to_have: list[str] = []
    seen: set[str] = set()   # avoid duplicates

    lower_text = text.lower()

    for skill in SKILLS_DB:
        # Use word-boundary matching for short skills to avoid false positives
        # e.g. "r" should not match inside "requirements"
        if len(skill) <= 2:
            pattern = rf"\b{re.escape(skill)}\b"
            if not re.search(pattern, lower_text):
                continue
        else:
            if skill not in lower_text:
                continue

        # Deduplicate (title-cased for display)
        display = skill.title()
        if display in seen:
            continue
        seen.add(display)

        # Classify based on surrounding context
        context = _get_context_window(text, skill)
        bucket = _classify_skill(context)

        if bucket == "required":
            required.append(display)
        else:
            nice_to_have.append(display)

    return required, nice_to_have
