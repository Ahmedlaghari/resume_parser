"""
main.py — FastAPI application for Module 2: JD Analyzer.

Two endpoints:
  POST /analyze-jd        → accepts raw JSON { "text": "..." }
  POST /analyze-jd-file   → accepts .txt or .pdf file upload

Both return the same JobDescription JSON schema.

Run with:
    uvicorn main:app --reload --port 8002
"""

import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# KeyBERT for keyword extraction — import lazily to keep startup fast
try:
    from keybert import KeyBERT
    kw_model = KeyBERT()          # uses MiniLM by default (~80 MB, no GPU needed)
    KEYBERT_AVAILABLE = True
except ImportError:
    kw_model = None
    KEYBERT_AVAILABLE = False

# Our analyzer modules
from analyzer import (
    clean_text,
    extract_job_title, extract_company, extract_location,
    extract_employment_type, extract_experience, extract_salary,
    extract_responsibilities, extract_qualifications,
    extract_benefits, extract_industry,
    extract_and_classify_skills,
    detect_seniority,
    JobDescription,
)

# --------------------------------------------------------------------------
load_dotenv()   # reads .env file — add ANTHROPIC_API_KEY here if needed later
# --------------------------------------------------------------------------

app = FastAPI(
    title="JD Analyzer — Module 2",
    description="Extracts structured data from raw job description text.",
    version="1.0.0",
)

# Allow all origins in dev (tighten this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================================
# Core analysis function — shared by both endpoints
# ==========================================================================

def analyze_jd_text(raw_text: str) -> JobDescription:
    """
    Run the full analysis pipeline on raw JD text.
    Returns a JobDescription pydantic object (serialises to JSON automatically).
    """
    # ── Step 1: clean the text ────────────────────────────────────────────
    text = clean_text(raw_text)

    # ── Step 2: extract individual fields ────────────────────────────────
    experience = extract_experience(text)

    # ── Step 3: skill extraction & classification ─────────────────────────
    required_skills, nice_to_have_skills = extract_and_classify_skills(text)

    # ── Step 4: keyword extraction (KeyBERT) ─────────────────────────────
    if KEYBERT_AVAILABLE and kw_model and text.strip():
        raw_keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),   # single words AND two-word phrases
            stop_words="english",
            top_n=10,
        )
        keywords = [kw for kw, _score in raw_keywords]
    else:
        # Fallback: just use the required skills as keywords
        keywords = required_skills[:10]

    # ── Step 5: assemble the output ───────────────────────────────────────
    return JobDescription(
        job_title=extract_job_title(text),
        company=extract_company(text),
        location=extract_location(text),
        employment_type=extract_employment_type(text),
        seniority_level=detect_seniority(text, experience),
        experience_required=experience,
        salary_range=extract_salary(text),
        required_skills=required_skills,
        nice_to_have_skills=nice_to_have_skills,
        responsibilities=extract_responsibilities(text),
        qualifications=extract_qualifications(text),
        keywords=keywords,
        industry=extract_industry(text),
        benefits=extract_benefits(text),
    )


# ==========================================================================
# ENDPOINT 1 — Raw text input
# ==========================================================================

class JDTextInput(BaseModel):
    text: str   # the raw job description string


@app.post("/analyze-jd", response_model=JobDescription)
async def analyze_jd(payload: JDTextInput):
    """
    Accept raw JD text and return structured JSON.

    Example request body:
        { "text": "We are looking for a Senior ML Engineer at Acme Corp..." }
    """
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text field is empty.")

    return analyze_jd_text(payload.text)


# ==========================================================================
# ENDPOINT 2 — File upload (.txt or .pdf)
# ==========================================================================

@app.post("/analyze-jd-file", response_model=JobDescription)
async def analyze_jd_file(file: UploadFile = File(...)):
    """
    Accept a .txt or .pdf file and return structured JSON.

    Test with curl:
        curl -X POST http://localhost:8002/analyze-jd-file \
             -F "file=@sample_jds/sample_ml_engineer.txt"
    """
    filename = file.filename or ""

    if filename.endswith(".txt"):
        raw_bytes = await file.read()
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    elif filename.endswith(".pdf"):
        # PDF support requires pypdf: pip install pypdf
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(
                status_code=422,
                detail="PDF support requires pypdf.  Run: pip install pypdf"
            )
        raw_bytes = await file.read()
        reader = PdfReader(io.BytesIO(raw_bytes))
        raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    else:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a .txt or .pdf file."
        )

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="File appears to be empty.")

    return analyze_jd_text(raw_text)


# ==========================================================================
# Health check
# ==========================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "module": "2 — JD Analyzer",
        "keybert_available": KEYBERT_AVAILABLE,
    }
