# ============================================================
# main.py — FastAPI Application Entry Point
# ============================================================
# This file:
#   1. Creates the FastAPI app
#   2. Defines the POST /parse-resume endpoint
#   3. Handles file uploads (PDF or DOCX)
#   4. Calls our parser and returns JSON
#
# To run:
#   uvicorn main:app --reload
#
# Then open: http://localhost:8000/docs  ← auto-generated UI
# ============================================================

import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from parser.file_reader import extract_text
from parser.extractor import extract_resume_data
from parser.models import ResumeData

# Load environment variables from .env
load_dotenv()

# ── Create the FastAPI app ───────────────────────────────────
app = FastAPI(
    title="Resume Parser API",
    description=(
        "Module 1 of the AI-Powered Talent Screening Platform. "
        "Upload a PDF or DOCX resume and receive structured JSON."
    ),
    version="1.0.0",
)

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
MAX_FILE_SIZE_MB = 5


@app.get("/")
def root():
    """Health check — tells you the server is running."""
    return {"status": "running", "module": "Resume Parser", "version": "1.0.0"}


@app.post("/parse-resume", response_model=ResumeData)
async def parse_resume(file: UploadFile = File(...)):
    """
    Upload a PDF or DOCX resume file and get back structured JSON.

    - **file**: The resume file to parse (PDF or DOCX, max 5MB)

    Returns a JSON object with all extracted resume fields.
    """

    # ── Validation 1: Check file extension ──────────────────
    _, ext = os.path.splitext(file.filename or "")
    ext = ext.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a PDF or DOCX file."
        )

    # ── Validation 2: Check file size ───────────────────────
    # Read the file content into memory first to check size
    content = await file.read()

    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_FILE_SIZE_MB} MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # ── Save to a temporary file ─────────────────────────────
    # We need a real file path because pdfplumber and docx
    # require opening files from disk, not from memory bytes.
    tmp_path = None
    try:
        # Create a temp file with the correct extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # ── Step 1: Extract raw text ─────────────────────────
        raw_text = extract_text(tmp_path)

        # ── Step 2: Parse structured fields ──────────────────
        resume_data = extract_resume_data(raw_text)

        return resume_data

    except ValueError as e:
        # Known, expected errors (bad file format, empty PDF, etc.)
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        # Unexpected errors — log and return 500
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while parsing resume: {str(e)}"
        )

    finally:
        # Always clean up the temp file, even if parsing failed
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Optional: run directly with `python main.py` ────────────
if __name__ == "__main__":
    import uvicorn

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", 8000))

    uvicorn.run("main:app", host=host, port=port, reload=True)
