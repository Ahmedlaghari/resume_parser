# Module 1 — Resume Parser

Part of the **AI-Powered Talent Screening Platform** portfolio project.

Takes a PDF or DOCX resume, extracts raw text, and returns clean structured JSON using NLP and regex.

---

## Project Structure

```
module_1_resume_parser/
├── main.py               # FastAPI app & /parse-resume endpoint
├── parser/
│   ├── __init__.py       # Makes this a Python package
│   ├── file_reader.py    # PDF and DOCX text extraction
│   ├── extractor.py      # NLP field extraction (regex + spaCy)
│   └── models.py         # Pydantic schemas for output
├── sample_resumes/       # Put test resumes here
├── test_parser.py        # Quick test without starting server
├── requirements.txt
└── .env
```

---

## Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the spaCy English language model
python -m spacy download en_core_web_sm
```

---

## Running the API

```bash
uvicorn main:app --reload
```

Server starts at: `http://localhost:8000`

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## Testing

### Option A — Quick test (no server needed)
```bash
python test_parser.py
```
Runs the parser on `sample_resumes/sample_resume.txt` and prints JSON output.

### Option B — Test the API with curl
```bash
curl -X POST http://localhost:8000/parse-resume \
  -F "file=@sample_resumes/sample_resume.txt"
```

### Option C — Postman / Thunder Client
1. Method: `POST`
2. URL: `http://localhost:8000/parse-resume`
3. Body: `form-data`
4. Key: `file` (type: File)
5. Value: select your PDF or DOCX

---

## Output Format

```json
{
  "name": "Ali Hassan",
  "email": "ali.hassan@gmail.com",
  "phone": "+92-300-1234567",
  "location": "Karachi",
  "linkedin": "linkedin.com/in/alihassan",
  "github": "github.com/alihassan",
  "summary": "Software engineer with 3 years of experience...",
  "skills": ["Python", "Machine Learning", "TensorFlow", "SQL"],
  "experience": [
    {
      "title": "Software Engineer",
      "company": "DataVision Pvt. Ltd., Karachi",
      "duration": "Jan 2022 - Present",
      "description": "Built and maintained ML inference pipelines..."
    }
  ],
  "education": [
    {
      "degree": "BS Computer Science",
      "institution": "University of Karachi",
      "year": "2020"
    }
  ],
  "projects": [
    {
      "name": "Sentiment Analyzer",
      "description": "Built a tweet sentiment classifier...",
      "tech_stack": ["Python", "BERT", "Flask", "HuggingFace Transformers"]
    }
  ],
  "certifications": ["AWS Cloud Practitioner", "DeepLearning.AI ML Specialization"]
}
```

---

## Acceptance Checklist

- [x] Parser reads both PDF and DOCX files
- [x] Extracts name, email, phone via regex + spaCy NER
- [x] Skills list extracted from Skills section
- [x] Experience entries extracted with title, company, duration, description
- [x] Education entries extracted with degree, institution, year
- [x] Projects extracted with name, description, tech_stack
- [x] Certifications extracted as list
- [x] FastAPI endpoint runs and returns JSON
- [x] All fields are Optional — missing data returns null, not a crash
- [ ] Tested on 3+ different real resume formats (your task!)
- [ ] Code pushed to GitHub

---

## Notes for Module 2

The `ResumeData` JSON output from this module is the input to:
- **Module 2** (Job Description Analyzer) — compare skills/experience
- **Module 3** (Matching Engine) — score candidates against JDs

Keep the JSON schema stable. If you add fields later, make them Optional.
