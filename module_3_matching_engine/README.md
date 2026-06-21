# Module 3 — Candidate-Job Matching Engine

Weighted, semantically-aware scoring engine that ranks candidate resumes
(Module 1 output) against a job description (Module 2 output).

## Setup

```bash
cd module_3_matching_engine
pip install -r requirements.txt
cp .env.example .env   # then put your real GROQ_API_KEY in .env
```

The first time `semantic_similarity_score()` or skill-level fuzzy
matching runs, `sentence-transformers` will download the
`all-MiniLM-L6-v2` model (~90MB) from Hugging Face. That requires
normal internet access — it's a one-time download, then it's cached
locally.

## Run the API

```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or:

```bash
python sample_data/build_full_request.py   # assembles sample_jd.json + all 5 sample resumes
curl -X POST http://127.0.0.1:8000/match-candidates \
  -H "Content-Type: application/json" \
  -d @sample_data/full_request_example.json
```

## Run the tests

```bash
pip install pytest
python -m pytest tests/test_logic.py -v
```

`tests/test_logic.py` covers the deterministic logic directly:
experience scoring, education scoring, the weighted skills formula
(including the exact hand-calculated example from the spec: Python=1.0,
Java=0.5, Docker=0.7, AWS=0.6 → 82.14), category-weight normalization,
verdict buckets, and skill-weight clipping.

It also tests the skill-matcher's *fuzzy-matching logic* (does a
similarity above threshold get treated as a match, does one below it
get rejected) using a small fake embedder instead of the real
sentence-transformers model. That's a deliberate choice for fast,
network-free testing — the fake embedder hand-codes a few synonym
pairs (e.g. "ML" ≈ "Machine Learning") so the *threshold logic* is
verified, but it does not test embedding *quality*. Once you've
installed the real dependencies, swap in the real `get_embedder()` to
confirm actual semantic matches feel right for your domain's
vocabulary — tune `SIMILARITY_THRESHOLD` in `skills_matcher.py`
(default 0.75) if matches feel too loose or too strict.

## What was verified before handing this off

Using the fake embedder above, I ran the full FastAPI pipeline
end-to-end against `sample_data/` (1 JD, 5 candidates: Jane Doe, Ali
Khan, Sara Ahmed, Bilal Raza, Maria Lopez) with the custom weights from
the spec (Python=1.0, Java=0.3, Docker=0.7, AWS=0.6, PyTorch=0.9). The
resulting ranking (Maria > Jane > Ali > Sara > Bilal) and Sara's
individual breakdown matched hand-calculation exactly, confirming the
weighting visibly changes rankings as intended — a candidate missing
Python and PyTorch (both high-weight) ranks below one missing only
lower-weight skills, even with fewer total skills listed.

I also exercised the edge cases from the spec's TEST PLAN: a candidate
with zero matching skills (returns a low-but-valid score, no crash),
category weights that don't sum to 1.0 (auto-normalized — verified
`{10, 10, 0, 0}` → `{0.5, 0.5, 0, 0}`), skill weights outside `[0, 1]`
(clipped), and an empty candidate list (correctly rejected with HTTP
422 by Pydantic validation).

What I could **not** verify in this sandbox: actual embedding-based
semantic quality (e.g. does "PyTorch" really score high against "deep
learning frameworks" with the real model) and the live Groq API call
in `explainer.py`, since this environment has no internet access to
Hugging Face and no API key configured. `explainer.py` has a tested
fallback path (template-based explanation) that fires automatically if
the API call fails for any reason, so the endpoint never breaks
because of this piece — but you should sanity-check the real LLM
output quality once you add your key.

Note on `explainer.py`'s client setup: unlike the Anthropic SDK, the
Groq client raises immediately if `GROQ_API_KEY` is missing — it
doesn't wait until you make a call. Building the client eagerly at
import time (the same pattern `extractor.py` uses) would crash the
whole FastAPI app on startup for anyone without a key set, since
`main.py` imports `generate_explanation` at module level. `explainer.py`
instead builds the client lazily on first use (`_get_client()`,
memoized with `lru_cache`), so a missing key only surfaces inside the
try/except in `generate_explanation` and gets routed to the
deterministic fallback instead. Worth applying the same lazy-init
pattern to `extractor.py` if you want Module 2 to be equally resilient
to a missing key.

## Project structure

```
module_3_matching_engine/
├── main.py                    # FastAPI app, POST /match-candidates
├── matcher/
│   ├── models.py               # Pydantic request/response schemas
│   ├── skills_matcher.py       # Layer 2: weighted skill matching
│   ├── semantic_matcher.py     # embedding model + overall fit score
│   ├── experience_matcher.py   # years-of-experience scoring
│   ├── education_matcher.py    # degree-level scoring
│   ├── scorer.py                # Layer 1: combines sub-scores, verdicts
│   └── explainer.py             # the one LLM call, with fallback
├── sample_data/
│   ├── sample_jd.json
│   ├── sample_resumes/          # 5 sample candidates
│   └── build_full_request.py    # assembles a full test request
├── tests/
│   └── test_logic.py
├── requirements.txt
└── .env.example
```

## Next steps (not part of this module)

- Module 7 (Streamlit dashboard) renders `summary_list` as the main
  table and `detailed_results` as the expandable drill-down — that's
  why the API returns both instead of one merged structure.
- If you start matching against hundreds of candidates, `skills_matcher.py`
  currently re-embeds the JD skill against every candidate skill per
  request; swap to the `faiss-cpu`/`chromadb` route mentioned in the
  spec to pre-embed and index candidate skills once.
