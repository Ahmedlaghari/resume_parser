# ============================================================
# connector.py — runs all three modules together
#
# What it does:
#   1. Sends resume files to Module 1 (port 8000)
#   2. Sends JD file to Module 2 (port 8002)
#   3. Feeds both outputs into Module 3 (port 8001)
#   4. Prints the final ranked results
#
# Before running:
#   Terminal 1: cd module1_parser       && uvicorn main:app --port 8000
#   Terminal 2: cd module2_jd_analyzer  && uvicorn main:app --port 8002
#   Terminal 3: cd module3_matching_engine && uvicorn main:app --port 8001
#   Terminal 4: python connector.py resume1.pdf resume2.pdf --jd jd.txt
#
# Usage:
#   python connector.py RESUME [RESUME ...] --jd JD_FILE [options]
#
# Examples:
#   python connector.py resumes/sara.pdf resumes/ahmed.pdf --jd jds/neuralworks.txt
#   python connector.py resumes/sara.pdf --jd jds/ml_engineer.pdf --no-explain
#   python connector.py resumes/*.pdf --jd jds/neuralworks.txt --weights weights.json
# ============================================================

import sys
import json
import argparse
import pathlib
import requests

# ── Server URLs — change ports here if yours differ ──────────
MODULE1_URL = "http://localhost:8000/parse-resume"
MODULE2_URL = "http://localhost:8002/analyze-jd-file"
MODULE3_URL = "http://localhost:8001/match-candidates"

# ── Weights to use — edit or pass --weights file to override ─
DEFAULT_SKILL_WEIGHTS = {
    "Python":       1.0,
    "PyTorch":      0.9,
    "TensorFlow":   0.8,
    "Docker":       0.7,
    "MLflow":       0.6,
    "AWS":          0.6,
    "SageMaker":    0.5,
    "Kubernetes":   0.5,
    "Apache Spark": 0.4,
    "Kafka":        0.3,
    "Agile":        0.2,
}

DEFAULT_CATEGORY_WEIGHTS = {
    "skills_weight":     0.50,
    "experience_weight": 0.25,
    "education_weight":  0.10,
    "semantic_weight":   0.15,
}


# ============================================================
# STEP 1 — send one resume file to Module 1
# ============================================================

def parse_resume(file_path: pathlib.Path) -> dict:
    print(f"  → Sending {file_path.name} to Module 1...")

    with open(file_path, "rb") as f:
        response = requests.post(
            MODULE1_URL,
            files={"file": (file_path.name, f)},
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Module 1 failed for {file_path.name}: "
            f"{response.status_code} — {response.text}"
        )

    result = response.json()
    print(f"  ✓ Parsed: {result.get('name', file_path.name)}")
    return result


# ============================================================
# STEP 2 — send JD file to Module 2
# ============================================================

def analyze_jd(file_path: pathlib.Path) -> dict:
    print(f"  → Sending {file_path.name} to Module 2...")

    with open(file_path, "rb") as f:
        response = requests.post(
            MODULE2_URL,
            files={"file": (file_path.name, f)},
            timeout=60,
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Module 2 failed for {file_path.name}: "
            f"{response.status_code} — {response.text}"
        )

    result = response.json()
    print(f"  ✓ Analyzed: {result.get('job_title', file_path.name)}")
    return result


# ============================================================
# STEP 3 — adapt Module 1 + 2 outputs → Module 3 input
# ============================================================

def adapt_resume(m1: dict) -> dict:
    """Translates Module 1 field names into Module 3's expected shape."""

    # Calculate total years from experience list
    import re
    total_years = 0.0
    for job in m1.get("experience", []):
        start = str(job.get("start_year") or job.get("start") or "")
        end   = str(job.get("end_year")   or job.get("end")   or "")
        s = re.search(r"(\d{4})", start)
        e = re.search(r"(\d{4})", end)
        if s and e:
            total_years += float(e.group(1)) - float(s.group(1))
        else:
            total_years += 1.5   # rough fallback per job entry

    # Find highest degree
    degree_rank = {"phd": 4, "doctorate": 4, "masters": 3, "master": 3,
                   "ms": 3, "msc": 3, "bs": 2, "bsc": 2, "bachelor": 2,
                   "bachelors": 2, "associate": 1, "a-levels": 1}
    best_degree = None
    best_rank   = 0
    for edu in m1.get("education", []):
        raw = edu.get("degree", "").lower()
        for key, rank in degree_rank.items():
            if key in raw and rank > best_rank:
                best_rank   = rank
                best_degree = {"phd": "PhD", "doctorate": "PhD",
                               "masters": "Masters", "master": "Masters",
                               "ms": "Masters", "msc": "Masters",
                               "bs": "Bachelors", "bsc": "Bachelors",
                               "bachelor": "Bachelors", "bachelors": "Bachelors",
                               "associate": "Associate", "a-levels": "Associate"}[key]

    # Build summary text from summary + project descriptions
    parts = [m1.get("summary", "")]
    for p in m1.get("projects", []):
        if p.get("description"):
            parts.append(p["description"])
    summary_text = " ".join(filter(None, parts))

    return {
        "candidate_name":         m1.get("name", "Unknown"),
        "skills":                  m1.get("skills", []),
        "total_years_experience":  round(total_years, 1),
        "highest_degree":          best_degree,
        "summary_text":            summary_text,
    }


def adapt_jd(m2: dict) -> dict:
    """Translates Module 2 field names into Module 3's expected shape."""
    import re

    # Parse "5+ years" or "3-5 years" → min, max
    exp_str = m2.get("experience_required", "")
    min_exp, max_exp = 0.0, None
    m = re.search(r"(\d+)\s*\+", exp_str)
    if m:
        min_exp = float(m.group(1))
    else:
        m = re.search(r"(\d+)\s*[-to]+\s*(\d+)", exp_str)
        if m:
            min_exp = float(m.group(1))
            max_exp = float(m.group(2))
        else:
            m = re.search(r"(\d+)", exp_str)
            if m:
                min_exp = float(m.group(1))

    responsibilities = m2.get("responsibilities", [])
    qualifications   = m2.get("qualifications", [])

    return {
        "job_title":            m2.get("job_title", ""),
        "required_skills":      m2.get("required_skills", []),
        "preferred_skills":     m2.get("nice_to_have_skills", []),
        "min_years_experience": min_exp,
        "max_years_experience": max_exp,
        "required_education":   None,
        "responsibilities_text": "\n".join(responsibilities),
        "qualifications_text":   "\n".join(qualifications),
    }


# ============================================================
# STEP 4 — send everything to Module 3
# ============================================================

def match_candidates(
    adapted_jd:      dict,
    adapted_resumes: list[dict],
    skill_weights:   dict,
    category_weights: dict,
    generate_explanations: bool,
) -> dict:
    print(f"  → Sending {len(adapted_resumes)} candidate(s) to Module 3...")

    payload = {
        "job_description":       adapted_jd,
        "candidates":            adapted_resumes,
        "skill_weights":         skill_weights,
        "category_weights":      category_weights,
        "generate_explanations": generate_explanations,
    }

    response = requests.post(MODULE3_URL, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"Module 3 failed: {response.status_code} — {response.text}"
        )

    print("  ✓ Ranking complete")
    return response.json()


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(result: dict):
    print("\n" + "=" * 62)
    print(f"  JOB:  {result['job_title']}")
    print(f"  {result['total_candidates_evaluated']} candidate(s) evaluated")
    print("=" * 62)

    print("\n📋  RANKING\n")
    print(f"  {'#':<4} {'Name':<24} {'Score':>6}  Verdict")
    print(f"  {'─'*4} {'─'*24} {'─'*6}  {'─'*20}")
    for c in result["summary_list"]:
        print(f"  #{c['rank']:<3} {c['candidate_name']:<24} {c['final_score']:>6.1f}  {c['verdict']}")
        print(f"       {c['one_line_reason']}")

    print("\n\n📊  BREAKDOWN\n")
    for d in result["detailed_results"]:
        b = d["breakdown"]
        print(f"  ── {d['candidate_name']}  ({d['final_score']}/100) ──")
        print(f"     Skills      {b['skills_score']:>6.1f}/100")
        print(f"     Experience  {b['experience_score']:>6.1f}/100")
        print(f"     Education   {b['education_score']:>6.1f}/100")
        print(f"     Semantic    {b['semantic_similarity_score']:>6.1f}/100")

        matched = [s["skill"] for s in d["skill_match_detail"] if s["matched"]]
        missing = [s["skill"] for s in d["skill_match_detail"] if not s["matched"]]
        if matched:
            print(f"     ✓  {', '.join(matched)}")
        if missing:
            print(f"     ✗  {', '.join(missing)}")
        if d.get("explanation"):
            print(f"     💬 {d['explanation']}")
        print()

    print("=" * 62)


# ============================================================
# HEALTH CHECK — make sure all three servers are up
# ============================================================

def check_servers():
    checks = [
        ("Module 1", "http://localhost:8000/"),
        ("Module 2", "http://localhost:8002/health"),
        ("Module 3", "http://localhost:8001/"),
    ]
    all_ok = True
    for name, url in checks:
        try:
            r = requests.get(url, timeout=5)
            print(f"  ✓ {name} is running ({url})")
        except Exception:
            print(f"  ✗ {name} is NOT reachable at {url}")
            all_ok = False
    return all_ok


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run resume(s) and a JD through all three modules."
    )
    parser.add_argument(
        "resumes", nargs="+", type=pathlib.Path,
        help="One or more resume files (PDF or DOCX)"
    )
    parser.add_argument(
        "--jd", required=True, type=pathlib.Path,
        help="Job description file (.txt or .pdf)"
    )
    parser.add_argument(
        "--weights", type=pathlib.Path, default=None,
        help="Optional JSON file with skill_weights and category_weights"
    )
    parser.add_argument(
        "--no-explain", action="store_true",
        help="Skip LLM explanation (faster, no GROQ_API_KEY needed)"
    )
    args = parser.parse_args()

    # Validate files exist
    for r in args.resumes:
        if not r.exists():
            print(f"Error: resume file not found: {r}")
            sys.exit(1)
    if not args.jd.exists():
        print(f"Error: JD file not found: {args.jd}")
        sys.exit(1)

    # Load custom weights if provided
    skill_weights    = DEFAULT_SKILL_WEIGHTS
    category_weights = DEFAULT_CATEGORY_WEIGHTS
    if args.weights:
        w = json.loads(args.weights.read_text())
        skill_weights    = w.get("skill_weights",    skill_weights)
        category_weights = w.get("category_weights", category_weights)

    print("\n🔍  Checking servers...")
    if not check_servers():
        print("\nStart the missing servers first, then re-run connector.py")
        sys.exit(1)

    # ── Step 1: parse all resumes via Module 1 ───────────────
    print(f"\n📄  Parsing {len(args.resumes)} resume(s) via Module 1...")
    module1_outputs = []
    for resume_path in args.resumes:
        module1_outputs.append(parse_resume(resume_path))

    # ── Step 2: analyze JD via Module 2 ─────────────────────
    print(f"\n📋  Analyzing JD via Module 2...")
    module2_output = analyze_jd(args.jd)

    # ── Step 3: adapt outputs for Module 3 ──────────────────
    print(f"\n🔗  Adapting outputs for Module 3...")
    adapted_resumes = [adapt_resume(r) for r in module1_outputs]
    adapted_jd      = adapt_jd(module2_output)

    # ── Step 4: rank via Module 3 ───────────────────────────
    print(f"\n⚖️   Ranking via Module 3...")
    result = match_candidates(
        adapted_jd=adapted_jd,
        adapted_resumes=adapted_resumes,
        skill_weights=skill_weights,
        category_weights=category_weights,
        generate_explanations=not args.no_explain,
    )

    # ── Step 5: print results ────────────────────────────────
    print_results(result)


if __name__ == "__main__":
    main()
