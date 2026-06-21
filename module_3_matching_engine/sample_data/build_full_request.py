"""
build_full_request.py
-----------------------
Glues sample_jd.json + every file in sample_resumes/ into one request
body, with the custom weights from the spec's worked example baked in.
Run this once to produce full_request_example.json, then POST that
file to /match-candidates.

Usage:
    python sample_data/build_full_request.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).parent

jd = json.loads((HERE / "sample_jd.json").read_text())

candidates = []
for f in sorted((HERE / "sample_resumes").glob("*.json")):
    candidates.append(json.loads(f.read_text()))

request = {
    "job_description": jd,
    "candidates": candidates,
    "category_weights": {
        "skills_weight": 0.5,
        "experience_weight": 0.25,
        "education_weight": 0.15,
        "semantic_weight": 0.10,
    },
    "skill_weights": {
        "Python": 1.0,
        "Java": 0.3,
        "Docker": 0.7,
        "AWS": 0.6,
        "PyTorch": 0.9,
    },
    "generate_explanations": True,
}

out_path = HERE / "full_request_example.json"
out_path.write_text(json.dumps(request, indent=2))
print(f"Wrote {out_path} with {len(candidates)} candidates.")
