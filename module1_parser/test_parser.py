# ============================================================
# test_parser.py — Test the parser WITHOUT starting the server
# ============================================================
# Run: python test_parser.py
#
# This lets you quickly test extraction logic on any text file.
# Useful during development before testing the full API.
# ============================================================

import json
import sys
import os

# Make sure Python can find our parser package
sys.path.insert(0, os.path.dirname(__file__))

from parser.extractor import extract_resume_data


def test_from_text_file(filepath: str):
    """Read a plain .txt file and run the parser on it."""
    print(f"\n{'='*60}")
    print(f"Testing with: {filepath}")
    print('='*60)

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"\n[Raw Text Preview — first 300 chars]\n{raw_text[:300]}...\n")

    # Run the parser
    result = extract_resume_data(raw_text)

    # Print as formatted JSON
    print("[Parsed Output]")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    # Test with the sample resume
    sample_path = os.path.join(
        os.path.dirname(__file__),
        "sample_resumes",
        "sample_resume.txt"
    )

    if os.path.exists(sample_path):
        test_from_text_file(sample_path)
    else:
        print(f"Sample file not found at: {sample_path}")
        print("Create a .txt file in sample_resumes/ and try again.")
