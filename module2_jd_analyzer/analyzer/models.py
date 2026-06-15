"""
models.py — Pydantic schema for the Job Description output.

This is the CONTRACT between Module 2 and Module 3.
Every field here maps exactly to the JSON spec in the project doc.
Optional fields default to None so missing data never crashes anything.
"""

from pydantic import BaseModel
from typing import List, Optional


class JobDescription(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    employment_type: Optional[str] = None
    seniority_level: Optional[str] = None       # Junior/Mid/Senior/Lead/Principal
    experience_required: Optional[str] = None   # e.g. "3-5 years"
    salary_range: Optional[str] = None          # may not exist in the JD
    required_skills: List[str] = []
    nice_to_have_skills: List[str] = []
    responsibilities: List[str] = []
    qualifications: List[str] = []
    keywords: List[str] = []                    # KeyBERT top-10 keywords
    industry: Optional[str] = None
    benefits: List[str] = []                    # may not exist in the JD
