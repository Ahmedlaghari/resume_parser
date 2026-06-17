"""
models.py — Pydantic schema for the Job Description output.

This is the CONTRACT between Module 2 and Module 3.
Every field here maps exactly to the JSON spec in the project doc.

Field descriptions are load-bearing: instructor injects them into the LLM
prompt as part of the JSON schema, so the model knows exactly what to fill.

Fields populated by extractor.py:
    job_title, company, location, employment_type, experience_required,
    salary_range, responsibilities, qualifications, benefits, industry

Fields populated by other modules (left as defaults by extractor.py):
    seniority_level   ← seniority.py
    required_skills   ← skill_classifier.py
    nice_to_have_skills ← skill_classifier.py
    keywords          ← KeyBERT (Module 3)
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class JobDescription(BaseModel):

    # ── Populated by extractor.py ─────────────────────────────────────────
    job_title: Optional[str] = Field(
        None,
        description=(
            "The title of the role being advertised, e.g. 'Senior ML Engineer'. "
            "Usually the first prominent line or stated after 'We are hiring a …'."
        ),
    )
    company: Optional[str] = Field(
        None,
        description=(
            "Name of the hiring company or organisation. "
            "Often found after 'at', 'join', or 'about us'."
        ),
    )
    location: Optional[str] = Field(
        None,
        description=(
            "Work location, city, country, or remote status. "
            "Examples: 'Karachi, Pakistan', 'Remote', 'Hybrid – London'."
        ),
    )
    employment_type: Optional[str] = Field(
        None,
        description=(
            "One of: Full-time, Part-time, Contract, Internship. "
            "Infer from keywords like 'permanent', 'freelance', 'intern'."
        ),
    )
    experience_required: Optional[str] = Field(
        None,
        description=(
            "Years of experience required, preserving the original phrasing. "
            "Examples: '3-5 years', '5+ years', 'minimum 3 years'."
        ),
    )
    salary_range: Optional[str] = Field(
        None,
        description=(
            "Salary or compensation range, preserving original format. "
            "Examples: '$80,000 – $100,000', 'PKR 150,000/month', '80k–100k USD'."
        ),
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description=(
            "Bullet-point list of key responsibilities from sections titled "
            "'Responsibilities', 'What You'll Do', 'Your Role', etc. "
            "Return each bullet as a separate string."
        ),
    )
    qualifications: List[str] = Field(
        default_factory=list,
        description=(
            "Bullet-point list of requirements or qualifications from sections "
            "titled 'Requirements', 'Qualifications', 'Must Have', etc. "
            "Return each bullet as a separate string."
        ),
    )
    benefits: List[str] = Field(
        default_factory=list,
        description=(
            "Bullet-point list of benefits or perks from sections titled "
            "'Benefits', 'Perks', 'What We Offer', 'Why Join Us', etc. "
            "Return each bullet as a separate string."
        ),
    )
    industry: Optional[str] = Field(
        None,
        description=(
            "The industry or domain this role belongs to. "
            "Examples: 'Technology', 'Finance', 'Healthcare', 'E-commerce', "
            "'Cybersecurity', 'Gaming', 'Telecommunications', 'Education'."
        ),
    )

    # ── Populated by other modules (extractor.py leaves these as defaults) ─
    seniority_level: Optional[str] = Field(
        None,
        description="Junior | Mid | Senior | Lead | Principal — set by seniority.py.",
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Skills explicitly required — set by skill_classifier.py.",
    )
    nice_to_have_skills: List[str] = Field(
        default_factory=list,
        description="Preferred / bonus skills — set by skill_classifier.py.",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Top-10 KeyBERT keywords — set by Module 3.",
    ) 