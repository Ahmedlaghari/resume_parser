"""
text_cleaner.py — Normalize raw job description text before any extraction.

JDs copied from websites are full of:
  - HTML tags (<p>, <li>, <strong>, &nbsp;, etc.)
  - Unicode bullets (•, ●, ◦, –, →)
  - Windows line endings (\r\n)
  - Multiple blank lines

clean_text() returns a plain, consistent string that every other
module can safely work with.
"""

import re


# --------------------------------------------------------------------------
# Unicode / HTML bullet characters we want to turn into plain hyphens.
# Add more here if you encounter them in real JDs.
# --------------------------------------------------------------------------
BULLET_CHARS = r"[•●◦▪▸→►✓✔–—]"


def _strip_html(text: str) -> str:
    """Remove all HTML / XML tags and decode common HTML entities."""
    # Replace <br>, <li>, <p> etc. with a newline so we don't lose structure
    text = re.sub(r"<br\s*/?>|</p>|</li>|</div>", "\n", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    return text


def _normalize_bullets(text: str) -> str:
    """Replace fancy Unicode bullets with a plain hyphen-space."""
    return re.sub(BULLET_CHARS, "-", text)


def _normalize_whitespace(text: str) -> str:
    """
    - Collapse multiple spaces/tabs into a single space on each line.
    - Collapse 3+ blank lines into 2 (preserves section breaks).
    - Strip leading/trailing whitespace from every line.
    """
    # Normalize Windows line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip each line individually
    lines = [line.strip() for line in text.split("\n")]

    # Collapse runs of 3+ blank lines → 2 blank lines
    cleaned: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    return "\n".join(cleaned)


def clean_text(raw_text: str) -> str:
    """
    Main entry point.  Run the full cleaning pipeline on raw JD text.

    Usage:
        from analyzer.text_cleaner import clean_text
        clean = clean_text(raw_jd_string)
    """
    text = _strip_html(raw_text)
    text = _normalize_bullets(text)
    text = _normalize_whitespace(text)
    return text.strip()
