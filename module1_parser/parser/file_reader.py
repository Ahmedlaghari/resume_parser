# ============================================================
# file_reader.py — Extract Raw Text from PDF and DOCX Files
# ============================================================
# This module has one job: given a file path, return all the
# text inside it as a single plain string.
#
# We use:
#   pdfplumber → reads PDFs, handles multi-column layouts well
#   python-docx → reads .docx Word files
# ============================================================

import pdfplumber
import docx
import os


def read_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.

    pdfplumber opens each page and extracts text, preserving
    line breaks. We join all pages with newlines between them.

    Returns:
        A single string with all text from the PDF.
    Raises:
        ValueError if the file can't be read or is empty.
    """
    try:
        text_parts = []

        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()

                if page_text:  # Some pages might be images (no text)
                    text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()

        if not full_text:
            raise ValueError(
                "PDF appears to contain no extractable text. "
                "It may be a scanned image — OCR support is not included in this module."
            )

        return full_text

    except pdfplumber.pdfminer.pdfparser.PDFSyntaxError:
        raise ValueError(f"File is not a valid PDF: {file_path}")
    except Exception as e:
        raise ValueError(f"Could not read PDF file: {str(e)}")


def read_docx(file_path: str) -> str:
    """
    Extract all text from a DOCX (Word) file.

    python-docx reads the document's paragraphs. Each paragraph
    is a block of text (like a <p> tag in HTML).

    Returns:
        A single string with all text from the DOCX.
    Raises:
        ValueError if the file can't be read or is empty.
    """
    try:
        doc = docx.Document(file_path)

        # Each paragraph is one text block; filter empty ones
        paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]

        full_text = "\n".join(paragraphs).strip()

        if not full_text:
            raise ValueError("DOCX file appears to be empty or has no readable text.")

        return full_text

    except Exception as e:
        raise ValueError(f"Could not read DOCX file: {str(e)}")


def extract_text(file_path: str) -> str:
    """
    Unified entry point: detects file type and calls the right reader.

    Args:
        file_path: Path to the resume file (PDF or DOCX).

    Returns:
        Raw text string from the file.
    Raises:
        ValueError for unsupported file types or read errors.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".pdf":
        return read_pdf(file_path)
    elif extension in (".docx", ".doc"):
        return read_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            "Please upload a PDF or DOCX file."
        )
