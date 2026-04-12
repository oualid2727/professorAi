# pipeline/extractor.py

import re
from pathlib import Path


def extract_text_from_file(path: str) -> str:
    """
    Dispatch to the right extractor based on file extension.
    Returns a single string with all extracted text, or an empty string on failure.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix == ".pptx":
        return _extract_pptx(path)
    else:
        return ""


# ── PDF ──────────────────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    """
    Extract text from a PDF using pypdf.
    Falls back page-by-page so a single corrupt page doesn't kill the whole file.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pypdf is required: pip install pypdf")

    pages = []
    try:
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
                text = _clean(text)
                if text:
                    pages.append(f"[Page {i + 1}]\n{text}")
            except Exception:
                # Skip unreadable pages silently
                continue
    except Exception as e:
        return f"[PDF extraction failed: {e}]"

    return "\n\n".join(pages)


# ── PPTX ─────────────────────────────────────────────────────────────────────

def _extract_pptx(path: str) -> str:
    """
    Extract text from a PowerPoint file using python-pptx.
    Iterates every slide and every shape that holds a text frame.
    Preserves slide boundaries so downstream chunking can split on them.
    """
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError("python-pptx is required: pip install python-pptx")

    slides = []
    try:
        prs = Presentation(path)
        for i, slide in enumerate(prs.slides):
            parts = []
            for shape in slide.shapes:
                if not shape.has_text_frame:
                    continue
                for para in shape.text_frame.paragraphs:
                    line = " ".join(run.text for run in para.runs if run.text).strip()
                    if line:
                        parts.append(line)
            if parts:
                slides.append(f"[Slide {i + 1}]\n" + "\n".join(parts))
    except Exception as e:
        return f"[PPTX extraction failed: {e}]"

    return "\n\n".join(slides)


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Remove noise: excessive whitespace, null bytes, form-feed characters."""
    text = text.replace("\x00", "").replace("\f", "\n")
    # Collapse runs of 3+ newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse runs of spaces/tabs (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()