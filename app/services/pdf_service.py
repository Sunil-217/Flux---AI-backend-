"""Text extraction from uploaded files (PDF, Word, and plain-text/code formats)."""

import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    try:
        return "".join(page.get_text() for page in doc)
    finally:
        doc.close()  # release the file handle (avoids a leak)


def extract_text_from_docx(file_path: str) -> str:
    # Imported lazily so a missing optional dependency never crashes startup —
    # only .docx uploads fail (with a clear message) if python-docx isn't installed.
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError(
            "Word (.docx) support needs python-docx. Install it with: pip install python-docx"
        )
    doc = Document(file_path)
    parts = [p.text for p in doc.paragraphs]
    # Include table cells too — resumes/reports often use tables.
    for table in doc.tables:
        for row in table.rows:
            parts.append("\t".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def _extract_text_plain(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def extract_text_from_file(file_path: str, ext: str) -> str:
    """Extract text from a file based on its extension.
    .pdf -> PyMuPDF, .docx -> python-docx, everything else -> decode as text."""
    ext = (ext or "").lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    if ext == ".docx":
        return extract_text_from_docx(file_path)
    # .txt, .md, .csv, .json, and source-code files are read as plain text.
    return _extract_text_plain(file_path)
