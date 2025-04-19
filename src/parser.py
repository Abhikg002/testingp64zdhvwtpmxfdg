import pdfplumber
import docx
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    except Exception as e:
        print(f"⚠️ Failed to extract text from PDF: {pdf_path} — {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"⚠️ Failed to extract text from DOCX: {docx_path} — {e}")
        return ""

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file."""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"⚠️ Failed to extract text from TXT: {txt_path} — {e}")
        return ""

def extract_text_from_file(file_path):
    """
    General-purpose extractor for job description files (txt, pdf, docx).
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".txt":
        return extract_text_from_txt(file_path)
    elif ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        print(f"⚠️ Unsupported job description file format: {file_path}")
        return ""

def parse_resume(file_path):
    """Detect resume file type and extract text accordingly (PDF/DOCX only)."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        print(f"⚠️ Unsupported resume file format: {file_path}")
        return ""
