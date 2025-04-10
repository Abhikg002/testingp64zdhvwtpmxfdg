import pdfplumber
import docx
import os

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

def extract_text_from_docx(docx_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_txt(txt_path):
    """Extract text from a TXT file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

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
        raise ValueError("Unsupported job description file format! Use TXT, PDF, or DOCX.")

def parse_resume(file_path):
    """Detect resume file type and extract text accordingly (PDF/DOCX only)."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported resume format! Use PDF or DOCX.")
