"""
Document loaders for text extraction.

Supports:
- PDF files via pypdf
- DOCX files via python-docx
- TXT/Markdown files with encoding detection
- Fireflies meeting transcripts via API
"""

from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docx_loader import DOCXLoader
from app.ingestion.loaders.txt_loader import TXTLoader
from app.ingestion.loaders.fireflies_loader import FirefliesLoader

__all__ = [
    "PDFLoader",
    "DOCXLoader",
    "TXTLoader",
    "FirefliesLoader",
]
