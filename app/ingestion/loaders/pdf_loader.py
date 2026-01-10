"""
PDF document loader using pypdf.

Extracts text content from PDF files with page-level metadata.
"""

from __future__ import annotations

import io
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PDFPage:
    """Represents a single page from a PDF."""
    page_number: int
    text: str
    metadata: Dict[str, Any]


class PDFLoader:
    """
    PDF document loader using pypdf.
    
    Extracts text from PDF files, preserving page boundaries
    for better context in retrieval.
    """
    
    def __init__(self) -> None:
        self.supported_extensions = [".pdf"]
    
    def load(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Load and extract text from a PDF file.
        
        Args:
            file_content: Raw bytes of the PDF file
            filename: Original filename for metadata
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        logger.info(f"Loading PDF: {filename}")
        
        try:
            reader = PdfReader(io.BytesIO(file_content))
            
            # Extract document metadata
            doc_metadata = self._extract_metadata(reader)
            doc_metadata["filename"] = filename
            doc_metadata["total_pages"] = len(reader.pages)
            
            # Extract text from all pages
            pages: List[PDFPage] = []
            full_text_parts: List[str] = []
            
            for page_num, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    
                    if text:
                        pages.append(PDFPage(
                            page_number=page_num,
                            text=text,
                            metadata={"page": page_num}
                        ))
                        full_text_parts.append(f"[Page {page_num}]\n{text}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            full_text = "\n\n".join(full_text_parts)
            
            logger.info(f"Extracted {len(pages)} pages from PDF")
            
            return {
                "text": full_text,
                "pages": [
                    {"page_number": p.page_number, "text": p.text}
                    for p in pages
                ],
                "metadata": doc_metadata,
                "page_count": len(reader.pages),
                "extracted_pages": len(pages),
            }
            
        except Exception as e:
            logger.error(f"Failed to load PDF {filename}: {e}")
            raise ValueError(f"Failed to parse PDF: {str(e)}") from e
    
    def _extract_metadata(self, reader: PdfReader) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
        metadata: Dict[str, Any] = {}
        
        if reader.metadata:
            # Common PDF metadata fields
            field_mapping = {
                "/Title": "title",
                "/Author": "author",
                "/Subject": "subject",
                "/Creator": "creator",
                "/Producer": "producer",
                "/CreationDate": "creation_date",
                "/ModDate": "modification_date",
            }
            
            for pdf_field, our_field in field_mapping.items():
                value = reader.metadata.get(pdf_field)
                if value:
                    # Clean up the value (remove null bytes, etc.)
                    if isinstance(value, str):
                        value = value.replace('\x00', '').strip()
                    metadata[our_field] = value
        
        return metadata
    
    def load_pages(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Load PDF and return individual page records for chunking.
        
        This method is useful when you want to process pages separately.
        
        Args:
            file_content: Raw bytes of the PDF file
            filename: Original filename for metadata
            
        Returns:
            List of page records suitable for chunking
        """
        result = self.load(file_content, filename)
        
        records = []
        for page in result["pages"]:
            records.append({
                "text": page["text"],
                "metadata": {
                    "source_file": filename,
                    "page_number": page["page_number"],
                    "total_pages": result["page_count"],
                    **result["metadata"],
                }
            })
        
        return records
