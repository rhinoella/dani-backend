"""
TXT document loader with encoding detection.

Extracts text content from plain text files with smart encoding handling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional
import chardet

logger = logging.getLogger(__name__)


class TXTLoader:
    """
    Plain text file loader with encoding detection.
    
    Handles various text encodings (UTF-8, Latin-1, etc.) and
    provides text suitable for chunking.
    """
    
    def __init__(self) -> None:
        self.supported_extensions = [".txt", ".text", ".md", ".markdown"]
        self.default_encoding = "utf-8"
        self.fallback_encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    
    def load(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        Load and extract text from a TXT file.
        
        Args:
            file_content: Raw bytes of the text file
            filename: Original filename for metadata
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        logger.info(f"Loading TXT: {filename}")
        
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_content)
            
            # Decode text
            text = self._decode_text(file_content, encoding)
            
            # Clean text
            text = self._clean_text(text)
            
            # Calculate metadata
            lines = text.split("\n")
            word_count = len(text.split())
            char_count = len(text)
            
            metadata = {
                "filename": filename,
                "encoding": encoding,
                "line_count": len(lines),
                "word_count": word_count,
                "character_count": char_count,
            }
            
            # Check if it's markdown
            if filename.lower().endswith((".md", ".markdown")):
                metadata["format"] = "markdown"
                # Extract title from first heading if present
                title = self._extract_markdown_title(text)
                if title:
                    metadata["title"] = title
            else:
                metadata["format"] = "plain_text"
            
            logger.info(f"Extracted {len(lines)} lines, {word_count} words from TXT")
            
            return {
                "text": text,
                "lines": lines,
                "metadata": metadata,
                "line_count": len(lines),
                "word_count": word_count,
            }
            
        except Exception as e:
            logger.error(f"Failed to load TXT {filename}: {e}")
            raise ValueError(f"Failed to parse text file: {str(e)}") from e
    
    def _detect_encoding(self, content: bytes) -> str:
        """Detect the encoding of the text file."""
        try:
            result = chardet.detect(content)
            encoding = result.get("encoding", self.default_encoding)
            confidence = result.get("confidence", 0)
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence})")
            
            # If confidence is low, fallback to UTF-8
            if confidence < 0.5:
                return self.default_encoding
            
            return encoding or self.default_encoding
            
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using UTF-8")
            return self.default_encoding
    
    def _decode_text(self, content: bytes, encoding: str) -> str:
        """Decode bytes to string, trying multiple encodings if needed."""
        # Try the detected encoding first
        try:
            return content.decode(encoding)
        except (UnicodeDecodeError, LookupError) as e:
            logger.warning(f"Failed to decode with {encoding}: {e}")
        
        # Try fallback encodings
        for fallback_encoding in self.fallback_encodings:
            if fallback_encoding.lower() == encoding.lower():
                continue
            try:
                return content.decode(fallback_encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: decode with errors='replace'
        logger.warning("All encodings failed, using UTF-8 with replacement")
        return content.decode("utf-8", errors="replace")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove null bytes
        text = text.replace("\x00", "")
        
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Remove excessive whitespace while preserving paragraph structure
        lines = text.split("\n")
        cleaned_lines = []
        empty_count = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                empty_count += 1
                # Allow max 2 consecutive empty lines
                if empty_count <= 2:
                    cleaned_lines.append("")
            else:
                empty_count = 0
                cleaned_lines.append(stripped)
        
        return "\n".join(cleaned_lines).strip()
    
    def _extract_markdown_title(self, text: str) -> Optional[str]:
        """Extract title from markdown heading."""
        lines = text.split("\n")
        
        for line in lines[:10]:  # Check first 10 lines
            stripped = line.strip()
            if stripped.startswith("# "):
                return stripped[2:].strip()
        
        return None
    
    def load_paragraphs(self, file_content: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Load TXT and return paragraph-based records for chunking.
        
        Paragraphs are split by blank lines.
        
        Args:
            file_content: Raw bytes of the text file
            filename: Original filename for metadata
            
        Returns:
            List of paragraph records suitable for chunking
        """
        result = self.load(file_content, filename)
        text = result["text"]
        
        if not text:
            return []
        
        # Split by double newlines (paragraphs)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        records = []
        for idx, para in enumerate(paragraphs):
            records.append({
                "text": para,
                "metadata": {
                    "source_file": filename,
                    "paragraph_index": idx,
                    "total_paragraphs": len(paragraphs),
                    **result["metadata"],
                }
            })
        
        return records
