"""
Tests for Document Loaders (PDF, DOCX, TXT).

Tests text extraction from various document formats.
"""

import pytest
from unittest.mock import MagicMock, patch
from io import BytesIO

from app.ingestion.loaders.pdf_loader import PDFLoader
from app.ingestion.loaders.docx_loader import DOCXLoader
from app.ingestion.loaders.txt_loader import TXTLoader


# ============== PDF Loader Tests ==============

class TestPDFLoader:
    """Tests for PDFLoader class."""
    
    def test_init(self):
        """Test PDFLoader initialization."""
        loader = PDFLoader()
        assert loader.supported_extensions == [".pdf"]
    
    @patch('app.ingestion.loaders.pdf_loader.PdfReader')
    def test_load_extracts_text(self, mock_pdf_reader_class):
        """Test loading PDF extracts text."""
        # Setup mock pages
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2 content"
        
        # Setup mock reader instance
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader.metadata = {}
        mock_pdf_reader_class.return_value = mock_reader
        
        loader = PDFLoader()
        result = loader.load(b"fake pdf content", "test.pdf")
        
        assert "Page 1 content" in result["text"]
        assert "Page 2 content" in result["text"]
        assert result["page_count"] == 2
    
    @patch('app.ingestion.loaders.pdf_loader.PdfReader')
    def test_load_pages(self, mock_pdf_reader_class):
        """Test loading PDF pages individually."""
        mock_page1 = MagicMock()
        mock_page1.extract_text.return_value = "Page 1"
        mock_page2 = MagicMock()
        mock_page2.extract_text.return_value = "Page 2"
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page1, mock_page2]
        mock_reader.metadata = {}
        mock_pdf_reader_class.return_value = mock_reader
        
        loader = PDFLoader()
        pages = loader.load_pages(b"fake pdf", "test.pdf")
        
        assert len(pages) == 2
        assert pages[0]["text"] == "Page 1"
        assert pages[0]["metadata"]["page_number"] == 1
        assert pages[1]["text"] == "Page 2"
        assert pages[1]["metadata"]["page_number"] == 2
    
    @patch('app.ingestion.loaders.pdf_loader.PdfReader')
    def test_load_handles_empty_pages(self, mock_pdf_reader_class):
        """Test loader handles empty pages gracefully."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {}
        mock_pdf_reader_class.return_value = mock_reader
        
        loader = PDFLoader()
        result = loader.load(b"fake pdf", "test.pdf")
        
        assert result["text"] == ""
        assert result["page_count"] == 1
    
    @patch('app.ingestion.loaders.pdf_loader.PdfReader')
    def test_extract_metadata(self, mock_pdf_reader_class):
        """Test PDF metadata extraction."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader.metadata = {
            "/Title": "Test Document",
            "/Author": "Test Author",
        }
        mock_pdf_reader_class.return_value = mock_reader
        
        loader = PDFLoader()
        result = loader.load(b"fake pdf", "test.pdf")
        
        assert result["metadata"]["title"] == "Test Document"
        assert result["metadata"]["author"] == "Test Author"
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        loader = PDFLoader()
        assert ".pdf" in loader.supported_extensions


# ============== DOCX Loader Tests ==============

class TestDOCXLoader:
    """Tests for DOCXLoader class."""
    
    def test_init(self):
        """Test DOCXLoader initialization."""
        loader = DOCXLoader()
        assert ".docx" in loader.supported_extensions
    
    @patch('app.ingestion.loaders.docx_loader.DocxDocument')
    def test_load_extracts_text(self, mock_docx_document_class):
        """Test loading DOCX extracts text."""
        # Setup mock paragraphs with style
        mock_para1 = MagicMock()
        mock_para1.text = "Paragraph 1"
        mock_para1.style.name = "Normal"
        mock_para2 = MagicMock()
        mock_para2.text = "Paragraph 2"
        mock_para2.style.name = "Normal"
        
        # Setup mock document
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []
        mock_doc.core_properties.title = "Test Doc"
        mock_doc.core_properties.author = "Test Author"
        mock_doc.core_properties.subject = None
        mock_doc.core_properties.created = None
        mock_doc.core_properties.modified = None
        mock_doc.core_properties.last_modified_by = None
        mock_doc.core_properties.keywords = None
        mock_doc.core_properties.comments = None
        mock_docx_document_class.return_value = mock_doc
        
        loader = DOCXLoader()
        result = loader.load(b"fake docx", "test.docx")
        
        assert "Paragraph 1" in result["text"]
        assert "Paragraph 2" in result["text"]
    
    @patch('app.ingestion.loaders.docx_loader.DocxDocument')
    def test_load_sections(self, mock_docx_document_class):
        """Test loading DOCX sections with headings."""
        # Setup mock with heading
        mock_heading = MagicMock()
        mock_heading.text = "Section Title"
        mock_heading.style.name = "Heading 1"
        mock_para = MagicMock()
        mock_para.text = "Section content"
        mock_para.style.name = "Normal"
        
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_heading, mock_para]
        mock_doc.tables = []
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.subject = None
        mock_doc.core_properties.created = None
        mock_doc.core_properties.modified = None
        mock_doc.core_properties.last_modified_by = None
        mock_doc.core_properties.keywords = None
        mock_doc.core_properties.comments = None
        mock_docx_document_class.return_value = mock_doc
        
        loader = DOCXLoader()
        sections = loader.load_sections(b"fake docx", "test.docx")
        
        assert len(sections) >= 1
    
    @patch('app.ingestion.loaders.docx_loader.DocxDocument')
    def test_load_with_tables(self, mock_docx_document_class):
        """Test DOCX loader handles tables."""
        mock_cell1 = MagicMock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = MagicMock()
        mock_cell2.text = "Cell 2"
        
        mock_row = MagicMock()
        mock_row.cells = [mock_cell1, mock_cell2]
        
        mock_table = MagicMock()
        mock_table.rows = [mock_row]
        
        mock_doc = MagicMock()
        mock_doc.paragraphs = []
        mock_doc.tables = [mock_table]
        mock_doc.core_properties.title = None
        mock_doc.core_properties.author = None
        mock_doc.core_properties.subject = None
        mock_doc.core_properties.created = None
        mock_doc.core_properties.modified = None
        mock_doc.core_properties.last_modified_by = None
        mock_doc.core_properties.keywords = None
        mock_doc.core_properties.comments = None
        mock_docx_document_class.return_value = mock_doc
        
        loader = DOCXLoader()
        result = loader.load(b"fake docx", "test.docx")
        
        # Table content should be included
        assert "Cell 1" in result["text"]
        assert "Cell 2" in result["text"]

    def test_supported_extensions(self):
        """Test supported file extensions."""
        loader = DOCXLoader()
        assert ".docx" in loader.supported_extensions


# ============== TXT Loader Tests ==============

class TestTXTLoader:
    """Tests for TXTLoader class."""
    
    def test_init(self):
        """Test TXTLoader initialization."""
        loader = TXTLoader()
        assert ".txt" in loader.supported_extensions
        assert ".md" in loader.supported_extensions
    
    def test_load_from_utf8_content(self):
        """Test loading UTF-8 text content."""
        content = "Hello, World!\nThis is a test.".encode('utf-8')
        loader = TXTLoader()
        result = loader.load(content, "test.txt")
        
        assert "Hello, World!" in result["text"]
        assert "This is a test." in result["text"]
    
    def test_load_from_ascii_content(self):
        """Test loading ASCII text content."""
        content = b"Simple ASCII text"
        loader = TXTLoader()
        result = loader.load(content, "test.txt")
        
        assert "Simple ASCII text" in result["text"]
    
    def test_load_handles_empty_content(self):
        """Test loader handles empty content."""
        loader = TXTLoader()
        result = loader.load(b"", "empty.txt")
        
        assert result["text"] == ""
    
    def test_load_paragraphs(self):
        """Test loading text as paragraphs."""
        content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.".encode('utf-8')
        loader = TXTLoader()
        paragraphs = loader.load_paragraphs(content, "test.txt")
        
        assert len(paragraphs) == 3
        assert paragraphs[0]["text"] == "Paragraph 1."
        assert paragraphs[1]["text"] == "Paragraph 2."
        assert paragraphs[2]["text"] == "Paragraph 3."
    
    def test_metadata_includes_char_count(self):
        """Test metadata includes character count."""
        content = "Hello World".encode('utf-8')
        loader = TXTLoader()
        result = loader.load(content, "test.txt")
        
        assert "character_count" in result["metadata"]
        assert result["metadata"]["character_count"] == 11
    
    def test_metadata_includes_line_count(self):
        """Test metadata includes line count."""
        content = "Line 1\nLine 2\nLine 3".encode('utf-8')
        loader = TXTLoader()
        result = loader.load(content, "test.txt")
        
        assert "line_count" in result["metadata"]
        assert result["metadata"]["line_count"] == 3
    
    def test_encoding_detection(self):
        """Test encoding detection works."""
        # UTF-8 encoded text
        content = "Héllo Wörld".encode('utf-8')
        loader = TXTLoader()
        result = loader.load(content, "test.txt")
        
        assert "Héllo Wörld" in result["text"]
    
    def test_supported_extensions(self):
        """Test supported file extensions."""
        loader = TXTLoader()
        assert ".txt" in loader.supported_extensions
        assert ".md" in loader.supported_extensions
        assert ".markdown" in loader.supported_extensions


# ============== Integration Tests ==============

class TestLoaderIntegration:
    """Integration tests for document loaders."""
    
    def test_pdf_loader_available(self):
        """Test PDF loader can be imported."""
        from app.ingestion.loaders import PDFLoader
        assert PDFLoader is not None
    
    def test_docx_loader_available(self):
        """Test DOCX loader can be imported."""
        from app.ingestion.loaders import DOCXLoader
        assert DOCXLoader is not None
    
    def test_txt_loader_available(self):
        """Test TXT loader can be imported."""
        from app.ingestion.loaders import TXTLoader
        assert TXTLoader is not None
    
    def test_all_loaders_have_load_method(self):
        """Test all loaders have load method."""
        pdf_loader = PDFLoader()
        docx_loader = DOCXLoader()
        txt_loader = TXTLoader()
        
        assert hasattr(pdf_loader, 'load')
        assert hasattr(docx_loader, 'load')
        assert hasattr(txt_loader, 'load')
    
    def test_txt_loader_real_content(self):
        """Test TXT loader with real content."""
        content = """# Document Title

This is the first paragraph with some content.

## Section 1

This is section 1 content.

## Section 2

This is section 2 content.
"""
        loader = TXTLoader()
        result = loader.load(content.encode('utf-8'), "document.md")
        
        assert "Document Title" in result["text"]
        assert "Section 1" in result["text"]
        assert result["metadata"]["filename"] == "document.md"
