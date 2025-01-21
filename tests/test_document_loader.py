import os
import pytest
from unittest.mock import patch, mock_open
from src.utils.document_loader import DocumentLoader

class TestDocumentLoader:
    @pytest.fixture
    def document_loader(self):
        return DocumentLoader()

    @pytest.fixture
    def sample_text_file(self, tmp_path):
        file_path = tmp_path / "test_document.txt"
        content = "This is a test document.\nIt has multiple lines.\nFor testing purposes."
        file_path.write_text(content)
        return str(file_path)

    def test_init(self, document_loader):
        """Test DocumentLoader initialization"""
        assert isinstance(document_loader, DocumentLoader)

    def test_load_single_document(self, document_loader, sample_text_file):
        """Test loading a single document"""
        documents = document_loader.load_documents(sample_text_file)
        assert len(documents) == 1
        assert "test document" in documents[0].page_content.lower()

    def test_load_multiple_documents(self, document_loader, tmp_path):
        """Test loading multiple documents from a directory"""
        # Create multiple test files
        for i in range(3):
            file_path = tmp_path / f"test_doc_{i}.txt"
            file_path.write_text(f"Test document {i} content")

        documents = document_loader.load_documents(str(tmp_path))
        assert len(documents) == 3

    def test_load_nonexistent_file(self, document_loader):
        """Test handling of nonexistent file"""
        with pytest.raises(FileNotFoundError):
            document_loader.load_documents("nonexistent_file.txt")

    @patch('builtins.open', mock_open(read_data="Test content"))
    def test_document_content_extraction(self, document_loader):
        """Test document content extraction"""
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True
            documents = document_loader.load_documents("mock_file.txt")
            assert len(documents) == 1
            assert "Test content" in documents[0].page_content

    def test_empty_file_handling(self, document_loader, tmp_path):
        """Test handling of empty files"""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("")
        
        documents = document_loader.load_documents(str(empty_file))
        assert len(documents) == 1
        assert documents[0].page_content.strip() == ""

    def test_file_encoding(self, document_loader, tmp_path):
        """Test handling of different file encodings"""
        # Test UTF-8 with special characters
        special_chars = "Hello, 世界! äöü"
        file_path = tmp_path / "utf8_test.txt"
        file_path.write_text(special_chars, encoding='utf-8')
        
        documents = document_loader.load_documents(str(file_path))
        assert len(documents) == 1
        assert documents[0].page_content == special_chars

    def test_large_file_handling(self, document_loader, tmp_path):
        """Test handling of large files"""
        large_content = "Large content\n" * 1000
        file_path = tmp_path / "large_file.txt"
        file_path.write_text(large_content)
        
        documents = document_loader.load_documents(str(file_path))
        assert len(documents) == 1
        assert len(documents[0].page_content) == len(large_content)

    def test_metadata_extraction(self, document_loader, tmp_path):
        """Test metadata extraction from documents"""
        file_path = tmp_path / "test_doc.txt"
        content = "Test content"
        file_path.write_text(content)
        
        documents = document_loader.load_documents(str(file_path))
        assert len(documents) == 1
        assert "source" in documents[0].metadata
        assert documents[0].metadata["source"] == str(file_path)
        assert "created_at" in documents[0].metadata