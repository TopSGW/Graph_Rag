import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import mimetypes
from PIL import Image
import pytesseract
import PyPDF2
import json
from datetime import datetime
import hashlib

class DocumentLoader:
    def __init__(self, directory_path: str = None):
        self.directory_path = directory_path or os.getcwd()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        # Initialize supported file types
        self.supported_types = {
            '.txt': self._process_text_file,
            '.pdf': self._process_pdf_file,
            '.jpg': self._process_image_file,
            '.jpeg': self._process_image_file,
            '.png': self._process_image_file,
        }
        # Initialize mimetypes
        mimetypes.init()

    def _get_file_type(self, file_path: Path) -> str:
        """Detect file type using mimetypes"""
        try:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type or "application/octet-stream"
        except Exception as e:
            print(f"Error detecting file type: {str(e)}")
            return "application/octet-stream"

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Error calculating file hash: {str(e)}")
            return None

    def _extract_metadata(self, file_path: Path, content_type: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from file"""
        stats = file_path.stat()
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "title": file_path.stem,  # Removed replace('_', ' ') for consistency
            "file_type": content_type,
            "file_extension": file_path.suffix,
            "file_size": int(stats.st_size),  # Ensure integer type
            "created_at": str(int(datetime.fromtimestamp(stats.st_ctime).timestamp())),  # Store as Unix timestamp string
            "modified_at": str(int(datetime.fromtimestamp(stats.st_mtime).timestamp())),  # Store as Unix timestamp string
            "file_hash": self._calculate_file_hash(file_path),
            "content_type": content_type,
            "is_hidden": file_path.name.startswith('.'),
            # Convert path components to string for Neo4j compatibility
            "path_components": str(list(file_path.parts))
        }
        return metadata

    def _process_text_file(self, file_path: Path) -> str:
        """Process text file and return content"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def _process_pdf_file(self, file_path: Path) -> str:
        """Process PDF file and return content"""
        try:
            text = []
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            print(f"Error processing PDF file {file_path}: {str(e)}")
            return ""

    def _process_image_file(self, file_path: Path) -> str:
        """Process image file using OCR and return extracted text"""
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            print(f"Error processing image file {file_path}: {str(e)}")
            return ""

    def load_documents(self, file_pattern: str = "*.*") -> List[Document]:
        """
        Load and process documents from the specified directory
        Returns a list of Document objects ready for vector store
        """
        documents = []
        try:
            path = Path(self.directory_path)
            for file_path in path.glob(file_pattern):
                try:
                    # Skip directories
                    if file_path.is_dir():
                        continue

                    # Get file type and check if supported
                    content_type = self._get_file_type(file_path)
                    if not content_type:
                        continue

                    # Extract metadata
                    metadata = self._extract_metadata(file_path, content_type)

                    # Process file based on extension
                    extension = file_path.suffix.lower()
                    if extension in self.supported_types:
                        content = self.supported_types[extension](file_path)
                        
                        # Skip if no content extracted
                        if not content:
                            continue

                        # Preprocess content
                        content = self.preprocess_text(content)
                        
                        # Split content into chunks
                        texts = self.text_splitter.split_text(content)
                        
                        # Create Document objects for each chunk
                        for i, text in enumerate(texts):
                            chunk_metadata = metadata.copy()
                            chunk_metadata.update({
                                "chunk_index": str(i),  # Convert to string for Neo4j
                                "chunk_total": str(len(texts))  # Convert to string for Neo4j
                            })
                            
                            doc = Document(
                                page_content=text,
                                metadata=chunk_metadata
                            )
                            documents.append(doc)
                            
                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            return documents
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the loaded documents"""
        try:
            path = Path(self.directory_path)
            files = []
            total_size = 0
            file_types = {}
            
            for file_path in path.glob("*.*"):
                if file_path.is_file():
                    content_type = self._get_file_type(file_path)
                    size = file_path.stat().st_size
                    
                    file_info = {
                        "name": file_path.name,
                        "size": str(size),  # Convert to string for Neo4j
                        "content_type": content_type,
                        "last_modified": str(int(datetime.fromtimestamp(file_path.stat().st_mtime).timestamp())),
                        "file_hash": self._calculate_file_hash(file_path)
                    }
                    files.append(file_info)
                    total_size += size
                    
                    # Track file types
                    if content_type in file_types:
                        file_types[content_type] = str(int(file_types[content_type]) + 1)
                    else:
                        file_types[content_type] = "1"
            
            stats = {
                "total_files": str(len(files)),  # Convert to string for Neo4j
                "total_size": str(total_size),  # Convert to string for Neo4j
                "file_types": json.dumps(file_types),  # Convert dict to JSON string for Neo4j
                "files": json.dumps(files)  # Convert list to JSON string for Neo4j
            }
            return stats
        except Exception as e:
            print(f"Error getting document stats: {str(e)}")
            return {
                "total_files": "0",
                "total_size": "0",
                "file_types": "{}",
                "files": "[]"
            }

    def read_single_document(self, filename: str) -> Optional[Document]:
        """Read a single document by filename and return as Document object"""
        try:
            file_path = Path(self.directory_path) / filename
            if not file_path.exists():
                raise FileNotFoundError(f"File {filename} not found")
            
            # Get file type and check if supported
            content_type = self._get_file_type(file_path)
            if not content_type:
                return None

            # Extract metadata
            metadata = self._extract_metadata(file_path, content_type)
            
            # Process file based on extension
            extension = file_path.suffix.lower()
            if extension in self.supported_types:
                content = self.supported_types[extension](file_path)
                if content:
                    content = self.preprocess_text(content)
                    return Document(
                        page_content=content,
                        metadata=metadata
                    )
            return None
        except Exception as e:
            print(f"Error reading document {filename}: {str(e)}")
            return None

    def get_document_titles(self) -> List[str]:
        """Get a list of document titles with their types"""
        try:
            path = Path(self.directory_path)
            titles = []
            for file_path in path.glob("*.*"):
                if file_path.is_file() and file_path.suffix.lower() in self.supported_types:
                    titles.append(file_path.stem)
            return titles
        except Exception as e:
            print(f"Error getting document titles: {str(e)}")
            return []

    def preprocess_text(self, text: str) -> str:
        """Preprocess text content for better quality"""
        # Remove multiple newlines
        text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
        # Remove multiple spaces
        text = ' '.join(text.split())
        return text