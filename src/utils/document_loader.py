import os
from typing import List, Dict, Any
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

class DocumentLoader:
    def __init__(self, directory_path: str = None):
        self.directory_path = directory_path or os.getcwd()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def load_documents(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        Load and process documents from the specified directory
        Returns a list of Document objects ready for vector store
        """
        documents = []
        try:
            path = Path(self.directory_path)
            for file_path in path.glob(file_pattern):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        # Create base metadata for the document
                        base_metadata = {
                            "source": str(file_path),
                            "filename": file_path.name,
                            "title": file_path.stem.replace('_', ' '),
                            "file_type": file_path.suffix,
                            "created_at": str(file_path.stat().st_ctime),
                            "modified_at": str(file_path.stat().st_mtime)
                        }
                        
                        # Split content into chunks
                        texts = self.text_splitter.split_text(content)
                        
                        # Create Document objects for each chunk
                        for i, text in enumerate(texts):
                            chunk_metadata = base_metadata.copy()
                            chunk_metadata.update({
                                "chunk_index": i,
                                "chunk_total": len(texts)
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
        """Get statistics about the loaded documents"""
        try:
            path = Path(self.directory_path)
            files = list(path.glob("*.txt"))
            
            stats = {
                "total_files": len(files),
                "files": [
                    {
                        "name": f.name,
                        "size": f.stat().st_size,
                        "last_modified": str(f.stat().st_mtime)
                    }
                    for f in files
                ],
                "total_size": sum(f.stat().st_size for f in files)
            }
            return stats
        except Exception as e:
            print(f"Error getting document stats: {str(e)}")
            return {
                "total_files": 0,
                "files": [],
                "total_size": 0
            }

    def read_single_document(self, filename: str) -> Document:
        """Read a single document by filename and return as Document object"""
        try:
            file_path = Path(self.directory_path) / filename
            if not file_path.exists():
                raise FileNotFoundError(f"File {filename} not found")
            
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                metadata = {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "title": file_path.stem.replace('_', ' '),
                    "file_type": file_path.suffix,
                    "created_at": str(file_path.stat().st_ctime),
                    "modified_at": str(file_path.stat().st_mtime)
                }
                
                return Document(
                    page_content=content,
                    metadata=metadata
                )
        except Exception as e:
            print(f"Error reading document {filename}: {str(e)}")
            return None

    def get_document_titles(self) -> List[str]:
        """Get a list of document titles (filenames without extension)"""
        try:
            path = Path(self.directory_path)
            return [f.stem.replace('_', ' ') for f in path.glob("*.txt")]
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