from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag.config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

class Document:
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}

class PDFDocumentLoader:
    def __init__(self, directory_path: str):
        self.directory_path = Path(directory_path)
        if not self.directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
    
    def load(self) -> List[Any]:
        documents = []
        pdf_files = list(self.directory_path.glob("**/*.pdf"))
        
        if not pdf_files:
            return documents
        
        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                docs = loader.load()
                documents.extend(docs)
            except Exception:
                pass
        
        return documents


class TextSplitter:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split(self, documents: List[Any]) -> List[Any]:
        if not documents:
            return []
        
        return self.splitter.split_documents(documents)
