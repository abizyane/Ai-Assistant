import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self, pdf_path):
        self.pdf_path = self._validate_pdf_path(pdf_path)
        self.vector_store = self._initialize_vector_store()
    
    def _validate_pdf_path(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        return pdf_path
    
    def _load_and_split_documents(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(documents)
    
    def _initialize_vector_store(self):
        texts = self._load_and_split_documents()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_documents(texts, embeddings)


def main():
    try:
        doc_processor = DocumentProcessor("data/knowledgeBase.pdf")
        
    except Exception as e:
        print(f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()