import os
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import time

from dotenv import load_dotenv
import pyinputplus as pyip
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Configuration constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
API_KEY = os.getenv("GEMINI_API_KEY")


class Document:
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = content
        self.metadata = metadata or {}


class FAISSVectorStore:
    def __init__(self, documents: List[Any], embedding_model: Any):
        self.store = FAISS.from_documents(documents, embedding_model)
    
    def similarity_search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
        docs = self.store.similarity_search(query, k=k)
        return [Document(doc.page_content, doc.metadata) for doc in docs]
    
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.store.save_local(path)
    
    @classmethod
    def load(cls, path: str, embedding_model: Any) -> "FAISSVectorStore":
        instance = cls.__new__(cls)
        instance.store = FAISS.load_local(path, embedding_model)
        return instance

    def as_retriever(self):
        return self.store.as_retriever()


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


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )


class LLM:
    def __init__(self, model_name: str = LLM_MODEL, api_key: str = API_KEY, temperature: float = DEFAULT_TEMPERATURE):
        if not api_key:
            raise ValueError("API key not found. Set the GEMINI_API_KEY environment variable.")
        
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.initialize_model()
    
    def initialize_model(self):
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name=self.model_name)


class ChatLLM:
    def __init__(self, model_name: str = LLM_MODEL, api_key: str = API_KEY, temperature: float = DEFAULT_TEMPERATURE):
        if not api_key:
            raise ValueError("API key not found. Set the GEMINI_API_KEY environment variable.")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            system_prompt=(
                "You are a 1337 Coding School Staff member. Provide accurate and helpful "
                "information about the school based on the provided context. For ANY question "
                "not specifically about 1337 Coding School or not found in your context, "
                "respond EXACTLY with: 'I don't know, I'm a 1337 AI Assistant and I can only "
                "answer questions related to the school'. Never attempt to answer general knowledge "
                "questions, political questions, or anything unrelated to 1337 Coding School."
            )
        )


class ChatChain:
    def __init__(self, chat_llm: ChatLLM, vector_store: FAISSVectorStore):
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=chat_llm.llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=False
        )
    
    def invoke(self, query: str, chat_history: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        chat_history = chat_history or []
        return self.chain.invoke({
            "question": query,
            "chat_history": chat_history
        })


class RAGAssistant:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(RAGAssistant, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, data_dir: str):
        if RAGAssistant._initialized:
            return
            
        try:
            # Initialize document processing pipeline
            document_loader = PDFDocumentLoader(data_dir)
            documents = document_loader.load()
            
            text_splitter = TextSplitter()
            chunks = text_splitter.split(documents)
            
            embedding_model = EmbeddingModel()
            
            # Check if vector store already exists
            store_path = Path(VECTOR_STORE_PATH)
            
            if store_path.exists():
                try:
                    self.vector_store = FAISSVectorStore.load(str(store_path), embedding_model.model)
                except Exception:
                    if not documents:
                        raise ValueError("No documents provided to create vector store")
                    
                    self.vector_store = FAISSVectorStore(chunks, embedding_model.model)
                    self.vector_store.save(str(store_path))
            else:
                if not documents:
                    raise ValueError("No documents provided to create vector store")
                
                self.vector_store = FAISSVectorStore(chunks, embedding_model.model)
                self.vector_store.save(str(store_path))
            
            # Initialize LLM components
            chat_llm = ChatLLM()
            self.chat_chain = ChatChain(chat_llm, self.vector_store)
            
            self.chat_history = []
            RAGAssistant._initialized = True
        except Exception as e:
            raise
    
    def get_response(self, query: str) -> str:
        try:
            response = self.chat_chain.invoke(query, self.chat_history)
            answer = response.get('answer', 'No answer found')
            return answer
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def run_chat_session(self):
        print("Hello! I'm a virtual assistant for the 1337 school. Ask me anything about the school.")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        
        while True:
            try:
                query = pyip.inputStr(prompt="You: ", blank=False)
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Assistant: Goodbye!")
                    break
                
                response = self.get_response(query)
                print(f"Assistant: {response}")
                
                self.chat_history.append((query, response))
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")


def main():
    try:
        data_dir = os.getenv("DATA_DIR", "knowledge_base/")
        assistant = RAGAssistant(data_dir)
        assistant.run_chat_session()
    except Exception as e:
        print(f"Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()