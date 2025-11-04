from typing import List, Any
from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from rag.config import EMBEDDING_MODEL, DATABASE_URL
from rag.document_processing import Document
import psycopg2


class PGVectorStore:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.embedding_model = EmbeddingModel().model
        self.connection_string = DATABASE_URL
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        try:
            self.store = PGVector(
                connection_string=self.connection_string,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
            )
        except psycopg2.OperationalError as e:
            raise ConnectionError(f"Could not connect to the database: {e}")

    def add_documents(self, documents: List[Any]):
        self.store.add_documents(documents)

    def as_retriever(self):
        return self.store.as_retriever()


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: str = "cpu"):
        self.model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device}
        )
