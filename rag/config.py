import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 5
VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.0-flash-exp")
API_KEY = os.getenv("GEMINI_API_KEY")
DATA_DIR = os.getenv("DATA_DIR", "rag/knowledge_base")
DATABASE_URL = os.getenv("DATABASE_URL")
