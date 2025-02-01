import os
from typing import List, Tuple
from django.conf import settings
from .models import Message, Conversation
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RAGService:
    _instance = None
    _is_initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RAGService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self.vector_store = None
            self.model = None
            self._initialize_rag()
            RAGService._is_initialized = True

    def _initialize_rag(self):
        genai.configure(api_key=settings.RAG_SETTINGS['GEMINI_API_KEY'])
        self.model = genai.GenerativeModel(model_name=settings.RAG_SETTINGS['MODEL_NAME'])
        self._load_and_process_documents()

    def _load_and_process_documents(self):
        docs_path = os.path.join(settings.BASE_DIR, 'knowledge_base')
        documents = []

        for filename in os.listdir(docs_path):
            if filename.endswith('.pdf'):
                file_path = os.path.join(docs_path, filename)
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
        )
        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        self.vector_store = FAISS.from_documents(texts, embeddings)

    def _get_relevant_context(self, query: str) -> str:
        docs = self.vector_store.similarity_search(query, k=10)
        return "\n".join(doc.page_content for doc in docs)

    def _format_chat_history(self, conversation: Conversation) -> List[Tuple[str, str]]:
        messages = conversation.messages.order_by('created_at')
        return [(msg.question, msg.answer) for msg in messages]

    def _create_prompt(self, query: str, context: str, conversation: Conversation) -> str:
        chat_history = self._format_chat_history(conversation)
        
        history_text = ""
        if chat_history:
            history_text = "Previous conversation:\n"
            for question, answer in chat_history[-5:]:
                history_text += f"User: {question}\nAssistant: {answer}\n"

        prompt = f"""You are a helpful assistant for the 1337 coding school. Use the following context to answer the question.
If you don't know the answer or can't find it in the context, just say "I don't know" and offer to help with something else.

Context:
{context}

{history_text}
Current question: {query}

Please provide a clear and concise answer based on the context provided."""

        return prompt

    def get_response(self, query: str, conversation: Conversation) -> str:
        try:
            context = self._get_relevant_context(query)
            prompt = self._create_prompt(query, context, conversation)
            response = self.model.generate_content(prompt)

            Message.objects.create(
                conversation=conversation,
                question=query,
                answer=response.text
            )
            
            return response.text

        except Exception as e:
            error_response = "I apologize, but I encountered an error processing your request. Please try again."
            Message.objects.create(
                conversation=conversation,
                question=query,
                answer=error_response
            )
            return error_response