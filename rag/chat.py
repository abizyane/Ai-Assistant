from typing import List, Tuple, Dict, Any
import pyinputplus as pyip
import uuid
from rag.llm import ChatLLM
from rag.vector_store import PGVectorStore
from rag.document_processing import PDFDocumentLoader, TextSplitter
from langchain.chains import ConversationalRetrievalChain
from rag.config import DATA_DIR
from rag.database import save_chat_history, load_chat_history

class ChatChain:
    def __init__(self, chat_llm: ChatLLM, vector_store: PGVectorStore):
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
    
    def __init__(self, data_dir: str = DATA_DIR, session_id: str = None):
        if RAGAssistant._initialized and getattr(self, 'session_id', None) == session_id:
            return
            
        try:
            self.session_id = session_id or str(uuid.uuid4())
            document_loader = PDFDocumentLoader(data_dir)
            documents = document_loader.load()
            
            text_splitter = TextSplitter()
            chunks = text_splitter.split(documents)
            
            self.vector_store = PGVectorStore()
            if chunks:
                self.vector_store.add_documents(chunks)
            
            chat_llm = ChatLLM()
            self.chat_chain = ChatChain(chat_llm, self.vector_store)
            
            self.chat_history = load_chat_history(self.session_id)
            RAGAssistant._initialized = True
        except Exception as e:
            raise
    
    def get_response(self, query: str) -> str:
        try:
            response = self.chat_chain.invoke(query, self.chat_history)
            answer = response.get('answer', 'No answer found')
            save_chat_history(self.session_id, query, answer)
            self.chat_history.append((query, answer))
            return answer
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def run_chat_session(self):
        print(f"Hello! I'm a virtual assistant for the 1337 school. Your session ID is {self.session_id}.")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        
        while True:
            try:
                query = pyip.inputStr(prompt="You: ", blank=False)
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Assistant: Goodbye!")
                    break
                
                response = self.get_response(query)
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")
