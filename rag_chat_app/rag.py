import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import pyinputplus as pyip
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self, pdf_path):
        self.path = self._validate_pdf_path(pdf_path)
        self.vector_store = self._initialize_vector_store()
    
    def _validate_pdf_path(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        return pdf_path

    def load_documents(self):
        documents = []
        for file_name in os.listdir(self.path):
            if file_name.endswith('.pdf'):
                file_path = os.path.join(self.path, file_name)
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
        return documents

    def split_documents(self):
        documents = self.load_documents()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        return splitter.split_documents(documents)
    
    def _initialize_vector_store(self):
        texts = self.split_documents()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        return FAISS.from_documents(texts, embeddings)

MODEL = "gemini-2.0-flash-exp"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class Chatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.chat_chain = self._create_chat_chain()
    
    def _create_chat_chain(self):
        llm = ChatGoogleGenerativeAI(
            model=MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=0.7,
            system_prompt="You are a 1337 Coding School Staff member. You should provide accurate and helpful information about the school based on the provided context. If you don't know the answer, you have to say 'I don't know, im a 1337 Ai Assistant and i can only answer questions related to the school'.",
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(),
            return_source_documents=False,
        )
    
    def get_response(self, query, chat_history=[]):
        try:
            response = self.chat_chain.invoke({
                "question": query, 
                "chat_history": chat_history
            })
            return response['answer']
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
    
    def chat(self):
        print("Hello! I'm a virtual assistant for the 1337 school. Ask me anything about the school.")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        
        chat_history = []
        while True:
            try:
                query = pyip.inputStr(prompt="You: ", blank=False)
                
                if query.lower() in ['exit', 'quit', 'bye']:
                    print("Assistant: Goodbye!")
                    break
                
                response = self.get_response(query, chat_history)
                print(f"Assistant: {response}")
                
                chat_history.append((query, response))
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"An error occurred: {str(e)}")

DATA_DIR = "data/"

def main():
    try:
        doc_processor = DocumentProcessor(DATA_DIR)
        chatbot = Chatbot(doc_processor.vector_store)
        chatbot.chat()
        
    except Exception as e:
        print(f"Application failed to start: {str(e)}")

if __name__ == "__main__":
    main()