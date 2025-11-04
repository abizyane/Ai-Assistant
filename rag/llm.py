from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from rag.config import LLM_MODEL, API_KEY, DEFAULT_TEMPERATURE

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
