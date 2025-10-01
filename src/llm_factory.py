from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from pydantic import SecretStr
import os

class LLMFactory:
    def __init__(self):
        self.llm = os.getenv("LLM")

    def get_llm(self):
        return self.llm
    
    def get_LMStudio_llm(self, model:str):

        llm = ChatOpenAI(base_url=os.getenv("base_url_LMStudio", "http://localhost:8000"), model=model, api_key=SecretStr("lm-studio"), streaming=True, max_retries=5)

        return llm
    
    def get_llm_vertexai(self, temperature: float = 0.5) -> ChatVertexAI:
        """
        Retorna o modelo chat-bison do Vertex AI.
        """
        gemini_pro_llm = ChatVertexAI(
            model_name="gemini-2.0-flash-001",
            temperature=temperature,
            # max_output_tokens=500,
            project="656370118038",  # Substitua pelo seu ID do projeto
            location="us-central1",
            # credentials=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),  # Caminho para o arquivo de credenciais
            

        )

        return gemini_pro_llm