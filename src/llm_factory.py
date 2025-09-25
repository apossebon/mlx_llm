from langchain_openai import ChatOpenAI
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