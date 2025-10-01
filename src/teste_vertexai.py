# Testar a conexão
from llm_factory import LLMFactory

factory = LLMFactory()
llm = factory.get_llm_vertexai(temperature=0.7)

# Teste simples
response = llm.invoke("Olá, teste de conexão!")
print(response.content)