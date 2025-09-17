# --------------------------------
# Exemplo de uso
# --------------------------------
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from chatmlx import ChatMLX
from langchain.agents import create_agent
import asyncio
from message_utils import pretty_print_messages, print_conversation_summary

@tool
async def getDataHora():
    """
    Obtém a data e hora atual no fuso horário de São Paulo.
    
    Returns:
        dict: Um dicionário contendo:
            - data_hora (str): Data e hora atual no formato DD/MM/AAAA HH:MM:SS
            
    """

    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

async def main():
    """
    Exemplo de uso assíncrono.
    """
    myllm = ChatMLX()
    myllm.init()

    agent = create_agent(
        model=myllm,
        tools=[getDataHora],
        prompt="You are a helpful assistant that can answer questions and use tools.",
    )

    while True:
        prompt = input("Digite sua pergunta: ")
        if prompt == "exit":
            break
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        print("\n📜 RESULTADO DO AGENT:")
        pretty_print_messages(result)
        print_conversation_summary(result)

    


if __name__ == "__main__":
    asyncio.run(main())

    # from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage 
    # from langchain_core.tools import tool
    # from langchain.agents import create_agent
    
    # @tool
    # def get_weather(location: str) -> str:
    #     """Get the weather at a location."""
    #     return f"It's sunny and 72°F in {location}."

    # @tool
    # def search_web(query: str) -> str:
    #     """Search the web for information."""
    #     return f"Search results for: {query}"

    # # Instanciar o modelo customizado
    # custom_llm = ChatMLX(
    #     model_name=Qwen_MODEL_ID,
    #     api_key="your-api-key",
    #     temperature=0.5,
    #     max_tokens=1024,
    #     top_p=0.85,
    #     top_k=40,
    #     repetition_penalty=1.15,
    #     repetition_context_size=50,
    # )
    # # Carrega o modelo/tokenizer uma vez (opcional — _ensure_loaded() faz lazy-load)
    # custom_llm.init()
    # print("✅ ChatMLX inicializado com sucesso!")

    # # create_agent pode chamar bind_tools internamente — agora ele NÃO cria nova instância
    # agent = create_agent(
    #     model=custom_llm,
    #     tools=[get_weather, search_web],
    #     prompt="You are a helpful assistant that can answer questions and use tools.",
    # )

    # result = agent.invoke({"messages": [HumanMessage("Hello, what is the weather in São Paulo?")]})
    
    # # Usar funções de impressão baseadas na documentação LangChain
    # from message_utils import pretty_print_messages, print_conversation_summary
    
    # print("\n📜 RESULTADO DO AGENT:")
    # pretty_print_messages(result)
    # print_conversation_summary(result)