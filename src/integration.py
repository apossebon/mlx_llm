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
from langgraph.checkpoint.memory import InMemorySaver

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
    myllm = ChatMLX(max_tokens=4096)
    myllm.init()

    agent = create_agent(
        model=myllm,
        tools=[getDataHora],
        prompt="You are a helpful assistant that can answer questions and use tools.",
        checkpointer= InMemorySaver()
    )

    config={"configurable": {"thread_id": "1"}}

    while True:
        prompt = input("\n\n\nDigite sua pergunta: ")
        if prompt == "exit":
            break

        input_text = {"messages": [HumanMessage(content=prompt)]}

        async for step, metadata in agent.astream(input_text, config=config, stream_mode="messages"):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                print(text, end="")
                
            elif metadata["langgraph_node"] == "tools" and (text := step.text()):
                print("Chamada de Tools:")
                print(text, end="")
                print("\n")
        
        
        # print("\n📜 RESULTADO DO AGENT:")
        # pretty_print_messages(result)
        # print_conversation_summary(result)

    


if __name__ == "__main__":
    asyncio.run(main())
  