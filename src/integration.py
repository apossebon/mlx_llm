# --------------------------------
# Exemplo de uso
# --------------------------------
from datetime import datetime
import uuid
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import SummarizationMiddleware
# from chatmlx import ChatMLX
# from chatmlx_gpt import ChatMLX
# from mychat_model import MyChatModel
from mychatmodel import MyChatModel
from langchain.agents import create_agent
import asyncio
from message_utils import pretty_print_messages, print_conversation_summary
from langgraph.checkpoint.memory import InMemorySaver

@tool
async def getDataHora():
    """
    Obtém a data e hora atual no fuso horário de São Paulo.
    
    Returns:
        str: Data e hora atual no formato DD/MM/AAAA HH:MM:SS
            - data_hora (str): Data e hora atual no formato DD/MM/AAAA HH:MM:SS
            
    """

    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def sync_main():
    myllm = MyChatModel(model_name="mlx-community/gpt-oss-20b-MXFP4-Q8", max_tokens=4098, use_gpt_harmony_response_format=True)
    myllm.init()
    


    

async def main():
    """
    Exemplo de uso assíncrono.
    """
    myllm = MyChatModel(max_tokens=4098, use_gpt_harmony_response_format=True)
    myllm.init()

    # myllm_summarization = MyChatModel(max_tokens=1024, use_gpt_harmony_response_format=True)
    # myllm_summarization.init()
    
    client = MultiServerMCPClient(
        {
            "ddg-search": {
                "transport": "streamable_http",
                "url": "http://192.168.1.105:8001/mcp/"
            },
            "yfinance-tools": {
                "transport": "streamable_http",
                "url": "http://192.168.1.105:8002/mcp/"
            },
            # "postgres-tools": {
            #     "transport": "streamable_http",
            #     "url": "http://192.168.1.105:8003/mcp/"
            # },
        }
    )
    mcp_tools = await client.get_tools()
    # mcp_tools.append(getDataHora)

    # summarization_middleware = SummarizationMiddleware(
    #         model=myllm,
    #         max_tokens_before_summary=4096,  # Trigger summarization at 4000 tokens
    #         messages_to_keep=10,  # Keep last 10 messages after summary
    #         summary_prompt="Custom prompt for summarization messagens",  # Optional
    #     )

    agent = create_agent(
        model=myllm,
        tools= mcp_tools,
        prompt=(
        "You are a helpful assistant that can answer questions and use tools. "
        "When choosing a tool, ALWAYS include every required argument from the tool schema. "
        "If the user has not provided a required argument (e.g., 'ticker' for stock tools), "
        "ask a brief follow-up question to obtain it before calling the tool."
        ),
        checkpointer= InMemorySaver(),
        # middleware=[summarization_middleware]
    )

    id_session = uuid.uuid4()
    config={"configurable": {"thread_id": id_session}}

    while True:
        prompt = input("\n\n\nDigite sua pergunta: ")
        if prompt == "exit":
            break

        input_text = {"messages": [HumanMessage(content=prompt)]}


        # result = await agent.ainvoke(input_text, config=config)
        # # Acessar todas as mensagens
        # all_messages = result["messages"]
        
        # # Pegar apenas a última mensagem (resposta do assistente)
        # last_message = all_messages[-1]
        
        # # Você pode acessar diferentes propriedades da última mensagem:
        # print("\n📜 ÚLTIMA MENSAGEM:")
        # print(f"Tipo: {last_message.type}")
        # print(f"Conteúdo: {last_message.content}")
        try:
            async for step, metadata in agent.astream(input_text, config=config, stream_mode="messages"):
                if metadata["langgraph_node"] == "agent" and (text := step.text()):
                    print(text, end="")
                    
                elif metadata["langgraph_node"] == "tools" and (text := step.text()):
                    # print("Chamada de Tools:")
                    # print(text, end="")
                    print("\n")
        except Exception as e:
            print(f"🔍 Error: {e}")
        
        # print("\n📜 RESULTADO DO AGENT:")
        # pretty_print_messages(result)
        # print_conversation_summary(result)

    


if __name__ == "__main__":
    asyncio.run(main())
  