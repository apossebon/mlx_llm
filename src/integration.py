# --------------------------------
# Exemplo de uso
# --------------------------------
from datetime import datetime
import uuid
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import trim_messages
from typing import Dict, Any
from chatmlx import ChatMLX
# from chatmlx_gpt import ChatMLX
# from mychat_model import MyChatModel
from mychatmodel import MyChatModel
from langchain.agents import create_agent
import asyncio
from message_utils import pretty_print_messages, print_conversation_summary
from langgraph.checkpoint.memory import InMemorySaver
from llm_factory import LLMFactory
import server

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
    myllm = MyChatModel(max_tokens=4098, use_gpt_harmony_response_format=True)
    myllm.init()

    #teste sincrono 
    
    result = myllm.invoke("Qual é a capital do Brasil?")
    print(result)
    

def pre_model_hook(state: Dict[str, Any]) -> Dict[str, Any] | None:
    # state["messages"] contém o histórico atual
    trimmed = trim_messages(
        state.get("messages", []),
        strategy="last",          # preserva as mais recentes
        token_counter=len,        # conta por "mensagem" (não por tokens)
        max_tokens= 10,             # mantém apenas 5 mensagens
        include_system=True,      # mantém system se houver
        start_on="human",
        end_on=("human", "tool"),
    )
    return {"messages": trimmed}
    

async def main():
    """
    Exemplo de uso assíncrono.
    """
    #myllm = MyChatModel(max_tokens=4098, use_gpt_harmony_response_format=True, use_prompt_cache=False)
    myllm = ChatMLX(max_tokens=4098, use_gpt_harmony_response_format=False)
    myllm.init()

    llm_factory = LLMFactory()

    myllm_summarization = llm_factory.get_LMStudio_llm("lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit")
   

    summarization_middleware = SummarizationMiddleware(
                model=myllm_summarization,
                max_tokens_before_summary=1024,  # Trigger summarization at 4000 tokens
                messages_to_keep=10,  # Keep last 10 messages after summary
                # summary_prompt="Custom prompt for summarization messagens",  # Optional
            )
   
    
    client = MultiServerMCPClient(
        {
            "ddg-search": {
                "transport": "streamable_http",
                "url": "http://192.168.1.103:8001/mcp/"
            },
            "yfinance-tools": {
                "transport": "streamable_http",
                "url": "http://192.168.1.103:8002/mcp/"
            },
            # "postgres-tools": {
            #     "transport": "streamable_http",
            #     "url": "http://192.168.1.103:8003/mcp/"
            # },
        }
    )
    mcp_tools = await client.get_tools()
    # mcp_tools.append(getDataHora)

    # summarization_middleware = SummarizationMiddleware(
    #         model=myllm_summarization,
    #         # max_tokens_before_summary=1024,  # Trigger summarization at 4000 tokens
    #         messages_to_keep=10,  # Keep last 10 messages after summary
    #         # summary_prompt="Custom prompt for summarization messagens",  # Optional
    #     )

    agent = create_agent(
        model=myllm,
        tools= mcp_tools,
        prompt=(
        "Voce é um dedicados assistente de investimentos que pode responder perguntas sobre investimentos e usar ferramentas para obter informações sobre investimentos."
        "## INSTRUÇÕES:\n"
        "Use a ferramenta yfinance-tools para obter informações sobre investimentos."
        "Use a ferramenta ddg-search para obter informações sobre investimentos."
        "Sempre encontre o ticker ou valide o ticker da empresa com a ferrament especifica."
        "Sempre que for realizar uma analise de sentimento, procure sempre com a busca utilizendo 'Ultimas noticias da empresa'"
        "## DADOS:\n"
        "Utilize quantas vezes necessario as ferramentas disponivies antes de formular a resposta final"
        
        ),
        checkpointer= InMemorySaver(),
        # pre_model_hook=pre_model_hook,
        middleware=[summarization_middleware]
    )

    id_session = uuid.uuid4()
    config={"configurable": {"thread_id": id_session}}

    while True:
        prompt = input("\n\n\nDigite sua pergunta: ")
        if prompt == "exit":
            break

        input_text = {"messages": [HumanMessage(content=prompt)]}
        async for event in agent.astream_events({"messages": [("user", prompt)]},
                                 version="v1", config=config):
            # eventos úteis:
            if event["event"] == "on_chat_model_stream":
                # delta token-a-token do modelo
                #print(f'{event["metadata"]["langgraph_node"]}')
                if event["metadata"]["langgraph_node"] != "SummarizationMiddleware.before_model":
                    print(event["data"]["chunk"].content or "", end="")
                
            elif event["event"] == "on_tool_start":
                print(f"\n[tool start] {event['name']}")
                print(f"\n[tool start] {event['data']}")
            elif event["event"] == "on_tool_end":
                print(f"\n[tool end]   {event['name']}")
        print()


        # result = await agent.ainvoke(input_text, config=config)
        # # Acessar todas as mensagens
        # all_messages = result["messages"]
        
        # # Pegar apenas a última mensagem (resposta do assistente)
        # last_message = all_messages[-1]
        
        # # Você pode acessar diferentes propriedades da última mensagem:
        # print("\n📜 ÚLTIMA MENSAGEM:")
        # print(f"Tipo: {last_message.type}")
        # print(f"Conteúdo: {last_message.content}")
        # try:
        #     async for step, metadata in agent.astream(input_text, config=config, stream_mode="messages"):
        #         if metadata["langgraph_node"] == "agent" and (text := step.text()):
        #             print(text, end="")
                    
        #         elif metadata["langgraph_node"] == "tools" and (text := step.text()):
        #             # print("Chamada de Tools:")
        #             # print(text, end="")
        #             print("\n")
        # except Exception as e:
        #     print(f"🔍 Error: {e}")
        
        # print("\n📜 RESULTADO DO AGENT:")
        # pretty_print_messages(result)
        # print_conversation_summary(result)

    


if __name__ == "__main__":
    # sync_main()
    asyncio.run(main())
  