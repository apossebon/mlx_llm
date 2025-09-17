#!/usr/bin/env python3
"""
Exemplo de uso de memória/checkpointer com ChatMLX
Demonstra diferentes tipos de persistência
"""

import sys
sys.path.append('src')

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from src.chatmlx import ChatMLX


@tool
def get_weather(location: str) -> str:
    """Obtém informações do clima."""
    return f"🌤️ Clima em {location}: Ensolarado, 25°C"


@tool
def save_preference(preference: str, value: str) -> str:
    """Salva uma preferência do usuário."""
    return f"✅ Preferência '{preference}' salva como '{value}'"


def demo_memory_types():
    """Demonstra diferentes tipos de memória"""
    print("🧠 DEMO: Tipos de Memória no LangChain Agent")
    print("=" * 60)
    
    # Criar modelo
    model = ChatMLX()
    model.init()
    
    # 1. Agent SEM memória (não lembra conversas anteriores)
    print("\n1️⃣ Agent SEM Memória:")
    print("-" * 30)
    
    agent_no_memory = create_agent(
        model=model,
        tools=[get_weather, save_preference],
        prompt="Você é um assistente útil. Responda em português."
        # Sem checkpointer = sem memória
    )
    
    # Primeira pergunta
    result1 = agent_no_memory.invoke({
        "messages": [HumanMessage("Meu nome é João e moro em São Paulo")]
    })
    print(f"Resposta 1: {result1['messages'][-1].content}")
    
    # Segunda pergunta - não vai lembrar do nome
    result2 = agent_no_memory.invoke({
        "messages": [HumanMessage("Qual é o meu nome?")]
    })
    print(f"Resposta 2: {result2['messages'][-1].content}")
    
    
    # 2. Agent COM memória InMemorySaver
    print("\n2️⃣ Agent COM Memória (InMemorySaver):")
    print("-" * 40)
    
    checkpointer = InMemorySaver()
    agent_with_memory = create_agent(
        model=model,
        tools=[get_weather, save_preference],
        prompt="Você é um assistente útil. Responda em português.",
        checkpointer=checkpointer  # ✅ Memória ativada
    )
    
    # Configuração com thread_id para persistência
    config = {"configurable": {"thread_id": "user_joao"}}
    
    # Primeira pergunta
    result1 = agent_with_memory.invoke({
        "messages": [HumanMessage("Meu nome é João e moro em São Paulo")]
    }, config=config)
    print(f"Resposta 1: {result1['messages'][-1].content}")
    
    # Segunda pergunta - VAI lembrar do nome!
    result2 = agent_with_memory.invoke({
        "messages": [HumanMessage("Qual é o meu nome?")]
    }, config=config)  # Mesmo thread_id
    print(f"Resposta 2: {result2['messages'][-1].content}")
    
    # Terceira pergunta - contexto completo
    result3 = agent_with_memory.invoke({
        "messages": [HumanMessage("Qual é o clima na minha cidade?")]
    }, config=config)
    print(f"Resposta 3: {result3['messages'][-1].content}")


def demo_multiple_conversations():
    """Demonstra múltiplas conversas com threads separados"""
    print("\n3️⃣ Múltiplas Conversas (Threads Separados):")
    print("-" * 50)
    
    model = ChatMLX()
    model.init()
    
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[get_weather],
        prompt="Você é um assistente útil. Responda em português.",
        checkpointer=checkpointer
    )
    
    # Conversa com João
    config_joao = {"configurable": {"thread_id": "joao_123"}}
    agent.invoke({
        "messages": [HumanMessage("Meu nome é João")]
    }, config=config_joao)
    
    # Conversa com Maria (thread diferente)
    config_maria = {"configurable": {"thread_id": "maria_456"}}
    agent.invoke({
        "messages": [HumanMessage("Meu nome é Maria")]
    }, config=config_maria)
    
    # Testar memória separada
    result_joao = agent.invoke({
        "messages": [HumanMessage("Qual é o meu nome?")]
    }, config=config_joao)
    print(f"João pergunta seu nome: {result_joao['messages'][-1].content}")
    
    result_maria = agent.invoke({
        "messages": [HumanMessage("Qual é o meu nome?")]
    }, config=config_maria)
    print(f"Maria pergunta seu nome: {result_maria['messages'][-1].content}")


def demo_conversation_history():
    """Mostra o histórico completo da conversa"""
    print("\n4️⃣ Histórico Completo da Conversa:")
    print("-" * 40)
    
    model = ChatMLX()
    model.init()
    
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=[get_weather, save_preference],
        prompt="Você é um assistente útil. Responda em português.",
        checkpointer=checkpointer
    )
    
    config = {"configurable": {"thread_id": "demo_conversation"}}
    
    # Sequência de mensagens
    messages = [
        "Olá! Meu nome é Pedro",
        "Salve minha cor favorita como azul",
        "Qual é o clima em Rio de Janeiro?",
        "Qual é o meu nome e minha cor favorita?"
    ]
    
    for i, msg in enumerate(messages, 1):
        print(f"\n💬 Mensagem {i}: {msg}")
        result = agent.invoke({
            "messages": [HumanMessage(msg)]
        }, config=config)
        print(f"🤖 Resposta: {result['messages'][-1].content}")
    
    # Mostrar histórico completo
    print(f"\n📜 Histórico completo ({len(result['messages'])} mensagens):")
    for i, msg in enumerate(result['messages']):
        msg_type = type(msg).__name__
        content = getattr(msg, 'content', '')[:80]
        print(f"  {i+1}. {msg_type}: {content}...")


def main():
    """Função principal"""
    print("🧠 ChatMLX - Demonstração de Memória e Persistência")
    print("📚 Baseado na documentação do LangChain")
    print("=" * 70)
    
    try:
        demo_memory_types()
        demo_multiple_conversations()
        demo_conversation_history()
        
        print("\n🎉 DEMONSTRAÇÃO CONCLUÍDA!")
        print("✅ InMemorySaver: Memória em RAM (volátil)")
        print("💾 SqliteSaver: Arquivo SQLite (persistente)")
        print("🏢 PostgresSaver: PostgreSQL (produção)")
        print("🔄 Thread ID: Separa conversas diferentes")
        print("📝 Estado: Mantém contexto da conversa")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
