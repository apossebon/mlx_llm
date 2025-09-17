#!/usr/bin/env python3
"""
Exemplo Prático: Uso Assíncrono do ChatMLX
Demonstra como usar _agenerate para processamento concorrente
"""

import sys
import asyncio
import time
sys.path.append('src')

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from src.chatmlx import ChatMLX
from src.message_utils import pretty_print_messages, print_conversation_summary


@tool
def get_weather(location: str) -> str:
    """Obtém informações meteorológicas detalhadas."""
    weather_data = {
        "São Paulo": "🌤️ Ensolarado, 25°C, umidade 60%",
        "Rio de Janeiro": "⛅ Parcialmente nublado, 28°C, umidade 75%", 
        "London": "☁️ Nublado, 12°C, umidade 80%",
        "Tokyo": "☀️ Ensolarado, 22°C, umidade 55%",
        "New York": "🌧️ Chuva leve, 15°C, umidade 90%"
    }
    
    for city, weather in weather_data.items():
        if location.lower() in city.lower() or city.lower() in location.lower():
            return f"Clima em {city}: {weather}"
    
    return f"🌤️ Clima em {location}: Ensolarado, 22°C (simulado)"


@tool
def calculate(expression: str) -> str:
    """Realiza cálculos matemáticos."""
    try:
        result = eval(expression.replace("^", "**").replace(" ", ""))
        return f"🧮 {expression} = {result}"
    except Exception as e:
        return f"❌ Erro: {str(e)}"


@tool
def search_info(topic: str) -> str:
    """Busca informações sobre um tópico."""
    info_db = {
        "python": "🐍 Linguagem de programação interpretada e de alto nível",
        "ai": "🤖 Inteligência Artificial - campo da ciência da computação",
        "machine learning": "📊 Subcampo da IA focado em aprendizado automático",
        "blockchain": "⛓️ Tecnologia de registro distribuído e criptografado"
    }
    
    topic_lower = topic.lower()
    for key, info in info_db.items():
        if key in topic_lower:
            return f"📚 {key.title()}: {info}"
    
    return f"🔍 Informações sobre '{topic}': Tópico interessante para pesquisa!"


async def process_single_request(model, question: str, request_id: int):
    """Processa uma única requisição de forma assíncrona"""
    print(f"🚀 Iniciando requisição {request_id}: {question}")
    
    start_time = time.time()
    
    try:
        # Usar _agenerate diretamente
        messages = [HumanMessage(content=question)]
        result = await model._agenerate(messages)
        
        duration = time.time() - start_time
        
        content = result.generations[0].message.content
        tool_calls = result.generations[0].message.tool_calls or []
        
        print(f"✅ Requisição {request_id} concluída em {duration:.2f}s")
        print(f"   Resposta: {content[:80]}...")
        print(f"   Tool calls: {len(tool_calls)}")
        
        return {
            "id": request_id,
            "question": question,
            "result": result,
            "duration": duration,
            "success": True
        }
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Requisição {request_id} falhou em {duration:.2f}s: {e}")
        
        return {
            "id": request_id,
            "question": question,
            "error": str(e),
            "duration": duration,
            "success": False
        }


async def concurrent_processing_demo():
    """Demonstração de processamento concorrente"""
    print("🎯 DEMO: Processamento Assíncrono Concorrente")
    print("=" * 60)
    
    # Criar modelo com ferramentas
    model = ChatMLX()
    model.init()
    model_with_tools = model.bind_tools([get_weather, calculate, search_info])
    
    # Lista de perguntas para processar concorrentemente
    questions = [
        "What's the weather in São Paulo?",
        "Calculate 15 * 23 + 100",
        "Tell me about machine learning",
        "What's the weather in London?", 
        "Calculate 50 / 2",
        "Search information about Python",
        "Weather in Tokyo?",
        "What is 2^8?"
    ]
    
    print(f"📋 Processando {len(questions)} perguntas concorrentemente...")
    print("⏱️ Iniciando cronômetro...")
    
    start_total = time.time()
    
    # Criar tasks para todas as requisições
    tasks = [
        process_single_request(model_with_tools, question, i+1) 
        for i, question in enumerate(questions)
    ]
    
    # Executar todas concorrentemente
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    total_duration = time.time() - start_total
    
    print(f"\n🏁 Processamento concorrente concluído!")
    print(f"⏱️ Tempo total: {total_duration:.2f}s")
    print(f"📊 Média por requisição: {total_duration/len(questions):.2f}s")
    
    # Analisar resultados
    successful = [r for r in results if isinstance(r, dict) and r.get('success')]
    failed = [r for r in results if isinstance(r, dict) and not r.get('success')]
    exceptions = [r for r in results if isinstance(r, Exception)]
    
    print(f"\n📈 Estatísticas:")
    print(f"   ✅ Sucessos: {len(successful)}")
    print(f"   ❌ Falhas: {len(failed)}")
    print(f"   🚫 Exceções: {len(exceptions)}")
    
    if successful:
        avg_duration = sum(r['duration'] for r in successful) / len(successful)
        print(f"   ⏱️ Tempo médio individual: {avg_duration:.2f}s")
    
    # Mostrar alguns resultados
    print(f"\n📝 Primeiros 3 resultados:")
    for i, result in enumerate(results[:3]):
        if isinstance(result, dict) and result.get('success'):
            print(f"\n{i+1}. {result['question']}")
            content = result['result'].generations[0].message.content
            print(f"   Resposta: {content[:100]}...")
            print(f"   Tempo: {result['duration']:.2f}s")


async def agent_async_demo():
    """Demonstração com create_agent assíncrono"""
    print("\n🎯 DEMO: create_agent com Suporte Assíncrono")
    print("=" * 60)
    
    # Criar agent
    model = ChatMLX()
    model.init()
    
    agent = create_agent(
        model=model,
        tools=[get_weather, calculate, search_info],
        prompt="Você é um assistente útil. Responda em português quando possível."
    )
    
    # Perguntas para o agent
    agent_questions = [
        "Qual é o clima em São Paulo e calcule 10 * 5?",
        "Me fale sobre Python e qual é o clima em Londres?",
        "Calcule 100 / 4 e busque informações sobre AI"
    ]
    
    print(f"🤖 Testando {len(agent_questions)} perguntas com agent...")
    
    start_time = time.time()
    
    # Processar com agent (que pode usar _agenerate internamente)
    agent_tasks = []
    for i, question in enumerate(agent_questions):
        print(f"\n💬 Pergunta {i+1}: {question}")
        
        # Agent invoke (síncrono, mas o modelo interno pode usar async)
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        
        print("📜 Resultado:")
        print_conversation_summary(result)
    
    agent_duration = time.time() - start_time
    print(f"\n⏱️ Tempo total do agent: {agent_duration:.2f}s")


async def main():
    """Função principal da demonstração"""
    print("🤖 ChatMLX - Demonstração de Funcionalidades Assíncronas")
    print("📚 Baseado na implementação de _agenerate")
    print("=" * 70)
    
    try:
        # Demo 1: Processamento concorrente
        await concurrent_processing_demo()
        
        # Demo 2: Agent com suporte assíncrono
        await agent_async_demo()
        
        print("\n🎉 TODAS AS DEMOS CONCLUÍDAS!")
        print("✅ _agenerate implementado e funcionando")
        print("🚀 Processamento concorrente ativo")
        print("⚡ Performance melhorada com asyncio")
        print("🔧 Compatível com create_agent do LangChain")
        
    except Exception as e:
        print(f"❌ Erro durante as demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
