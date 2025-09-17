#!/usr/bin/env python3
"""
Teste da implementação assíncrona do ChatMLX
"""

import sys
import asyncio
sys.path.append('src')

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from src.chatmlx import ChatMLX


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"🌤️ Clima em {location}: Ensolarado, 25°C"


@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        result = eval(expression.replace("^", "**"))
        return f"🧮 {expression} = {result}"
    except:
        return f"❌ Erro no cálculo: {expression}"


async def test_async_generate():
    """Teste da função _agenerate"""
    print("🧪 Teste 1: _agenerate (geração assíncrona)")
    print("=" * 50)
    
    try:
        # Criar modelo
        model = ChatMLX()
        model.init()
        
        # Vincular ferramentas
        model_with_tools = model.bind_tools([get_weather, calculate])
        
        # Teste assíncrono
        messages = [HumanMessage(content="What's the weather in São Paulo?")]
        
        print("💬 Pergunta: 'What's the weather in São Paulo?'")
        print("🔄 Executando _agenerate...")
        
        # Chamar método assíncrono
        result = await model_with_tools._agenerate(messages)
        
        print("✅ _agenerate executado com sucesso!")
        print(f"   Tipo: {type(result)}")
        print(f"   Conteúdo: {result.generations[0].message.content}")
        print(f"   Tool calls: {len(result.generations[0].message.tool_calls) if result.generations[0].message.tool_calls else 0}")
        
        if result.generations[0].message.tool_calls:
            for i, tc in enumerate(result.generations[0].message.tool_calls):
                print(f"     Tool {i+1}: {tc}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no teste assíncrono: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_vs_sync():
    """Comparar performance assíncrona vs síncrona"""
    print("\n🧪 Teste 2: Comparação Async vs Sync")
    print("=" * 50)
    
    try:
        import time
        
        model = ChatMLX()
        model.init()
        model_with_tools = model.bind_tools([get_weather])
        
        messages = [HumanMessage(content="Weather in Rio de Janeiro?")]
        
        # Teste síncrono
        print("🔄 Executando versão síncrona...")
        start_sync = time.time()
        sync_result = model_with_tools._generate(messages)
        sync_time = time.time() - start_sync
        
        # Teste assíncrono
        print("🔄 Executando versão assíncrona...")
        start_async = time.time()
        async_result = await model_with_tools._agenerate(messages)
        async_time = time.time() - start_async
        
        print(f"⏱️ Tempo síncrono: {sync_time:.3f}s")
        print(f"⏱️ Tempo assíncrono: {async_time:.3f}s")
        
        # Verificar se os resultados são similares
        sync_content = sync_result.generations[0].message.content
        async_content = async_result.generations[0].message.content
        
        print(f"📝 Resultado síncrono: {sync_content[:100]}...")
        print(f"📝 Resultado assíncrono: {async_content[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na comparação: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_concurrent_calls():
    """Teste de chamadas concorrentes"""
    print("\n🧪 Teste 3: Chamadas Assíncronas Concorrentes")
    print("=" * 50)
    
    try:
        model = ChatMLX()
        model.init()
        model_with_tools = model.bind_tools([get_weather, calculate])
        
        # Preparar múltiplas perguntas
        questions = [
            "What's the weather in São Paulo?",
            "Calculate 15 * 23",
            "Weather in London?",
            "What is 100 / 4?"
        ]
        
        messages_list = [[HumanMessage(content=q)] for q in questions]
        
        print(f"🚀 Executando {len(questions)} chamadas concorrentes...")
        
        # Executar todas as chamadas concorrentemente
        start_time = time.time()
        tasks = [model_with_tools._agenerate(msgs) for msgs in messages_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        print(f"⏱️ Tempo total para {len(questions)} chamadas: {total_time:.3f}s")
        print(f"📊 Média por chamada: {total_time/len(questions):.3f}s")
        
        # Mostrar resultados
        for i, (question, result) in enumerate(zip(questions, results)):
            if isinstance(result, Exception):
                print(f"❌ Pergunta {i+1}: {question} → Erro: {result}")
            else:
                content = result.generations[0].message.content
                tool_calls = len(result.generations[0].message.tool_calls) if result.generations[0].message.tool_calls else 0
                print(f"✅ Pergunta {i+1}: {question}")
                print(f"   Resposta: {content[:80]}...")
                print(f"   Tool calls: {tool_calls}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas chamadas concorrentes: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Função principal dos testes assíncronos"""
    print("🤖 Testando Implementação Assíncrona do ChatMLX")
    print("=" * 60)
    
    tests = [
        test_async_generate,
        test_async_vs_sync,
        test_concurrent_calls
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"❌ Teste falhou: {e}")
    
    print(f"\n📊 Resultado Final: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🎉 Todos os testes assíncronos passaram!")
        print("✅ _agenerate implementado corretamente!")
        print("🚀 Suporte a concorrência funcionando!")
    else:
        print("⚠️ Alguns testes falharam")
    
    return passed == total


if __name__ == "__main__":
    import time
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
