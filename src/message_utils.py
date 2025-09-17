#!/usr/bin/env python3
"""
Utilitários para imprimir mensagens de agents LangChain
Baseado na documentação: https://docs.langchain.com/oss/python/langchain/agents
"""

from typing import Dict, List, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage


def pretty_print_messages(result: Dict[str, Any]) -> None:
    """
    Imprime todas as mensagens de um resultado de agent de forma organizada.
    
    Args:
        result: Resultado do agent.invoke() contendo chave 'messages'
    """
    if "messages" not in result:
        print("❌ Resultado não contém chave 'messages'")
        return
    
    messages = result["messages"]
    if not messages:
        print("📭 Nenhuma mensagem encontrada")
        return
    
    print("\n" + "="*70)
    print("📜 HISTÓRICO DA CONVERSA")
    print("="*70)
    
    for i, message in enumerate(messages, 1):
        print(f"\n{i}. {_format_message(message)}")
    
    print("\n" + "="*70)
    print(f"📊 Total: {len(messages)} mensagens")
    print("="*70)


def pretty_print_single_message(message: BaseMessage) -> None:
    """
    Imprime uma única mensagem usando o método .pretty_print() do LangChain.
    
    Args:
        message: Mensagem do LangChain
    """
    if hasattr(message, 'pretty_print'):
        message.pretty_print()
    else:
        print(_format_message(message))


def print_conversation_summary(result: Dict[str, Any]) -> None:
    """
    Imprime um resumo da conversa.
    
    Args:
        result: Resultado do agent.invoke() contendo chave 'messages'
    """
    if "messages" not in result:
        print("❌ Resultado não contém chave 'messages'")
        return
    
    messages = result["messages"]
    
    # Contar tipos de mensagens
    human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
    ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
    tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    system_count = sum(1 for m in messages if isinstance(m, SystemMessage))
    
    # Contar tool calls
    total_tool_calls = 0
    for m in messages:
        if isinstance(m, AIMessage) and hasattr(m, 'tool_calls') and m.tool_calls:
            total_tool_calls += len(m.tool_calls)
    
    print("\n📊 RESUMO DA CONVERSA")
    print("-" * 30)
    print(f"👤 Mensagens do usuário: {human_count}")
    print(f"🤖 Mensagens da IA: {ai_count}")
    print(f"🔧 Mensagens de ferramentas: {tool_count}")
    print(f"⚙️ Mensagens do sistema: {system_count}")
    print(f"🛠️ Total de tool calls: {total_tool_calls}")
    print(f"📝 Total de mensagens: {len(messages)}")


def print_last_message(result: Dict[str, Any]) -> None:
    """
    Imprime apenas a última mensagem (resposta final do agent).
    
    Args:
        result: Resultado do agent.invoke() contendo chave 'messages'
    """
    if "messages" not in result or not result["messages"]:
        print("❌ Nenhuma mensagem encontrada")
        return
    
    last_message = result["messages"][-1]
    print("\n🎯 RESPOSTA FINAL:")
    print("-" * 20)
    pretty_print_single_message(last_message)


def print_tool_calls_only(result: Dict[str, Any]) -> None:
    """
    Imprime apenas as mensagens que contêm tool calls.
    
    Args:
        result: Resultado do agent.invoke() contendo chave 'messages'
    """
    if "messages" not in result:
        print("❌ Resultado não contém chave 'messages'")
        return
    
    messages = result["messages"]
    tool_messages = []
    
    # Encontrar mensagens com tool calls
    for i, message in enumerate(messages):
        if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
            tool_messages.append((i+1, message))
        elif isinstance(message, ToolMessage):
            tool_messages.append((i+1, message))
    
    if not tool_messages:
        print("🔧 Nenhum tool call encontrado")
        return
    
    print("\n🔧 TOOL CALLS DETECTADOS:")
    print("-" * 30)
    
    for pos, message in tool_messages:
        print(f"\n{pos}. {_format_message(message)}")


def print_streaming_messages(chunk: Dict[str, Any]) -> None:
    """
    Imprime mensagens durante streaming conforme documentação LangChain.
    
    Args:
        chunk: Chunk do stream contendo mensagens
    """
    if "messages" not in chunk or not chunk["messages"]:
        return
    
    latest_message = chunk["messages"][-1]
    
    if hasattr(latest_message, 'content') and latest_message.content:
        print(f"🤖 Agent: {latest_message.content}")
    elif hasattr(latest_message, 'tool_calls') and latest_message.tool_calls:
        tool_names = [tc.get('name', 'unknown') for tc in latest_message.tool_calls]
        print(f"🔧 Calling tools: {tool_names}")


def _format_message(message: BaseMessage) -> str:
    """
    Formata uma mensagem para exibição.
    
    Args:
        message: Mensagem do LangChain
        
    Returns:
        String formatada da mensagem
    """
    # Determinar tipo e emoji
    if isinstance(message, HumanMessage):
        emoji = "👤"
        msg_type = "USER"
    elif isinstance(message, AIMessage):
        emoji = "🤖"
        msg_type = "ASSISTANT"
    elif isinstance(message, ToolMessage):
        emoji = "🔧"
        msg_type = f"TOOL ({getattr(message, 'name', 'unknown')})"
    elif isinstance(message, SystemMessage):
        emoji = "⚙️"
        msg_type = "SYSTEM"
    else:
        emoji = "📝"
        msg_type = "MESSAGE"
    
    # Conteúdo da mensagem
    content = getattr(message, 'content', '')
    if len(content) > 100:
        content = content[:97] + "..."
    
    # Informações adicionais
    extras = []
    
    # Tool calls (para AIMessage)
    if isinstance(message, AIMessage) and hasattr(message, 'tool_calls') and message.tool_calls:
        tool_names = [tc.get('name', 'unknown') for tc in message.tool_calls]
        extras.append(f"🛠️ Tools: {', '.join(tool_names)}")
    
    # Tool call ID (para ToolMessage)
    if isinstance(message, ToolMessage) and hasattr(message, 'tool_call_id'):
        extras.append(f"🔗 ID: {message.tool_call_id}")
    
    # Message ID
    if hasattr(message, 'id') and message.id:
        extras.append(f"🆔 {message.id[:8]}...")
    
    # Montar string final
    result = f"{emoji} {msg_type}: {content}"
    
    if extras:
        result += f"\n   {' | '.join(extras)}"
    
    return result


def print_message_types_breakdown(result: Dict[str, Any]) -> None:
    """
    Imprime breakdown detalhado dos tipos de mensagens.
    
    Args:
        result: Resultado do agent.invoke() contendo chave 'messages'
    """
    if "messages" not in result:
        print("❌ Resultado não contém chave 'messages'")
        return
    
    messages = result["messages"]
    
    print("\n📋 BREAKDOWN POR TIPO DE MENSAGEM:")
    print("-" * 40)
    
    for i, message in enumerate(messages, 1):
        msg_type = type(message).__name__
        content_preview = str(getattr(message, 'content', ''))[:50]
        if len(content_preview) == 50:
            content_preview += "..."
        
        print(f"{i:2d}. {msg_type:15s} | {content_preview}")


# Função principal para demonstração
def demo_message_printing():
    """
    Demonstra as diferentes formas de imprimir mensagens.
    """
    print("🎯 DEMO: Formas de imprimir mensagens de agents LangChain")
    print("=" * 60)
    
    # Exemplo de resultado simulado
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
    
    example_result = {
        "messages": [
            HumanMessage(content="What's the weather in São Paulo?", id="msg_1"),
            AIMessage(
                content="I'll check the weather for you.",
                tool_calls=[{"name": "get_weather", "args": {"location": "São Paulo"}, "id": "call_1"}],
                id="msg_2"
            ),
            ToolMessage(
                content="🌤️ Clima em São Paulo: Ensolarado, 25°C",
                tool_call_id="call_1",
                name="get_weather",
                id="msg_3"
            ),
            AIMessage(
                content="The weather in São Paulo is sunny with 25°C. Perfect day to go outside!",
                id="msg_4"
            )
        ]
    }
    
    # Demonstrar diferentes funções
    print("\n1. 📜 PRETTY PRINT COMPLETO:")
    pretty_print_messages(example_result)
    
    print("\n2. 📊 RESUMO DA CONVERSA:")
    print_conversation_summary(example_result)
    
    print("\n3. 🎯 ÚLTIMA MENSAGEM:")
    print_last_message(example_result)
    
    print("\n4. 🔧 APENAS TOOL CALLS:")
    print_tool_calls_only(example_result)
    
    print("\n5. 📋 BREAKDOWN POR TIPO:")
    print_message_types_breakdown(example_result)


if __name__ == "__main__":
    demo_message_printing()
