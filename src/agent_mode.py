#!/usr/bin/env python3
"""
Modo Agent com execução real de ferramentas
Implementa um loop completo de conversação com execução automática de tools
"""

import sys
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from chatmlx import ChatMLX


class MLXAgent:
    """
    Agent que usa ChatMLX com execução automática de ferramentas
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Inicializa o agent com ChatMLX
        
        Args:
            model_name: Nome do modelo MLX a usar (opcional)
        """
        # Usar modelo padrão se não especificado
        if model_name is None:
            model_name = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"
        
        self.llm = ChatMLX(model_name=model_name)
        self.llm.init()
        self.tools = {}
        self.conversation_history = []
        
    def add_tool(self, tool_func):
        """
        Adiciona uma ferramenta ao agent
        
        Args:
            tool_func: Função decorada com @tool do LangChain
        """
        self.tools[tool_func.name] = tool_func
        
    def bind_tools(self, tools: List):
        """
        Vincula múltiplas ferramentas ao agent
        
        Args:
            tools: Lista de funções decoradas com @tool
        """
        for tool_func in tools:
            self.add_tool(tool_func)
        
        # Vincular ferramentas ao modelo LLM
        self.llm_with_tools = self.llm.bind_tools(tools)
        self.llm_with_tools.init()
        
    def execute_tool(self, tool_call: Dict) -> str:
        """
        Executa uma ferramenta específica
        
        Args:
            tool_call: Dicionário com informações da tool call
            
        Returns:
            Resultado da execução da ferramenta
        """
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        
        if tool_name not in self.tools:
            return f"❌ Ferramenta '{tool_name}' não encontrada!"
        
        try:
            # Executar a ferramenta
            tool_func = self.tools[tool_name]
            result = tool_func.invoke(tool_args)
            
            print(f"🔧 Executando {tool_name}({tool_args}) → {result}")
            return str(result)
            
        except Exception as e:
            error_msg = f"❌ Erro ao executar {tool_name}: {str(e)}"
            print(error_msg)
            return error_msg
    
    def chat(self, user_message: str, max_iterations: int = 5) -> str:
        """
        Processa uma mensagem do usuário com execução automática de ferramentas
        
        Args:
            user_message: Mensagem do usuário
            max_iterations: Máximo de iterações para evitar loops infinitos
            
        Returns:
            Resposta final do agent
        """
        print(f"\n👤 Usuário: {user_message}")
        
        # Adicionar mensagem do usuário ao histórico
        messages = self.conversation_history + [HumanMessage(content=user_message)]
        
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Iteração {iteration}")
            
            # Gerar resposta do modelo
            try:
                response = self.llm_with_tools.invoke(messages)
                print(f"🤖 Modelo: {response.content}")
                
                # Verificar se há tool calls
                if response.tool_calls:
                    print(f"🔧 Tool calls detectados: {len(response.tool_calls)}")
                    
                    # Adicionar resposta do modelo ao histórico
                    messages.append(response)
                    
                    # Executar cada tool call
                    for i, tool_call in enumerate(response.tool_calls):
                        print(f"\n🔧 Executando Tool Call {i+1}:")
                        print(f"   Nome: {tool_call.get('name')}")
                        print(f"   Args: {tool_call.get('args')}")
                        
                        # Executar ferramenta
                        tool_result = self.execute_tool(tool_call)
                        
                        # Criar ToolMessage com o resultado
                        tool_message = ToolMessage(
                            content=tool_result,
                            tool_call_id=tool_call.get("id"),
                            name=tool_call.get("name")
                        )
                        
                        # Adicionar resultado ao histórico
                        messages.append(tool_message)
                    
                    # Continuar o loop para gerar resposta final
                    continue
                
                else:
                    # Sem tool calls - resposta final
                    print(f"✅ Resposta final: {response.content}")
                    
                    # Atualizar histórico da conversa
                    self.conversation_history = messages + [response]
                    
                    return response.content
                    
            except Exception as e:
                error_msg = f"❌ Erro na iteração {iteration}: {str(e)}"
                print(error_msg)
                return error_msg
        
        # Se chegou ao limite de iterações
        final_msg = f"⚠️ Limite de {max_iterations} iterações atingido"
        print(final_msg)
        return final_msg
    
    def reset_conversation(self):
        """
        Limpa o histórico da conversa
        """
        self.conversation_history = []
        print("🔄 Histórico da conversa limpo")
    
    def show_conversation_history(self):
        """
        Mostra o histórico da conversa
        """
        print("\n📜 Histórico da Conversa:")
        print("=" * 50)
        
        for i, msg in enumerate(self.conversation_history):
            if isinstance(msg, HumanMessage):
                print(f"{i+1}. 👤 Usuário: {msg.content}")
            elif isinstance(msg, AIMessage):
                print(f"{i+1}. 🤖 Assistant: {msg.content}")
                if msg.tool_calls:
                    print(f"   🔧 Tool calls: {len(msg.tool_calls)}")
            elif isinstance(msg, ToolMessage):
                print(f"{i+1}. 🔧 Tool '{msg.name}': {msg.content}")
        
        print("=" * 50)


# Ferramentas de exemplo
@tool
def get_weather(location: str) -> str:
    """Obtém informações do clima para uma localização."""
    # Simulação de API de clima
    weather_data = {
        "São Paulo": "Ensolarado, 25°C",
        "Rio de Janeiro": "Parcialmente nublado, 28°C", 
        "San Francisco": "Nublado, 18°C",
        "New York": "Chuva leve, 15°C",
        "London": "Nublado, 12°C"
    }
    
    location_clean = location.strip().title()
    
    # Buscar por correspondência parcial
    for city, weather in weather_data.items():
        if location_clean.lower() in city.lower() or city.lower() in location_clean.lower():
            return f"🌤️ Clima em {city}: {weather}"
    
    # Se não encontrou, retornar clima genérico
    return f"🌤️ Clima em {location}: Ensolarado, 22°C (dados simulados)"


@tool  
def search_web(query: str) -> str:
    """Busca informações na web sobre um tópico."""
    # Simulação de busca web
    search_results = {
        "python": "Python é uma linguagem de programação de alto nível, interpretada e de propósito geral.",
        "machine learning": "Machine Learning é um subcampo da IA que permite aos computadores aprender sem serem explicitamente programados.",
        "langchain": "LangChain é um framework para desenvolvimento de aplicações com modelos de linguagem.",
        "mlx": "MLX é um framework de machine learning para Apple Silicon desenvolvido pela Apple."
    }
    
    query_lower = query.lower()
    
    # Buscar por correspondência parcial
    for topic, info in search_results.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            return f"🔍 Informações sobre '{topic}': {info}"
    
    # Resultado genérico se não encontrou
    return f"🔍 Resultados para '{query}': Informações encontradas sobre o tópico solicitado (simulado)"


@tool
def calculate(expression: str) -> str:
    """Calcula expressões matemáticas simples."""
    try:
        # Avaliar expressão matemática de forma segura
        # Remover espaços e caracteres não permitidos
        safe_expr = expression.replace(" ", "")
        
        # Lista de caracteres permitidos
        allowed_chars = "0123456789+-*/.()%"
        if not all(c in allowed_chars for c in safe_expr):
            return "❌ Expressão contém caracteres não permitidos"
        
        # Calcular
        result = eval(safe_expr)
        return f"🧮 {expression} = {result}"
        
    except Exception as e:
        return f"❌ Erro no cálculo: {str(e)}"


def main():
    """
    Função principal para demonstrar o agent
    """
    print("🤖 MLX Agent - Modo Interativo")
    print("=" * 50)
    
    # Criar agent
    agent = MLXAgent()
    
    # Adicionar ferramentas
    agent.bind_tools([get_weather, search_web, calculate])
    
    print("✅ Agent inicializado com ferramentas:")
    print("   🌤️  get_weather - Informações do clima")
    print("   🔍 search_web - Busca na web") 
    print("   🧮 calculate - Calculadora")
    print("\n💡 Digite 'quit' para sair, 'history' para ver histórico, 'reset' para limpar")
    
    while True:
        try:
            user_input = input("\n👤 Você: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'sair']:
                print("👋 Até logo!")
                break
            elif user_input.lower() in ['history', 'histórico']:
                agent.show_conversation_history()
                continue
            elif user_input.lower() == 'reset':
                agent.reset_conversation()
                continue
            elif not user_input:
                continue
            
            # Processar mensagem
            response = agent.chat(user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Até logo!")
            break
        except Exception as e:
            print(f"❌ Erro: {e}")


if __name__ == "__main__":
    main()
