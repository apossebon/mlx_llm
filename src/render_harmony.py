import json

import re
from typing import List, Dict
import uuid

from langchain_core.messages import BaseMessage

# OpenAI Harmony (para formato de mensagens e ferramentas compatível)
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
    Author,
    TextContent,
    ReasoningEffort
)


class RenderHarmony:
    def __init__(self):
        
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    

    def detect_harmony_tool_calls(self, response: str) -> List[Dict]:
        """
        Detecta tool calls no formato OpenAI Harmony e retorna no mesmo formato
        que _detect_real_tool_calls para manter compatibilidade.
        
        Formato Harmony esperado:
        <|start|>assistant<|channel|>commentary to=functions.function_name <|constrain|>json<|message|>{"param": "value"}<|call|>
        """
        tool_calls = []
        
        # Padrão regex para capturar tool calls no formato Harmony
        # Captura: recipient (função), conteúdo JSON e tipo de constraint
        harmony_pattern = r'<\|start\|>assistant<\|channel\|>commentary\s+to=functions\.([^<\s]+)(?:\s+<\|constrain\|>(\w+))?<\|message\|>(.*?)<\|call\|>'
        
        for match in re.finditer(harmony_pattern, response, re.DOTALL):
            function_name = match.group(1)  # Nome da função
            constraint_type = match.group(2) or "json"  # Tipo de constraint (padrão: json)
            content = match.group(3).strip()  # Conteúdo da mensagem
            
            try:
                # Se o constraint é JSON, tenta fazer parse
                if constraint_type.lower() == "json":
                    args_data = json.loads(content)
                else:
                    # Para outros tipos, mantém como string
                    args_data = {"content": content}
                
                tool_calls.append({
                    "name": function_name,
                    "args": args_data,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                })
                
            except json.JSONDecodeError:
                # Se falhar no parse JSON, tenta extrair argumentos de forma mais flexível
                try:
                    # Tenta encontrar estruturas JSON-like no conteúdo
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        args_data = json.loads(json_match.group())
                        tool_calls.append({
                            "name": function_name,
                            "args": args_data,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        })
                except json.JSONDecodeError:
                    # Como último recurso, cria argumentos vazios
                    tool_calls.append({
                        "name": function_name,
                        "args": {},
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    })
                    continue
        
        return tool_calls

    def detect_harmony_analysis_messages(self, response: str) -> List[str]:
        """
        Extrai mensagens do canal 'analysis' que contêm o raciocínio do modelo.
        Essas mensagens não devem ser mostradas ao usuário final.
        """
        analysis_messages = []
        
        # Padrão para capturar mensagens do canal analysis
        analysis_pattern = r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>'
        
        for match in re.finditer(analysis_pattern, response, re.DOTALL):
            analysis_content = match.group(1).strip()
            analysis_messages.append(analysis_content)
        
        return analysis_messages

    def detect_harmony_final_messages(self, response: str) -> List[str]:
        """
        Extrai mensagens do canal 'final' que devem ser mostradas ao usuário.
        """
        final_messages = []
        
        # Padrão para capturar mensagens do canal final
        # Captura tudo após <|channel|>final<|message|> até encontrar <|end|>, <|start|>, ou fim da string
        final_pattern = r'<\|channel\|>final<\|message\|>(.*?)(?=<\|(?:end|start)\||$)'
        
        for match in re.finditer(final_pattern, response, re.DOTALL):
            final_content = match.group(1).strip()
            final_messages.append(final_content)
        
        return final_messages

    def detect_harmony_commentary_messages(self, response: str) -> List[str]:
        """
        Extrai mensagens do canal 'commentary' que podem incluir preambles
        para o usuário sobre as ações que serão executadas.
        """
        commentary_messages = []
        
        # Padrão para capturar mensagens do canal commentary (não tool calls)
        # Exclui mensagens que têm 'to=functions.' (que são tool calls)
        # Captura tudo após <|channel|>commentary<|message|> até encontrar <|end|>, <|start|>, ou fim da string
        commentary_pattern = r'<\|channel\|>commentary(?!\s+to=functions\.)<\|message\|>(.*?)(?=<\|(?:end|start)\||$)'
        
        for match in re.finditer(commentary_pattern, response, re.DOTALL):
            commentary_content = match.group(1).strip()
            commentary_messages.append(commentary_content)
        
        return commentary_messages

    def render_harmony_conversation(self, messages: List[BaseMessage], bound_tools: List[Dict], ReasoningEffort: ReasoningEffort = ReasoningEffort.LOW):
        """
        Renderiza uma conversa no formato Harmony a partir de uma mensagem ou lista de mensagens.
        
        Args:
            messages (dict | List[dict]): Mensagem única ou lista de mensagens com estrutura:
                {
                    "role": "user" | "assistant" | "system" | "tool",
                    "content": str,
                    "tool_calls": List[dict] (opcional),
                    "tool_call_id": str (opcional, para mensagens de tool),
                    "name": str (opcional, nome da tool para mensagens de tool)
                }
        
        Returns:
            List[int]: Tokens renderizados para o modelo
        """
        harmony_messages = []
        functions_definition = self.convert_tools_to_harmony_format(bound_tools)
        system_instruction = "You are a helpful assistant that can answer questions and use tools."
        user_message_content = ""

        system_message = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(ReasoningEffort)
        )
        harmony_messages.append(system_message)


        for message in messages:
            content = message.content
            if message.type == "system":
                
                developer_message = Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(content).with_function_tools(functions_definition)
                )
                
                harmony_messages.append(developer_message)
            elif message.type == "human":
                user_message = Message.from_role_and_content(Role.USER, content)
                harmony_messages.append(user_message)
            elif message.type == "ai":
                assistant_message = Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
                harmony_messages.append(assistant_message)
            elif message.type == "tool":
                tool_message = Message.from_author_and_content(
                    Author.new(Role.TOOL, message.name),
                    content
                ).with_channel("commentary")
                harmony_messages.append(tool_message)


        
        # # Normalizar entrada: se for um dict único, transformar em lista
        # if isinstance(messages, dict):
        #     message_list = [messages]
        # elif isinstance(messages, list):
        #     message_list = messages
        # else:
        #     raise ValueError("messages deve ser um dict ou uma lista de dicts")
        
        # # Criar mensagem de sistema padrão (se não houver uma na lista)
        # has_system = any(msg.get("type") == "system" for msg in message_list)
        # if not has_system:
        #     system_message = Message.from_role_and_content(
        #         Role.SYSTEM,
        #         SystemContent.new().with_reasoning_effort(ReasoningEffort)
        #     )
        #     harmony_messages.append(system_message)
        
        # # Criar mensagem de desenvolvedor com tools (se não houver uma na lista)
        # has_developer = any(msg.get("type") == "developer" for msg in message_list)
        # if not has_developer and bound_tools:
        #     developer_message = Message.from_role_and_content(
        #         Role.DEVELOPER,
        #         DeveloperContent.new().with_instructions("You are a helpful assistant.").with_function_tools(
        #             self.convert_tools_to_harmony_format(bound_tools)
        #         )
        #     )
        #     harmony_messages.append(developer_message)
        
        # # Processar cada mensagem da lista
        # for message in message_list:
        #     role = message.get("role", "user")
        #     content = message.get("content", "")
            
        #     if role == "system":
        #         # Usar mensagem de sistema personalizada se fornecida
        #         system_content = SystemContent.new().with_reasoning_effort(ReasoningEffort)
        #         if content and content != "You are a helpful assistant that can answer questions and use tools.":
        #             # Se há conteúdo personalizado, criar mensagem de sistema customizada
        #             system_message = Message.from_role_and_content(Role.SYSTEM, content)
        #         else:
        #             system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
        #         harmony_messages.append(system_message)
                
        #     elif role == "user":
        #         user_message = Message.from_role_and_content(Role.USER, content)
        #         harmony_messages.append(user_message)
                
        #     elif role == "assistant":
        #         # Mensagem do assistente
        #         if content:
        #             assistant_message = Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
        #             harmony_messages.append(assistant_message)
                
        #         # Se há tool calls, adicionar no formato Harmony
        #         tool_calls = message.get("tool_calls", [])
        #         for tool_call in tool_calls:
        #             # Mensagem de análise (chain of thought)
        #             analysis_message = Message.from_role_and_content(
        #                 Role.ASSISTANT, 
        #                 f"Need to use function {tool_call['name']}."
        #             ).with_channel("analysis")
        #             harmony_messages.append(analysis_message)
                    
        #             # Mensagem de chamada da função
        #             function_call_message = Message.from_role_and_content(
        #                 Role.ASSISTANT,
        #                 json.dumps(tool_call.get("args", {}))
        #             ).with_channel("commentary").with_recipient(f"functions.{tool_call['name']}").with_content_type("<|constrain|> json")
        #             harmony_messages.append(function_call_message)
                    
        #     elif role == "tool":
        #         # Mensagem de resposta de tool
        #         tool_name = message.get("name", "unknown_tool")
        #         tool_message = Message.from_author_and_content(
        #             Author.new(Role.TOOL, f"functions.{tool_name}"),
        #             content
        #         ).with_channel("commentary")
        #         harmony_messages.append(tool_message)
                
        #     elif role == "developer":
        #         # Mensagem de desenvolvedor personalizada
        #         if bound_tools:
        #             developer_message = Message.from_role_and_content(
        #                 Role.DEVELOPER,
        #                 DeveloperContent.new().with_instructions(content).with_function_tools(
        #                     self.convert_tools_to_harmony_format(bound_tools)
        #                 )
        #             )
        #         else:
        #             developer_message = Message.from_role_and_content(
        #                 Role.DEVELOPER,
        #                 DeveloperContent.new().with_instructions(content)
        #             )
        #         harmony_messages.append(developer_message)
        
        # Construir conversa Harmony
        conversation = Conversation.from_messages(harmony_messages)
        
        # Renderizar para tokens
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        # print(f"Tokens Harmony gerados: {len(tokens)} para {len(message_list)} mensagem(ns)")
        return tokens
    
    def convert_tools_to_harmony_format(self, bound_tools: List[Dict]):
        """
        Converte as tools bound para o formato esperado pelo Harmony.
        
        Estrutura esperada do bound_tools:
        {
            'type': 'function', 
            'function': {
                'name': 'getDataHora', 
                'description': '...', 
                'parameters': {...}
            }
        }
        """
        harmony_tools = []
        
        # Verificar se bound_tools é uma lista ou dicionário
        if isinstance(bound_tools, dict):
            # Se é um dicionário único com uma função
            if 'function' in bound_tools:
                func_info = bound_tools['function']
                tool_name = func_info.get('name', 'unknown_function')
                description = func_info.get('description', f"Function {tool_name}")
                parameters = func_info.get('parameters', {})
                
                # Criar ToolDescription no formato Harmony
                from openai_harmony import ToolDescription
                
                tool_desc = ToolDescription.new(
                    tool_name,
                    description,
                    parameters=parameters
                )
                harmony_tools.append(tool_desc)
            else:
                # Se é um dicionário com múltiplas funções (formato antigo)
                for tool_name, tool_info in bound_tools.items():
                    description = tool_info.get("description", f"Function {tool_name}")
                    parameters = tool_info.get("parameters", {})
                    
                    from openai_harmony import ToolDescription
                    
                    tool_desc = ToolDescription.new(
                        tool_name,
                        description,
                        parameters=parameters
                    )
                    harmony_tools.append(tool_desc)
        
        elif isinstance(bound_tools, list):
            # Se é uma lista de ferramentas
            for tool_item in bound_tools:
                if isinstance(tool_item, dict) and 'function' in tool_item:
                    func_info = tool_item['function']
                    tool_name = func_info.get('name', 'unknown_function')
                    description = func_info.get('description', f"Function {tool_name}")
                    parameters = func_info.get('parameters', {})
                    
                    from openai_harmony import ToolDescription
                    
                    tool_desc = ToolDescription.new(
                        tool_name,
                        description,
                        parameters=parameters
                    )
                    harmony_tools.append(tool_desc)
        
        return harmony_tools