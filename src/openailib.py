from openai import OpenAI
import sys
import json
from datetime import datetime
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
)

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

# Carregar codificação Harmony para GPT-OSS
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

def getDataHora():
    """Retorna a data e hora atual formatada"""
    now = datetime.now()
    return {
        "data_hora": now.strftime("%d/%m/%Y %H:%M:%S"),
        "timestamp": now.timestamp(),
        "timezone": "America/Sao_Paulo"
    }

def chat_with_harmony_function_streaming():
    """Combinação: Harmony + Function Calling + Streaming"""
    try:
        print("=== Harmony + Function Calling + Streaming ===")
        
        # Definir as funções no formato Harmony
        functions_definition = """
# Instruções

Utilize as funções disponíveis para fornecer respostas precisas e detalhadas.

# Ferramentas

## functions

namespace functions {

// Retorna a data e hora atual formatada
type getDataHora = () => any;

} // namespace functions
        """
        
        # Criar mensagens no formato Harmony
        system_message = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort("Low")
        )
        
        developer_message = Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(functions_definition)
        )
        
        user_message = Message.from_role_and_content(
            Role.USER, 
            "Qual é a data e hora atual? Explique também como o tempo funciona."
        )
        
        # Construir conversa Harmony
        conversation = Conversation.from_messages([
            system_message, 
            developer_message, 
            user_message
        ])
        
        # Renderizar para tokens
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        print(f"Tokens Harmony gerados: {len(tokens)}")
        
        # Converter para formato OpenAI
        messages = [
            {
                "role": "system", 
                "content": "Você é um assistente que pode chamar funções. Use getDataHora() quando necessário e explique conceitos de forma detalhada."
            },
            {
                "role": "user", 
                "content": "Qual é a data e hora atual? Explique também como o tempo funciona."
            }
        ]
        
        # Primeira chamada com streaming
        print("\n�� Resposta inicial (streaming):")
        response = client.chat.completions.create(
            model="mlx-community/gpt-oss-20b-MXFP4-Q8",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        full_response = ""
        function_needed = False
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
                
                # Verificar se menciona função
                if "getDataHora" in content or "função" in content.lower():
                    function_needed = True
        
        print("\n")
        
        # Se função foi mencionada, executar e continuar
        if function_needed or "getDataHora" in full_response:
            print("\n🔧 Executando função getDataHora()...")
            result = getDataHora()
            print(f"Resultado: {result}")
            
            # Adicionar resultado ao contexto
            messages.append({
                "role": "assistant", 
                "content": full_response
            })
            messages.append({
                "role": "user", 
                "content": f"Aqui está a data e hora atual: {json.dumps(result, ensure_ascii=False)}. Continue sua explicação sobre como o tempo funciona."
            })
            
            # Segunda chamada com streaming
            print("\n📝 Continuação da resposta (streaming):")
            second_response = client.chat.completions.create(
                model="mlx-community/gpt-oss-20b-MXFP4-Q8",
                messages=messages,
                stream=True,
                temperature=0.7,
                max_tokens=1024
            )
            
            for chunk in second_response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
            
            print("\n")
        
    except Exception as e:
        print(f"Erro: {e}")
        print("Certifique-se de que o servidor MLX-LM está rodando!")

def chat_with_standard_function_streaming():
    """Function Calling + Streaming padrão (sem Harmony)"""
    try:
        print("=== Function Calling + Streaming Padrão ===")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "getDataHora",
                    "description": "Retorna a data e hora atual do sistema",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
        
        messages = [
            {"role": "user", "content": "Qual é a data e hora atual? Explique também como o tempo funciona."}
        ]
        
        # Primeira chamada com streaming
        print("�� Resposta inicial (streaming):")
        response = client.chat.completions.create(
            model="mlx-community/gpt-oss-20b-MXFP4-Q8",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        full_response = ""
        tool_calls = []
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
                full_response += content
            
            # Capturar tool calls
            if chunk.choices[0].delta.tool_calls:
                for tool_call in chunk.choices[0].delta.tool_calls:
                    if tool_call.id not in [tc.get('id') for tc in tool_calls]:
                        tool_calls.append({
                            'id': tool_call.id,
                            'type': tool_call.type,
                            'function': {'name': tool_call.function.name, 'arguments': ''}
                        })
                    else:
                        # Adicionar argumentos
                        for tc in tool_calls:
                            if tc['id'] == tool_call.id:
                                tc['function']['arguments'] += tool_call.function.arguments
        
        print("\n")
        
        # Processar tool calls
        if tool_calls:
            print("🔧 Processando tool calls...")
            for tool_call in tool_calls:
                if tool_call['function']['name'] == 'getDataHora':
                    result = getDataHora()
                    print(f"Resultado: {result}")
                    
                    messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "tool_calls": tool_calls
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call['id'],
                        "content": json.dumps(result)
                    })
                    
                    # Segunda chamada com streaming
                    print("\n📝 Continuação da resposta (streaming):")
                    second_response = client.chat.completions.create(
                        model="mlx-community/gpt-oss-20b-MXFP4-Q8",
                        messages=messages,
                        stream=True,
                        temperature=0.7,
                        max_tokens=1024
                    )
                    
                    for chunk in second_response:
                        if chunk.choices[0].delta.content is not None:
                            content = chunk.choices[0].delta.content
                            print(content, end="", flush=True)
                    
                    print("\n")
        
    except Exception as e:
        print(f"Erro: {e}")

def chat_with_harmony_streaming_only():
    """Harmony + Streaming (sem function calling)"""
    try:
        print("=== Harmony + Streaming ===")
        
        # Criar mensagens no formato Harmony
        system_message = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort("Low")
        )
        
        developer_message = Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions("Responda sempre em português brasileiro de forma detalhada")
        )
        
        user_message = Message.from_role_and_content(
            Role.USER, 
            "Conte uma história detalhada sobre Einstein e sua teoria da relatividade"
        )
        
        # Construir conversa Harmony
        conversation = Conversation.from_messages([
            system_message, 
            developer_message, 
            user_message
        ])
        
        # Renderizar para tokens
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        print(f"Tokens Harmony gerados: {len(tokens)}")
        
        # Converter para formato OpenAI
        messages = [
            {
                "role": "system", 
                "content": "Você é um assistente que responde em português brasileiro de forma detalhada."
            },
            {
                "role": "user", 
                "content": "Conte uma história detalhada sobre Einstein e sua teoria da relatividade"
            }
        ]
        
        # Chamada com streaming
        print("\n📝 Resposta (streaming):")
        response = client.chat.completions.create(
            model="mlx-community/gpt-oss-20b-MXFP4-Q8",
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                print(content, end="", flush=True)
        
        print("\n")
        
    except Exception as e:
        print(f"Erro: {e}")

# ... (manter as outras funções existentes) ...

if __name__ == "__main__":
    print("Escolha uma opção:")
    print("1. Function Calling Padrão")
    print("2. Harmony Function Calling")
    print("3. Harmony Simplificado")
    print("4. Harmony Streaming")
    print("5. Harmony + Function + Streaming")
    print("6. Function + Streaming Padrão")
    print("7. Harmony + Streaming (sem function)")
    
    choice = input("Digite sua escolha (1-7): ").strip()
    
    if choice == "1":
        chat_with_standard_function_calling()
    elif choice == "2":
        chat_with_harmony_function_calling()
    elif choice == "3":
        chat_with_harmony_simplified()
    elif choice == "4":
        chat_with_harmony_streaming()
    elif choice == "5":
        chat_with_harmony_function_streaming()
    elif choice == "6":
        chat_with_standard_function_streaming()
    elif choice == "7":
        chat_with_harmony_streaming_only()
    else:
        print("Opção inválida. Executando function calling padrão...")
        chat_with_standard_function_calling()

# Comando para iniciar o servidor MLX-LM:
# mlx_lm.server \
#   --model "mlx-community/gpt-oss-20b-MXFP4-Q8" \
#   --host "127.0.0.1" \
#   --port 8080 \
#   --temp 0.7 \
#   --top-p 0.9 \
#   --max-tokens 2048 \
#   --log-level INFO \
#   --use-default-chat-template