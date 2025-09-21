from mlx_lm import load, generate, stream_generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache, save_prompt_cache, load_prompt_cache
from datetime import datetime
import re
import json
import asyncio

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

from mlx_lm.server import run 

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


functions_definition_example = """
# Instruções

Utilize as funções disponíveis para fornecer respostas precisas e detalhadas.

# Ferramentas

namespace functions {
    type getDataHora = () => any;
} // namespace functions
"""

def getDataHora():
    """
    Obtém a data e hora atual no fuso horário de São Paulo.
    
    Returns:
        dict: Um dicionário contendo:
            - data_hora (str): Data e hora atual no formato DD/MM/AAAA HH:MM:SS
            
    """

    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    # return {
    #     "data_hora": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    #     "timezone": datetime.now().strftime("%z")
    # }

# Dicionário de funções disponíveis
available_functions = {
    "getDataHora": getDataHora
}

def detect_tool_call(text):
    tool_open = "<tool_call>"
    tool_close = "</tool_call>"
    start_tool = text.find(tool_open) + len(tool_open)
    end_tool = text.find(tool_close)

    return start_tool != -1 and end_tool != -1

def extract_tools_call(text)->tuple[dict, dict]:

    tool_open = "<tool_call>"
    tool_close = "</tool_call>"
    start_tool = text.find(tool_open) + len(tool_open)
    end_tool = text.find(tool_close)

    if start_tool == -1 or end_tool == -1:
        return None, None
    
    tool_call = json.loads(text[start_tool:end_tool].strip())
    tool_result = available_functions[tool_call["name"]](**tool_call["arguments"])
    return tool_result, tool_call 

def detect_function_call(text):
    """
    Detecta se o texto contém uma chamada de função no formato Harmony
    """
    # Padrão para detectar chamadas de função no formato Harmony
    function_call_pattern = r'<\|call\|>commentary<\|message\|>.*?<\|end\|>'
    
    if re.search(function_call_pattern, text):
        return True
    
    # Também verifica se há menção a funções específicas
    # for func_name in available_functions.keys():
    #     if f"functions.{func_name}" in text or func_name in text:
    #         return True
    
    return False

def extract_function_call(text):
    """
    Extrai informações da chamada de função do texto
    """
    # Procurar por padrões de chamada de função
    for func_name in available_functions.keys():
        # Verificar diferentes padrões de chamada
        patterns = [
            f"functions.{func_name}",
            f"<|call|>commentary<|message|>.*?{func_name}",
            f"to=functions.{func_name}",
            func_name
        ]
        
        for pattern in patterns:
            if pattern in text:
                return {
                    "function_name": func_name,
                    "arguments": {}  # Por enquanto, sem argumentos
                }
    
    return None

def execute_function(function_name, arguments):
    """
    Executa a função especificada com os argumentos fornecidos
    """
    if function_name in available_functions:
        try:
            result = available_functions[function_name](**arguments)
            return result
        except Exception as e:
            return {"error": f"Erro ao executar {function_name}: {str(e)}"}
    else:
        return {"error": f"Função {function_name} não encontrada"}

def create_harmony_tool_response(function_name, result):
    """
    Cria uma resposta de ferramenta no formato Harmony correto
    """
    result_json = json.dumps(result, ensure_ascii=False)
    
    # Formato correto para resposta de ferramenta no Harmony
    tool_response = f"<|channel|>commentary<|message|>{result_json}<|end|>"
    
    return tool_response

def create_harmony_conversation_with_tool_result(original_prompt, function_name, result, functions_definition):
    """
    Cria uma nova conversa Harmony incluindo o resultado da ferramenta
    """
    result_json = json.dumps(result, ensure_ascii=False)
    
    # Criar mensagens no formato Harmony
    system_message = Message.from_role_and_content(
        Role.SYSTEM,
        SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
    )
    
    developer_message = Message.from_role_and_content(
        Role.DEVELOPER,
        DeveloperContent.new().with_instructions(functions_definition)
    )
    
    user_message = Message.from_role_and_content(
        Role.USER, 
        original_prompt
    )
    

    # Criar mensagem de ferramenta corretamente
    tool_message = Message.from_author_and_content(
        Author.new(Role.TOOL, function_name),  # Especifica qual ferramenta
        TextContent(text=result_json)
    ).with_channel("commentary")
    
    # Construir conversa Harmony com resultado da ferramenta
    conversation = Conversation.from_messages([
        system_message, 
        developer_message, 
        user_message,
        tool_message
    ])
    
    return conversation


def render_harmony_conversation(usermessage: str, functions_definition: str):
    # Criar mensagens no formato Harmony
        system_message = Message.from_role_and_content(
            Role.SYSTEM,
            SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
        )
        
        developer_message = Message.from_role_and_content(
            Role.DEVELOPER,
            DeveloperContent.new().with_instructions(functions_definition)
        )
        
        user_message = Message.from_role_and_content(
            Role.USER, 
            usermessage
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
        return tokens



async def main():
    print("Hello from mlx-llm!")
    DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
    Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"
    Gemma_MODEL_ID = "mlx-community/gemma-3-27b-it-qat-4bit"
   
    model, tokenizer = load(DEFAULT_MODEL_ID)

    _mlx_sampler = make_sampler(temp=0.7, top_p=0.85, top_k=40)
    _mlx_logits_processors = make_logits_processors(repetition_penalty=1.15, repetition_context_size=50)
    _mlx_prompt_cache = make_prompt_cache(model)



    prompt = "Qual é a data e hora atual? Explique também como o tempo funciona."
    b_harmony = True

    while True:
        prompt = input("Digite sua pergunta: ")
        if prompt == "exit":
            break
        
        # Renderizar conversa Harmony inicial
        if b_harmony:
            prompt_harmony = render_harmony_conversation(prompt, functions_definition_example)
        else:
            messages = [{"role": "user", "content": prompt}]
   
            prompt_harmony = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tools=list(available_functions.values())
            )
        
        # Gerar resposta com interceptação de funções
        full_response = ""
        function_called = False
        function_result = None
        
        print("🤖 Resposta do LLM:")
        tokens_respose =[]
        for response in stream_generate(model, tokenizer, prompt_harmony, max_tokens=1024, sampler=_mlx_sampler, logits_processors=_mlx_logits_processors, prompt_cache=_mlx_prompt_cache):
            print(response.text, end="", flush=True)
            full_response += response.text
            tokens_respose.append(response.token)
            # print(response.token)


            if b_harmony:
            # Verificar se há chamada de função durante o streaming
                if detect_function_call(full_response) and not function_called:
                    function_called = True
                    print("\n\n🔧 Detectada chamada de função durante o streaming!")
                    
                    # Extrair informações da chamada
                    function_call = extract_function_call(full_response)
                    if function_call:
                        print(f"Executando função: {function_call['function_name']}")
                        
                        # Executar função
                        function_result = execute_function(function_call['function_name'], function_call['arguments'])
                        print(f"Resultado: {function_result}")
                        
                        # Parar o streaming atual
                        break
                    else:
                        print("⚠️ Não foi possível extrair informações da chamada de função")
                        print(f"Texto detectado: {full_response[-200:]}")  # Últimos 200 caracteres
            else:

                

                
                if detect_tool_call(full_response):
                    function_called = True
                    resptools, tool_call = extract_tools_call(full_response)
                    if resptools:
                       
                        function_result = resptools

                    
                    

        # Se uma função foi chamada, continuar com o resultado
        if function_called and function_result:
            print("\n\n🔄 Continuando com resultado da função...")
            
            if b_harmony:
                # Criar conversa Harmony com resultado da ferramenta
                conversation_with_result = create_harmony_conversation_with_tool_result(
                    prompt, function_call['function_name'], function_result, functions_definition_example
                )
                
                # Renderizar para tokens
                result_tokens = encoding.render_conversation_for_completion(conversation_with_result, Role.ASSISTANT)
                print(f"Tokens Harmony com resultado: {len(result_tokens)}")
                
                # Gerar resposta final
                print("🤖 Resposta final:")
                for response in stream_generate(model, tokenizer, result_tokens, max_tokens=1024, sampler=_mlx_sampler, logits_processors=_mlx_logits_processors, prompt_cache=_mlx_prompt_cache):
                    print(response.text, end="", flush=True)
            else:
                messages = [{"role": "tool", "name": tool_call["name"], "content": resptools}]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                )
                for tool_response in stream_generate(model, tokenizer, prompt=prompt, max_tokens=1024, sampler=_mlx_sampler, logits_processors=_mlx_logits_processors, prompt_cache=_mlx_prompt_cache):
                    print(tool_response.text, end="", flush=True)
        # print(full_response)
        print("\n\n" + "="*50 + "\n")




if __name__ == "__main__":
    asyncio.run(main())
