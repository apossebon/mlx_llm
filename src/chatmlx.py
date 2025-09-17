from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing import List, Optional, Any, Dict, Union, Sequence, Iterator, AsyncIterator
import os
import requests
import json
import uuid
import asyncio
import time
import re

# Pydantic (para atributos privados de runtime)
from pydantic import PrivateAttr

### MLX imports
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate, stream_generate

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"


class ChatMLX(BaseChatModel):
    """
    Exemplo de LLM customizado seguindo o padrão LangChain.

    Funcionalidades incluídas:
    ✅ Chat completions básicas
    ✅ Suporte a bind_tools()
    ✅ Tool calling com formato LangChain
    ✅ Compatibilidade com agentes
    ✅ Conversão automática de ferramentas
    ✅ Simulação de tool calls para demonstração
    ✅ Streaming síncrono (_stream)
    ✅ Streaming assíncrono (_astream)
    ✅ AIMessageChunk com chunks progressivos
    ✅ Callbacks para tokens em tempo real
    """

    # -----------------------------
    # Campos "declarativos" (pydantic)
    # -----------------------------
    model_name: Optional[str] = Qwen_MODEL_ID
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.15
    repetition_context_size: int = 50

    # Propriedades para suporte a ferramentas (parte do "modelo")
    bound_tools: List[Dict[str, Any]] = []
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # -----------------------------
    # Atributos privados (runtime)
    # NÃO entram em validação/serialização do Pydantic
    # -----------------------------
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _mlx_sampler: Any = PrivateAttr(default=None)
    _mlx_logits_processors: Any = PrivateAttr(default=None)
    _tool_functions: Dict[str, Any] = PrivateAttr(default_factory=dict)

    # --------------------------------
    # Inicialização / carregamento
    # --------------------------------
    def init(self) -> bool:
        """
        Initialize the model and tokenizer (runtime).
        """
        print(f"🔍 Debug - model_name: {self.model_name}")
        try:
            
            # opcional: inicializar sampler/logits processors, se desejar
            self._mlx_sampler = make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k)
            self._mlx_logits_processors = make_logits_processors(repetition_penalty=self.repetition_penalty, repetition_context_size=self.repetition_context_size)
            self._model, self._tokenizer = load(self.model_name)
            return True
        except Exception as e:
            print(f"🔍 Error - {e}")
            return False

    def _ensure_loaded(self):
        """
        Garante que _model e _tokenizer estejam carregados no runtime.
        """
        if self._model is None or self._tokenizer is None:
            ok = self.init()
            if not ok:
                raise RuntimeError("Falha ao carregar modelo/tokenizer no ChatMLX.init()")

    @property
    def _llm_type(self) -> str:
        """Return identifier of the LLM."""
        return "ChatMLX"

    # --------------------------------
    # Núcleo de geração
    # --------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate chat completion seguindo o padrão LangChain.
        """
        api_messages = self._convert_messages(messages)

        try:
            response = self._call_api(api_messages, stop, **kwargs)
            content = response.get("content", "Default response from custom LLM")
            tool_calls = response.get("tool_calls", [])
        except Exception as e:
            content = f"Response from {self.model_name}: Hello from ChatMLX! (Error: {str(e)})"
            tool_calls = []


        if tool_calls:
            message = AIMessage(content=content, tool_calls=tool_calls)
        else:
            message = AIMessage(content=content)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Async version of _generate - implementação assíncrona real.
        """
        # 1. Converter mensagens LangChain para formato da API
        api_messages = self._convert_messages(messages)

        try:
            # 2. Chamada assíncrona para a API
            response = await self._acall_api(api_messages, stop, **kwargs)
            content = response.get("content", "Default async response from custom LLM")
            tool_calls = response.get("tool_calls", [])
        except Exception as e:
            content = f"Async response from {self.model_name}: Hello from ChatMLX! (Error: {str(e)})"
            tool_calls = []

        # 3. Criar AIMessage com tool_calls se necessário
        if tool_calls:
            message = AIMessage(content=content, tool_calls=tool_calls)
        else:
            message = AIMessage(content=content)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream de resposta em chunks.
        """
        api_messages = self._convert_messages(messages)

        try:
            response_text, tool_calls = self._simulate_streaming_response(api_messages, stop, **kwargs)

            for chunk_text, is_final in self._create_text_chunks(response_text):
                if is_final and tool_calls:
                    chunk = AIMessageChunk(
                        content=chunk_text,
                        tool_calls=[self._convert_tool_call(tc) for tc in tool_calls],
                    )
                else:
                    chunk = AIMessageChunk(content=chunk_text)

                generation_chunk = ChatGenerationChunk(message=chunk)

                if run_manager:
                    run_manager.on_llm_new_token(chunk_text)

                yield generation_chunk
                time.sleep(0.05)

        except Exception as e:
            error_message = f"Streaming error from {self.model_name}: ChatMLX! (Error: {str(e)})"
            chunk = AIMessageChunk(content=error_message)
            yield ChatGenerationChunk(message=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Stream assíncrono.
        """
        api_messages = self._convert_messages(messages)

        try:
            response_text, tool_calls = self._simulate_streaming_response(api_messages, stop, **kwargs)

            for chunk_text, is_final in self._create_text_chunks(response_text):
                if is_final and tool_calls:
                    chunk = AIMessageChunk(
                        content=chunk_text,
                        tool_calls=[self._convert_tool_call(tc) for tc in tool_calls],
                    )
                else:
                    chunk = AIMessageChunk(content=chunk_text)

                generation_chunk = ChatGenerationChunk(message=chunk)

                if run_manager:
                    await run_manager.on_llm_new_token(chunk_text)

                yield generation_chunk
                await asyncio.sleep(0.05)

        except Exception as e:
            error_message = f"Async streaming error from {self.model_name}: ChatMLX! (Error: {str(e)})"
            chunk = AIMessageChunk(content=error_message)
            yield ChatGenerationChunk(message=chunk)

    # --------------------------------
    # Conversões / utilidades
    # --------------------------------
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        api_messages = []
        for message in messages:
            api_messages.append({"role": self._get_role(message.type), "content": message.content})
        return api_messages

    def _get_role(self, message_type: str) -> str:
        role_mapping = {"human": "user", "ai": "assistant", "system": "system"}
        return role_mapping.get(message_type, "user")

    def _call_api(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
        """
        Chamada real ao modelo MLX (generate).
        """
        # Garante que _model/_tokenizer existem
        self._ensure_loaded()

        # Monta o prompt via tokenizer
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=self.bound_tools,
            )
        except Exception as e:
            print(f"🔍 Error - {e}")
            return {"content": f"Error: {str(e)}", "tool_calls": []}

        response = generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            sampler=self._mlx_sampler,
            logits_processors=self._mlx_logits_processors,
        )

        # Detectar tool calls reais na resposta
        detected_tool_calls = self._detect_real_tool_calls(response)

        print(f"🔍 Debug _call_api:")
        print(f"   Resposta do modelo: {response[:100]}...")
        print(f"   Tool calls detectados: {len(detected_tool_calls)}")
        if detected_tool_calls:
            for i, tc in enumerate(detected_tool_calls):
                print(f"   Tool Call {i+1}: {type(tc)} - {tc}")

        if detected_tool_calls:
            return {"content": "", "tool_calls": detected_tool_calls}
        else:
            return {"content": response, "tool_calls": []}

    async def _acall_api(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
        """
        Versão assíncrona de _call_api - chamada assíncrona ao modelo MLX.
        """
        import asyncio
        
        # Garante que _model/_tokenizer existem
        self._ensure_loaded()

        # Monta o prompt via tokenizer
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=self.bound_tools,
            )
        except Exception as e:
            print(f"🔍 Async Error - {e}")
            return {"content": f"Async Error: {str(e)}", "tool_calls": []}

        # Executar generate de forma assíncrona usando asyncio.to_thread
        # para não bloquear o event loop
        try:
            response = await asyncio.to_thread(
                generate,
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=self.max_tokens,
                sampler=self._mlx_sampler,
                logits_processors=self._mlx_logits_processors,
            )
        except Exception as e:
            print(f"🔍 Async Generate Error - {e}")
            return {"content": f"Async Generate Error: {str(e)}", "tool_calls": []}

        # Detectar tool calls reais na resposta
        detected_tool_calls = self._detect_real_tool_calls(response)

        print(f"🔍 Debug _acall_api (async):")
        print(f"   Resposta do modelo: {response[:100]}...")
        print(f"   Tool calls detectados: {len(detected_tool_calls)}")
        if detected_tool_calls:
            for i, tc in enumerate(detected_tool_calls):
                print(f"   Tool Call {i+1}: {type(tc)} - {tc}")

        if detected_tool_calls:
            return {"content": "", "tool_calls": detected_tool_calls}
        else:
            return {"content": response, "tool_calls": []}

    # --------------------------------
    # Ferramentas
    # --------------------------------
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> "ChatMLX":
        """
        Vincula ferramentas ao modelo.
        Importante: NÃO criar nova instância aqui (para não perder _model/_tokenizer).
        """
        formatted_tools = []
        tool_functions = {}

        for tool in tools:
            if hasattr(tool, "name"):  # BaseTool
                formatted_tools.append(convert_to_openai_tool(tool))
                tool_functions[tool.name] = tool.func
            elif isinstance(tool, dict):  # já é dict no formato certo
                formatted_tools.append(tool)
                if "func" in tool and "name" in tool:
                    tool_functions[tool["name"]] = tool["func"]
            else:  # função/classe Pydantic
                formatted_tools.append(convert_to_openai_tool(tool))
                if hasattr(tool, "__name__"):
                    tool_functions[tool.__name__] = tool

        # Mutar a própria instância — não criar outra
        self.bound_tools = formatted_tools
        self.tool_choice = kwargs.get("tool_choice", self.tool_choice)
        self._tool_functions = tool_functions
        return self

    def _convert_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Caso precise adaptar o formato de tool_call para o AIMessageChunk final.
        Aqui apenas devolvemos o que já está em dict.
        """
        return tool_call

    def _simulate_tool_calls(self, messages: List[BaseMessage], content: str) -> List[Dict]:
        """
        Simula tool calls para demonstração.
        """
        if not self.bound_tools:
            return []

        # Evitar loop de tool calls
        recent_tool_calls = 0
        for msg in messages[-5:]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                recent_tool_calls += len(msg.tool_calls)
        if recent_tool_calls >= 2:
            return []

        last_message = messages[-1] if messages else None
        if not last_message or not hasattr(last_message, "content"):
            return []
        if hasattr(last_message, "type") and last_message.type == "tool":
            return []

        user_content = str(last_message.content).lower()

        tool_keywords = {
            "get_weather": ["weather", "temperature", "climate", "forecast"],
            "search_web": ["search", "find", "information", "look up"],
            "calculate": ["calculate", "math", "compute", "*", "+", "-", "/", "="],
        }

        for tool in self.bound_tools:
            tool_name = tool.get("function", {}).get("name", "")
            keywords = tool_keywords.get(tool_name, [])
            if any(keyword in user_content for keyword in keywords):
                args = self._generate_tool_args(tool_name, user_content)
                tool_call = {
                    "name": tool_name,
                    "args": args,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }
                return [tool_call]

        return []

    def _generate_tool_args(self, tool_name: str, user_content: str) -> Dict[str, Any]:
        """Gera argumentos de exemplo para cada ferramenta."""
        if tool_name == "get_weather":
            words = user_content.split()
            location = "San Francisco"
            location_indicators = ["in", "at", "for", "from"]
            for i, word in enumerate(words):
                if word.lower() in location_indicators and i + 1 < len(words):
                    next_word = words[i + 1].replace("?", "").replace(",", "").title()
                    if len(next_word) > 1:
                        location = next_word
                        break
            cities = ["paris", "london", "tokyo", "new york", "berlin", "madrid", "rome"]
            for city in cities:
                if city in user_content.lower():
                    location = city.title()
                    break
            return {"location": location}

        elif tool_name == "search_web":
            return {"query": user_content}

        elif tool_name == "calculate":
            math_expr = re.search(r"[\d\s+\-*/()]+", user_content)
            if math_expr:
                return {"expression": math_expr.group().strip()}
            return {"expression": "2 + 2"}

        return {"query": user_content}

    # --------------------------------
    # Streaming simulado
    # --------------------------------
    def _simulate_streaming_response(
        self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs
    ) -> tuple[str, List[Dict[str, Any]]]:
        last_message = messages[-1] if messages else {}
        user_content = last_message.get("content", "").lower()

        if "weather" in user_content:
            response_text = "I'll check the weather for you. Let me use the weather tool to get the current conditions."
        elif "search" in user_content or "find" in user_content:
            response_text = "I'll search for that information. Let me use the search tool to find relevant results."
        elif any(op in user_content for op in ["*", "+", "-", "/", "calculate"]):
            response_text = "I'll calculate that for you. Let me use the calculator tool to compute the result."
        else:
            response_text = (
                f"Hello! This is a streaming response from {self.model_name}. "
                f"I'm processing your request step by step."
            )

        tool_calls = []
        if self.bound_tools:
            tool_calls = self._simulate_tool_calls(
                [type("MockMessage", (), {"content": user_content, "type": "human"})()], response_text
            )

        return response_text, tool_calls

    def _create_text_chunks(self, text: str) -> Iterator[tuple[str, bool]]:
        if not text:
            yield ("", True)
            return
        words = text.split()
        for i, word in enumerate(words):
            is_final = i == len(words) - 1
            chunk_text = word if i == 0 else " " + word
            yield (chunk_text, is_final)
        if not words:
            yield (text, True)

    # --------------------------------
    # Detecção de tool calls reais (no texto)
    # --------------------------------
    def _detect_real_tool_calls(self, response: str) -> List[Dict]:
        tool_calls = []
        tool_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(tool_pattern, response, re.DOTALL)
        for match in matches:
            try:
                tool_data = json.loads(match.strip())
                tool_calls.append(
                    {
                        "name": tool_data["name"],
                        "args": tool_data.get("arguments", tool_data.get("args", {})),
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                )
            except json.JSONDecodeError:
                continue
        return tool_calls



    
