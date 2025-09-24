from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.tools import BaseTool

from pydantic import PrivateAttr, Field
from render_harmony import RenderHarmony
import asyncio
### MLX imports
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate, stream_generate

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"

class MyChatModel(BaseChatModel):

    # -----------------------------
    # Campos "declarativos" (pydantic)
    # -----------------------------
    model_name: Optional[str] = DEFAULT_MODEL_ID
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.15
    repetition_context_size: int = 50
    use_gpt_harmony_response_format: bool = False  # futuro

    # Evita estado compartilhado entre instâncias
    bound_tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # -----------------------------
    # Atributos privados (runtime)
    # -----------------------------
    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _mlx_sampler: Any = PrivateAttr(default=None)
    _mlx_logits_processors: Any = PrivateAttr(default=None)
    _tool_functions: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _render_harmony: RenderHarmony = PrivateAttr(default=None)

     # --------------------------------
    # Inicialização / carregamento
    # --------------------------------
    def init(self) -> bool:
        """Initialize the model and tokenizer (runtime)."""
        print(f"🔍 Debug - model_name: {self.model_name}")
        try:
            self._mlx_sampler = make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k)
            self._mlx_logits_processors = make_logits_processors(
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            )
            self._model, self._tokenizer = load(self.model_name)
            self._render_harmony = RenderHarmony()
            
            return True
        except Exception as e:
            print(f"🔍 Error - {e}")
            return False

    def _ensure_loaded(self):
        """Garante que _model e _tokenizer estejam carregados no runtime."""
        if self._model is None or self._tokenizer is None:
            ok = self.init()
            if not ok:
                raise RuntimeError("Falha ao carregar modelo/tokenizer no MyChatModel.init()")

    @property
    def _llm_type(self) -> str:
        return "MyChatModel"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}
    
    # ---------------------------
    # Geração síncrona
    # ---------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # payload = self._to_provider_format(messages, stop=stop, **kwargs)

        prompt = self._render_harmony.render_harmony_conversation(messages)

        try:
            response = generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=self.max_tokens,
                sampler=self._mlx_sampler,
                logits_processors=self._mlx_logits_processors,
            )
        except Exception as e:
            print(f"🔍  Generate Error - {e}")
            return {"content": f" Generate Error: {str(e)}", "tool_calls": []}

        analysis_messages = self._render_harmony.detect_harmony_analysis_messages(response)
        commentary_messages = self._render_harmony.detect_harmony_commentary_messages(response)
        final_messages = self._render_harmony.detect_harmony_final_messages(response)
        detected_tool_calls = self._render_harmony.detect_harmony_tool_calls(response)
        

        if detected_tool_calls:
            content = ""
            tool_calls = detected_tool_calls
        else:
            content = response
            tool_calls = []

        message = AIMessage(content=content, tool_calls=tool_calls or [])
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
        

    # ---------------------------
    # Geração assíncrona
    # ---------------------------
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        #   payload = self._to_provider_format(messages, stop=stop, **kwargs)
        prompt = self._render_harmony.render_harmony_conversation(messages,bound_tools=self.bound_tools)

        # ---- EXEMPLO (substitua pela sua chamada real) ----
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
   
        analysis_messages = self._render_harmony.detect_harmony_analysis_messages(response)
        commentary_messages = self._render_harmony.detect_harmony_commentary_messages(response)
        final_messages = self._render_harmony.detect_harmony_final_messages(response)
        detected_tool_calls = self._render_harmony.detect_harmony_tool_calls(response)

        if detected_tool_calls:
            content = ""
            tool_calls = detected_tool_calls
        else:
            content = response
            tool_calls = []

        message = AIMessage(content=content, tool_calls=tool_calls or [])
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # ---------------------------
    # Streaming síncrono -> AIMessageChunk
    # ---------------------------
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        prompt = self._render_harmony.render_harmony_conversation(messages,bound_tools=self.bound_tools)
        response = ""
        
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        
        try:
            def producer():
                try:
                    for response in stream_generate(self._model, self._tokenizer, prompt, 
                    max_tokens=self.max_tokens, sampler=self._mlx_sampler, 
                    logits_processors=self._mlx_logits_processors):
                        loop.call_soon_threadsafe(queue.put_nowait, response.text)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(e)})
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, {"__eos__": True})
            # Executa producer em thread separada
            prod_fut = loop.run_in_executor(None, producer)
            # Consome tokens da queue (não bloqueia)
            while True:
                token = queue.get()  # Assíncrono!
                if isinstance(token, dict) and token.get("__eos__"):
                    break
                if isinstance(token, dict) and "__err__" in token:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {token['__err__']}"))
                    return
                raw_piece = token  # já normalizado para str
               

                response += raw_piece
                detected_tool_calls = self._render_harmony.detect_harmony_tool_calls(response)
                # analysis_messages = self._render_harmony.detect_harmony_analysis_messages(response)
                # commentary_messages = self._render_harmony.detect_harmony_commentary_messages(response)
                final_messages = self._render_harmony.detect_harmony_final_messages(response)
                


                if detected_tool_calls:
                    content = ""
                    tool_calls = detected_tool_calls
                    print (f"🔍 Detected tool calls: {tool_calls}")
                else:
                    if final_messages:
                        content = token
                    else:
                        content = ""
                    tool_calls = []
                
                yield ChatGenerationChunk(message=AIMessageChunk(content=content, tool_calls=tool_calls))

                if detected_tool_calls:
                    while True:
                        itm = queue.get()
                        if isinstance(itm, dict) and itm.get("__eos__"):
                            break
                    break

               
        except Exception as e:
            print(f"🔍 Async Generate Error - {e}")
            error_message = {"content": f"Async Generate Error: {str(e)}", "tool_calls": []}
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_message))
            return 
            

    # ---------------------------
    # Streaming assíncrono -> AIMessageChunk
    # ---------------------------
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        
        prompt = self._render_harmony.render_harmony_conversation(messages,bound_tools=self.bound_tools)
        response = ""
        
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        
        try:
            def producer():
                try:
                    for response in stream_generate(self._model, self._tokenizer, prompt, 
                    max_tokens=self.max_tokens, sampler=self._mlx_sampler, 
                    logits_processors=self._mlx_logits_processors):
                        loop.call_soon_threadsafe(queue.put_nowait, response.text)
                except Exception as e:
                    loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(e)})
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, {"__eos__": True})
            # Executa producer em thread separada
            prod_fut = loop.run_in_executor(None, producer)
            # Consome tokens da queue (não bloqueia)
            while True:
                token = await queue.get()  # Assíncrono!
                if isinstance(token, dict) and token.get("__eos__"):
                    break
                if isinstance(token, dict) and "__err__" in token:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {token['__err__']}"))
                    return
                raw_piece = token  # já normalizado para str
               

                response += raw_piece
                detected_tool_calls = self._render_harmony.detect_harmony_tool_calls(response)
                # analysis_messages = self._render_harmony.detect_harmony_analysis_messages(response)
                # commentary_messages = self._render_harmony.detect_harmony_commentary_messages(response)
                final_messages = self._render_harmony.detect_harmony_final_messages(response)
                


                if detected_tool_calls:
                    content = ""
                    tool_calls = detected_tool_calls
                    print (f"🔍 Detected tool calls: {tool_calls}")
                else:
                    if final_messages:
                        content = token
                    else:
                        content = ""
                    tool_calls = []
                
                yield ChatGenerationChunk(message=AIMessageChunk(content=content, tool_calls=tool_calls))

                if detected_tool_calls:
                    while True:
                        itm = await queue.get()
                        if isinstance(itm, dict) and itm.get("__eos__"):
                            break
                    break

               
        except Exception as e:
            print(f"🔍 Async Generate Error - {e}")
            error_message = {"content": f"Async Generate Error: {str(e)}", "tool_calls": []}
            yield ChatGenerationChunk(message=AIMessageChunk(content=error_message))
            return 
            

    # --------------------------------
    # Ferramentas
    # --------------------------------
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> "MyChatModel":
        """Vincula ferramentas sem recriar instância."""
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
        self.bound_tools = formatted_tools
        self.tool_choice = kwargs.get("tool_choice", self.tool_choice)
        self._tool_functions = tool_functions
        
        return self

    # ---------------------------
    # Adapter: mensagens -> payload do provedor
    # ---------------------------
    def _to_provider_format(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        formatted = []
        for m in messages:
            # m.type: "system" | "human" | "ai" | "tool" ...
            formatted.append({"role": m.type, "content": m.content})
        payload = {
            "model": self.model_name,
            "messages": formatted,
        }
        if kwargs.get("stop"):
            payload["stop"] = kwargs["stop"]
        return payload
