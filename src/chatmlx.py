from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from typing import List, Optional, Any, Dict, Union, Sequence, Iterator, AsyncIterator, ClassVar
import os
import requests
import json
import uuid
import asyncio
import time
import re

# Pydantic (para atributos privados de runtime)
from pydantic import PrivateAttr, Field

### MLX imports
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate, stream_generate

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"
Magistral_MODEL_ID = "lmstudio-community/Magistral-Small-2509-MLX-4bit"


class ChatMLX(BaseChatModel):
    """
    LLM customizado compatível com LangChain + MLX.

    Funcionalidades:
    - Completions sync/async
    - Streaming sync/async
    - bind_tools() sem recriar instância
    - Detecção de tool calls via <tool_call>{...}</tool_call> (stream e não-stream)
    - Otimizações de performance (batch decode, stop window, buffers)
    """

    # -----------------------------
    # Constantes / Regex pré-compilado
    # -----------------------------
    START_TAG: ClassVar[str] = "<tool_call>"
    END_TAG: ClassVar[str] = "</tool_call>"
    _tool_re: ClassVar[re.Pattern] = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # -----------------------------
    # Campos "declarativos" (pydantic)
    # -----------------------------
    model_name: Optional[str] = Qwen_MODEL_ID
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
            return True
        except Exception as e:
            print(f"🔍 Error - {e}")
            return False

    def _ensure_loaded(self):
        """Garante que _model e _tokenizer estejam carregados no runtime."""
        if self._model is None or self._tokenizer is None:
            ok = self.init()
            if not ok:
                raise RuntimeError("Falha ao carregar modelo/tokenizer no ChatMLX.init()")

    @property
    def _llm_type(self) -> str:
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
        api_messages = self._convert_messages(messages)
        try:
            response = self._call_generate_mlx_lm(api_messages, stop, **kwargs)
            content = response.get("content", "Default response from custom LLM")
            tool_calls = response.get("tool_calls", [])
        except Exception as e:
            content = f"Response from {self.model_name}: Hello from ChatMLX! (Error: {str(e)})"
            tool_calls = []
        message = AIMessage(content=content, tool_calls=tool_calls or None)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        api_messages = self._convert_messages(messages)
        try:
            response = await self._acall_generate_mlx_lm(api_messages, stop, **kwargs)
            content = response.get("content", "Default async response from custom LLM")
            tool_calls = response.get("tool_calls", [])
        except Exception as e:
            content = f"Async response from {self.model_name}: Hello from ChatMLX! (Error: {str(e)})"
            tool_calls = []
        message = AIMessage(content=content, tool_calls=tool_calls or None)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    # ---------- HELPERS DE STREAM ----------
    def _iter_stream_tokens(self, prompt: str) -> Iterator[str]:
        """
        Itera saídas do MLX e normaliza para str.
        Otimização: decodifica tokens em lote para reduzir overhead.
        """
        token_buf: List[int] = []

        def flush_tokens():
            if token_buf:
                try:
                    text = self._tokenizer.decode(token_buf)
                    if text:
                        yield text
                finally:
                    token_buf.clear()

        for piece in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=self.max_tokens,
            sampler=self._mlx_sampler,
            logits_processors=self._mlx_logits_processors,
        ):
            # 1) strings diretas
            if isinstance(piece, str):
                yield from flush_tokens()
                if piece:
                    yield piece
                continue

            # 2) bytes
            if isinstance(piece, (bytes, bytearray)):
                yield from flush_tokens()
                s = piece.decode("utf-8", errors="ignore")
                if s:
                    yield s
                continue

            # 3) dict: 'text' e/ou 'token'
            if isinstance(piece, dict):
                if "text" in piece and piece["text"]:
                    yield from flush_tokens()
                    s = str(piece["text"])
                    if s:
                        yield s
                elif "token" in piece and piece["token"] is not None:
                    try:
                        token_buf.append(int(piece["token"]))
                    except Exception:
                        pass
                continue

            # 4) objeto com atributos (ex.: GenerationResponse)
            text_attr = getattr(piece, "text", None)
            if isinstance(text_attr, str) and text_attr:
                yield from flush_tokens()
                yield text_attr
                continue

            token_attr = getattr(piece, "token", None)
            if token_attr is not None:
                try:
                    token_buf.append(int(token_attr))
                except Exception:
                    pass
                continue

            # 5) fallback
            try:
                s = str(piece)
                if s:
                    yield from flush_tokens()
                    yield s
            except Exception:
                continue

        # flush final
        yield from flush_tokens()

    def _parse_stream_piece(
        self,
        scan_buffer: str,
        piece: str,
        in_tool: bool,
        tool_buf_parts: List[str],
    ) -> tuple[str, bool, str, str, List[Dict[str, Any]]]:
        """
        Parser incremental com buffer persistente de fronteira e buffer de partes para tool.
        Retorna: clean_out, in_tool, scan_buffer, (tool_buf_parts mutado), new_tools
        """
        scan_buffer += piece
        clean_out = ""
        new_tools: List[Dict[str, Any]] = []

        START = self.START_TAG
        END = self.END_TAG

        while True:
            if in_tool:
                end_idx = scan_buffer.find(END)
                if end_idx == -1:
                    tool_buf_parts.append(scan_buffer)
                    scan_buffer = ""
                    break
                else:
                    tool_content = "".join(tool_buf_parts) + scan_buffer[:end_idx]
                    scan_buffer = scan_buffer[end_idx + len(END):]
                    tool_buf_parts.clear()
                    in_tool = False
                    try:
                        tool_data = json.loads(tool_content.strip())
                        new_tools.append({
                            "name": tool_data["name"],
                            "args": tool_data.get("arguments", tool_data.get("args", {})),
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        })
                    except Exception:
                        pass
                    # continua; pode haver mais um START logo após o END
            else:
                start_idx = scan_buffer.find(START)
                if start_idx == -1:
                    # não achou START completo; preserve cauda para fronteira
                    keep = len(START) - 1  # 10 chars
                    if len(scan_buffer) > keep:
                        emit_upto = len(scan_buffer) - keep
                        clean_out += scan_buffer[:emit_upto]
                        scan_buffer = scan_buffer[emit_upto:]
                    break
                else:
                    # emite o texto antes do START e entra em modo tool
                    clean_out += scan_buffer[:start_idx]
                    scan_buffer = scan_buffer[start_idx + len(START):]
                    in_tool = True
                    tool_buf_parts.clear()
                    # volta ao loop para procurar END

        return clean_out, in_tool, scan_buffer, new_tools

    def _make_stop_checker(self, stop: List[str]):
        """Cria verificador de stop por janela deslizante (sem concatenar tudo)."""
        if not stop:
            return lambda new_text, tail: (new_text, False, tail)

        max_len = max(len(s) for s in stop)
        stop_tuple = tuple(stop)

        def check(new_text: str, window_tail: str) -> tuple[str, bool, str]:
            if not new_text:
                return "", False, window_tail
            probe = window_tail + new_text
            first_hit_idx = None
            first_hit_len = 0
            for s in stop_tuple:
                idx = probe.find(s)
                if idx != -1 and (first_hit_idx is None or idx < first_hit_idx):
                    first_hit_idx = idx
                    first_hit_len = len(s)
            if first_hit_idx is None:
                new_tail_len = max_len - 1
                new_tail = probe[-new_tail_len:] if new_tail_len > 0 and len(probe) > new_tail_len else probe
                return new_text, False, new_tail
            emit_len = max(0, first_hit_idx - len(window_tail))
            to_emit = new_text[:emit_len]
            return to_emit, True, ""  # paramos após emitir até antes do stop
        return check

    # ---------- STREAM SÍNCRONO ----------
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            self._ensure_loaded()
            api_messages = self._convert_messages(messages)
            prompt = self._tokenizer.apply_chat_template(
                api_messages, add_generation_prompt=True, tools=self.bound_tools
            )
        except Exception as e:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream init error] {e}"))
            return

        in_tool = False
        tool_buf_parts: List[str] = []
        scan_buffer = ""
        collected_tools: List[Dict[str, Any]] = []
        check_stop = self._make_stop_checker(stop or [])
        window_tail = ""

        try:
            for raw_piece in self._iter_stream_tokens(prompt):
                clean_piece, in_tool, scan_buffer, new_tools = self._parse_stream_piece(
                    scan_buffer=scan_buffer,
                    piece=raw_piece,
                    in_tool=in_tool,
                    tool_buf_parts=tool_buf_parts,
                )
                if new_tools:
                    collected_tools.extend(new_tools)

                to_emit, should_stop, window_tail = check_stop(clean_piece, window_tail)
                if to_emit:
                    chunk = AIMessageChunk(content=to_emit)
                    if run_manager:
                        try:
                            run_manager.on_llm_new_token(to_emit)
                        except Exception:
                            pass
                    yield ChatGenerationChunk(message=chunk)
                if should_stop:
                    break

            # descartar cauda parcial se não estiver dentro de tool
            if scan_buffer and not in_tool:
                scan_buffer = ""

            if collected_tools:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=collected_tools))

        except Exception as e:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {e}"))

    # ---------- STREAM ASSÍNCRONO ----------
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        try:
            self._ensure_loaded()
            api_messages = self._convert_messages(messages)
            prompt = self._tokenizer.apply_chat_template(
                api_messages, add_generation_prompt=True, tools=self.bound_tools
            )
        except Exception as e:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream init error] {e}"))
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        def producer():
            try:
                for p in self._iter_stream_tokens(prompt):
                    loop.call_soon_threadsafe(queue.put_nowait, p)
            except Exception as ex:
                loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(ex)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, {"__eos__": True})

        prod_fut = loop.run_in_executor(None, producer)

        in_tool = False
        tool_buf_parts: List[str] = []
        scan_buffer = ""
        collected_tools: List[Dict[str, Any]] = []
        check_stop = self._make_stop_checker(stop or [])
        window_tail = ""

        try:
            while True:
                item = await queue.get()
                if isinstance(item, dict) and item.get("__eos__"):
                    break
                if isinstance(item, dict) and "__err__" in item:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {item['__err__']}"))
                    return

                raw_piece = item  # já normalizado para str
                clean_piece, in_tool, scan_buffer, new_tools = self._parse_stream_piece(
                    scan_buffer=scan_buffer,
                    piece=raw_piece,
                    in_tool=in_tool,
                    tool_buf_parts=tool_buf_parts,
                )
                if new_tools:
                    collected_tools.extend(new_tools)

                to_emit, should_stop, window_tail = check_stop(clean_piece, window_tail)
                if to_emit:
                    chunk = AIMessageChunk(content=to_emit)
                    if run_manager:
                        try:
                            await run_manager.on_llm_new_token(to_emit)
                        except Exception:
                            pass
                    yield ChatGenerationChunk(message=chunk)

                if should_stop:
                    # drena até EOS para encerrar ordenadamente
                    while True:
                        itm = await queue.get()
                        if isinstance(itm, dict) and itm.get("__eos__"):
                            break
                    break

            if scan_buffer and not in_tool:
                scan_buffer = ""

            if collected_tools:
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=collected_tools))

            try:
                await prod_fut
            except Exception:
                pass

        except Exception as e:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[astream error] {e}"))

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

    def _call_generate_mlx_lm(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
        """Chamada não-stream do MLX (generate)."""
        self._ensure_loaded()
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

        detected_tool_calls = self._detect_real_tool_calls(response)
        if detected_tool_calls:
            return {"content": "", "tool_calls": detected_tool_calls}
        else:
            return {"content": response, "tool_calls": []}

    async def _acall_generate_mlx_lm(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
        """Versão assíncrona de _call_generate_mlx_lm usando thread pool (não bloqueia o loop)."""
        self._ensure_loaded()
        try:
            prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tools=self.bound_tools,
            )
        except Exception as e:
            print(f"🔍 Async Error - {e}")
            return {"content": f"Async Error: {str(e)}", "tool_calls": []}

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

        detected_tool_calls = self._detect_real_tool_calls(response)
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

    # --------------------------------
    # Detecção de tool calls no texto completo (fallback)
    # --------------------------------
    def _detect_real_tool_calls(self, response: str) -> List[Dict]:
        tool_calls = []
        for match in self._tool_re.findall(response):
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


