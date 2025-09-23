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

### MLX imports
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate, stream_generate

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"
Magistral_MODEL_ID = "lmstudio-community/Magistral-Small-2509-MLX-4bit"
# Harmony encoding
encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)


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
    # Harmony encoding
    # encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

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

    # --------------------------------
    # Inicialização / carregamento
    # --------------------------------
    def init(self) -> bool:
        """Initialize the model and tokenizer (runtime)."""
        if self.use_gpt_harmony_response_format:
            self.model_name = DEFAULT_MODEL_ID
        else:
            self.model_name = Qwen_MODEL_ID
            
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
        message = AIMessage(content=content, tool_calls=tool_calls or [])
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
        message = AIMessage(content=content, tool_calls=tool_calls or [])
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

    def _parse_harmony_stream_piece(
        self,
        scan_buffer: str,
        piece: str,
        in_final_channel: bool,
        final_buf_parts: List[str],
    ) -> tuple[str, bool, str, List[Dict[str, Any]]]:
        """
        Parser incremental para formato Harmony com suporte a canais.
        Filtra apenas mensagens do canal 'final' para exibição ao usuário.
        Retorna: clean_out, in_final_channel, scan_buffer, new_tools
        """
        scan_buffer += piece
        clean_out = ""
        new_tools: List[Dict[str, Any]] = []

        # Padrões para diferentes canais Harmony
        FINAL_START = "<|channel|>final<|message|>"
        CHANNEL_END_PATTERNS = ["<|end|>", "<|start|>"]
        
        # Padrão para tool calls no formato Harmony
        TOOL_CALL_PATTERN = r'<\|start\|>assistant<\|channel\|>commentary\s+to=functions\.([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+<\|constrain\|>(\w+))?<\|message\|>(.*?)<\|call\|>'

        while True:
            if in_final_channel:
                # Procurar fim do canal final
                end_found = False
                end_idx = len(scan_buffer)
                
                for end_pattern in CHANNEL_END_PATTERNS:
                    idx = scan_buffer.find(end_pattern)
                    if idx != -1 and idx < end_idx:
                        end_idx = idx
                        end_found = True
                
                if not end_found:
                    # Não encontrou fim, adiciona tudo ao buffer
                    final_buf_parts.append(scan_buffer)
                    scan_buffer = ""
                    break
                else:
                    # Encontrou fim, processa conteúdo do canal final
                    final_content = "".join(final_buf_parts) + scan_buffer[:end_idx]
                    scan_buffer = scan_buffer[end_idx:]
                    final_buf_parts.clear()
                    in_final_channel = False
                    
                    # Emite o conteúdo do canal final
                    clean_out += final_content.strip()
                    # continua; pode haver mais conteúdo após o fim
            else:
                # Procurar início do canal final
                final_start_idx = scan_buffer.find(FINAL_START)
                
                # Procurar tool calls
                tool_match = re.search(TOOL_CALL_PATTERN, scan_buffer, re.DOTALL)
                
                if tool_match and (final_start_idx == -1 or tool_match.start() < final_start_idx):
                    # Encontrou tool call antes do canal final
                    function_name = tool_match.group(1)
                    constraint_type = tool_match.group(2) or "json"
                    content = tool_match.group(3).strip()
                    
                    try:
                        if constraint_type.lower() == "json":
                            args_data = json.loads(content)
                        else:
                            args_data = {"content": content}
                        
                        new_tools.append({
                            "name": function_name,
                            "args": args_data,
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "tool_call",
                        })
                    except json.JSONDecodeError:
                        pass
                    
                    # Remove o tool call do buffer
                    scan_buffer = scan_buffer[tool_match.end():]
                    continue
                
                elif final_start_idx != -1:
                    # Encontrou início do canal final
                    # Descarta tudo antes do canal final (analysis, commentary, etc.)
                    scan_buffer = scan_buffer[final_start_idx + len(FINAL_START):]
                    in_final_channel = True
                    final_buf_parts.clear()
                    # volta ao loop para processar conteúdo do canal
                else:
                    # Não encontrou canal final nem tool calls
                    # Para tool calls, precisamos preservar mais contexto
                    # Procurar se há início de um padrão de tool call no buffer
                    tool_start_pos = scan_buffer.find('<|start|>assistant<|channel|>commentary to=functions.')
                    if tool_start_pos != -1:
                        # Se encontrou início de tool call, preserva a partir desse ponto
                        scan_buffer = scan_buffer[tool_start_pos:]
                    else:
                        # Preserva uma cauda para fronteira
                        tool_pattern_start = '<|start|>assistant<|channel|>commentary to=functions.'
                        keep = max(len(FINAL_START), max(len(p) for p in CHANNEL_END_PATTERNS), len(tool_pattern_start)) - 1
                        if len(scan_buffer) > keep:
                            # Descarta tudo exceto a cauda (não emite nada)
                            scan_buffer = scan_buffer[-keep:]
                    break

        return clean_out, in_final_channel, scan_buffer, new_tools

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
            if self.use_gpt_harmony_response_format:
                prompt = self.render_harmony_conversation(api_messages)
            else:
                prompt = self._tokenizer.apply_chat_template(
                    api_messages, add_generation_prompt=True, tools=self.bound_tools
                )
        except Exception as e:
            yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream init error] {e}"))
            return

        # Inicializar variáveis baseadas no formato
        if self.use_gpt_harmony_response_format:
            in_final_channel = False
            final_buf_parts: List[str] = []
        else:
            in_tool = False
            tool_buf_parts: List[str] = []
        
        scan_buffer = ""
        collected_tools: List[Dict[str, Any]] = []
        check_stop = self._make_stop_checker(stop or [])
        window_tail = ""

        try:
            for raw_piece in self._iter_stream_tokens(prompt):
                if self.use_gpt_harmony_response_format:
                    clean_piece, in_final_channel, scan_buffer, new_tools = self._parse_harmony_stream_piece(
                        scan_buffer=scan_buffer,
                        piece=raw_piece,
                        in_final_channel=in_final_channel,
                        final_buf_parts=final_buf_parts,
                    )
                else:
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

            # descartar cauda parcial se não estiver dentro de tool/canal
            if self.use_gpt_harmony_response_format:
                if scan_buffer and not in_final_channel:
                    scan_buffer = ""
            else:
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
            if self.use_gpt_harmony_response_format:
                prompt = self.render_harmony_conversation(api_messages)
            else:
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

        # Inicializar variáveis baseadas no formato
        if self.use_gpt_harmony_response_format:
            in_final_channel = False
            final_buf_parts: List[str] = []
        else:
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
                if self.use_gpt_harmony_response_format:
                    clean_piece, in_final_channel, scan_buffer, new_tools = self._parse_harmony_stream_piece(
                        scan_buffer=scan_buffer,
                        piece=raw_piece,
                        in_final_channel=in_final_channel,
                        final_buf_parts=final_buf_parts,
                    )
                else:
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

            # descartar cauda parcial se não estiver dentro de tool/canal
            if self.use_gpt_harmony_response_format:
                if scan_buffer and not in_final_channel:
                    scan_buffer = ""
            else:
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
            msg_dict = {
                "role": self._get_role(message.type), 
                "content": message.content
            }
            
            # Se for uma mensagem de tool, adicionar campos específicos
            if message.type == "tool":
                # ToolMessage tem atributos adicionais
                if hasattr(message, "name"):
                    msg_dict["name"] = message.name
                if hasattr(message, "tool_call_id"):
                    msg_dict["tool_call_id"] = message.tool_call_id
                    
            # Se for uma mensagem do assistente com tool calls
            elif message.type == "ai" and hasattr(message, "tool_calls") and message.tool_calls:
                msg_dict["tool_calls"] = message.tool_calls
                
            api_messages.append(msg_dict)
        return api_messages

    def _get_role(self, message_type: str) -> str:
        role_mapping = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        return role_mapping.get(message_type, "user")

    def _call_generate_mlx_lm(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
        """Chamada não-stream do MLX (generate)."""
        self._ensure_loaded()
        try:


            if self.use_gpt_harmony_response_format:
                prompt = self.render_harmony_conversation(messages)
            else:
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

        if self.use_gpt_harmony_response_format:
            detected_tool_calls = self._detect_harmony_tool_calls(response)
            if detected_tool_calls:
                return {"content": "", "tool_calls": detected_tool_calls}
            else:
                return {"content": response, "tool_calls": []}
        else:

            detected_tool_calls = self._detect_real_tool_calls(response)
            if detected_tool_calls:
                return {"content": "", "tool_calls": detected_tool_calls}
            else:
                return {"content": response, "tool_calls": []}

    async def _acall_generate_mlx_lm(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
        """Versão assíncrona de _call_generate_mlx_lm usando thread pool (não bloqueia o loop)."""
        self._ensure_loaded()
        try:
            if self.use_gpt_harmony_response_format:
                prompt = self.render_harmony_conversation(messages)
            else:
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

        if self.use_gpt_harmony_response_format:
            analysis_messages = self._detect_harmony_analysis_messages(response)
            print(f"🔍 Analysis Messages - {analysis_messages}")
            final_messages = self._detect_harmony_final_messages(response)
            print(f"🔍 Final Messages - {final_messages}")
            commentary_messages = self._detect_harmony_commentary_messages(response)
            print(f"🔍 Commentary Messages - {commentary_messages}")
            # response2 = response.replace(analysis_messages, "").replace(final_messages, "").replace(commentary_messages, "")
            detected_tool_calls = self._detect_harmony_tool_calls(response)
            if detected_tool_calls:
                return {"content": "", "tool_calls": detected_tool_calls}
            else:
                return {"content": response, "tool_calls": []}
        else:

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
    # --------------------------------
    # Futuro: renderização no formato Harmony
    # --------------------------------
    def _detect_harmony_tool_calls(self, response: str) -> List[Dict]:
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

    def _detect_harmony_analysis_messages(self, response: str) -> List[str]:
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

    def _detect_harmony_final_messages(self, response: str) -> List[str]:
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

    def _detect_harmony_commentary_messages(self, response: str) -> List[str]:
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

    def render_harmony_conversation(self, messages):
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
        
        # Normalizar entrada: se for um dict único, transformar em lista
        if isinstance(messages, dict):
            message_list = [messages]
        elif isinstance(messages, list):
            message_list = messages
        else:
            raise ValueError("messages deve ser um dict ou uma lista de dicts")
        
        # Criar mensagem de sistema padrão (se não houver uma na lista)
        has_system = any(msg.get("role") == "system" for msg in message_list)
        if not has_system:
            system_message = Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
            )
            harmony_messages.append(system_message)
        
        # Criar mensagem de desenvolvedor com tools (se não houver uma na lista)
        has_developer = any(msg.get("role") == "developer" for msg in message_list)
        if not has_developer and self.bound_tools:
            developer_message = Message.from_role_and_content(
                Role.DEVELOPER,
                DeveloperContent.new().with_instructions("You are a helpful assistant.").with_function_tools(
                    self._convert_tools_to_harmony_format()
                )
            )
            harmony_messages.append(developer_message)
        
        # Processar cada mensagem da lista
        for message in message_list:
            role = message.get("role", "user")
            content = message.get("content", "")
            
            if role == "system":
                # Usar mensagem de sistema personalizada se fornecida
                system_content = SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW)
                if content and content != "You are a helpful assistant that can answer questions and use tools.":
                    # Se há conteúdo personalizado, criar mensagem de sistema customizada
                    system_message = Message.from_role_and_content(Role.SYSTEM, content)
                else:
                    system_message = Message.from_role_and_content(Role.SYSTEM, system_content)
                harmony_messages.append(system_message)
                
            elif role == "user":
                user_message = Message.from_role_and_content(Role.USER, content)
                harmony_messages.append(user_message)
                
            elif role == "assistant":
                # Mensagem do assistente
                if content:
                    assistant_message = Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
                    harmony_messages.append(assistant_message)
                
                # Se há tool calls, adicionar no formato Harmony
                tool_calls = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    # Mensagem de análise (chain of thought)
                    analysis_message = Message.from_role_and_content(
                        Role.ASSISTANT, 
                        f"Need to use function {tool_call['name']}."
                    ).with_channel("analysis")
                    harmony_messages.append(analysis_message)
                    
                    # Mensagem de chamada da função
                    function_call_message = Message.from_role_and_content(
                        Role.ASSISTANT,
                        json.dumps(tool_call.get("args", {}))
                    ).with_channel("commentary").with_recipient(f"functions.{tool_call['name']}").with_content_type("<|constrain|> json")
                    harmony_messages.append(function_call_message)
                    
            elif role == "tool":
                # Mensagem de resposta de tool
                tool_name = message.get("name", "unknown_tool")
                tool_message = Message.from_author_and_content(
                    Author.new(Role.TOOL, f"functions.{tool_name}"),
                    content
                ).with_channel("commentary")
                harmony_messages.append(tool_message)
                
            elif role == "developer":
                # Mensagem de desenvolvedor personalizada
                if self.bound_tools:
                    developer_message = Message.from_role_and_content(
                        Role.DEVELOPER,
                        DeveloperContent.new().with_instructions(content).with_function_tools(
                            self._convert_tools_to_harmony_format()
                        )
                    )
                else:
                    developer_message = Message.from_role_and_content(
                        Role.DEVELOPER,
                        DeveloperContent.new().with_instructions(content)
                    )
                harmony_messages.append(developer_message)
        
        # Construir conversa Harmony
        conversation = Conversation.from_messages(harmony_messages)
        
        # Renderizar para tokens
        tokens = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        print(f"Tokens Harmony gerados: {len(tokens)} para {len(message_list)} mensagem(ns)")
        return tokens
    
    def _convert_tools_to_harmony_format(self):
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
        if isinstance(self.bound_tools, dict):
            # Se é um dicionário único com uma função
            if 'function' in self.bound_tools:
                func_info = self.bound_tools['function']
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
                for tool_name, tool_info in self.bound_tools.items():
                    description = tool_info.get("description", f"Function {tool_name}")
                    parameters = tool_info.get("parameters", {})
                    
                    from openai_harmony import ToolDescription
                    
                    tool_desc = ToolDescription.new(
                        tool_name,
                        description,
                        parameters=parameters
                    )
                    harmony_tools.append(tool_desc)
        
        elif isinstance(self.bound_tools, list):
            # Se é uma lista de ferramentas
            for tool_item in self.bound_tools:
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
