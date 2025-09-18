
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
# from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
# from langchain_core.callbacks import CallbackManagerForLLMRun
# from langchain_core.tools import BaseTool
# from langchain_core.utils.function_calling import convert_to_openai_tool
# from typing import List, Optional, Any, Dict, Union, Sequence, Iterator, AsyncIterator
# import os
# import requests
# import json
# import uuid
# import asyncio
# import time
# import re

# # Pydantic (para atributos privados de runtime)
# from pydantic import PrivateAttr

# ### MLX imports
# from mlx_lm.sample_utils import make_sampler, make_logits_processors
# from mlx_lm import load, generate, stream_generate

# DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
# Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"


# class ChatMLX(BaseChatModel):
#     """
#     Exemplo de LLM customizado seguindo o padrão LangChain.

#     Funcionalidades incluídas:
#     ✅ Chat completions básicas
#     ✅ Suporte a bind_tools()
#     ✅ Tool calling com formato LangChain
#     ✅ Compatibilidade com agentes
#     ✅ Conversão automática de ferramentas
#     ✅ Simulação de tool calls para demonstração
#     ✅ Streaming síncrono (_stream)
#     ✅ Streaming assíncrono (_astream)
#     ✅ AIMessageChunk com chunks progressivos
#     ✅ Callbacks para tokens em tempo real
#     """

#     # -----------------------------
#     # Campos "declarativos" (pydantic)
#     # -----------------------------
#     model_name: Optional[str] = Qwen_MODEL_ID
#     api_key: Optional[str] = None
#     temperature: float = 0.7
#     max_tokens: int = 1024
#     top_p: float = 0.85
#     top_k: int = 40
#     repetition_penalty: float = 1.15
#     repetition_context_size: int = 50

#     # Propriedades para suporte a ferramentas (parte do "modelo")
#     bound_tools: List[Dict[str, Any]] = []
#     tool_choice: Optional[Union[str, Dict[str, Any]]] = None

#     # -----------------------------
#     # Atributos privados (runtime)
#     # NÃO entram em validação/serialização do Pydantic
#     # -----------------------------
#     _model: Any = PrivateAttr(default=None)
#     _tokenizer: Any = PrivateAttr(default=None)
#     _mlx_sampler: Any = PrivateAttr(default=None)
#     _mlx_logits_processors: Any = PrivateAttr(default=None)
#     _tool_functions: Dict[str, Any] = PrivateAttr(default_factory=dict)

#     # --------------------------------
#     # Inicialização / carregamento
#     # --------------------------------
#     def init(self) -> bool:
#         """
#         Initialize the model and tokenizer (runtime).
#         """
#         print(f"🔍 Debug - model_name: {self.model_name}")
#         try:
#             # sampler/logits processors (opcional)
#             self._mlx_sampler = make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k)
#             self._mlx_logits_processors = make_logits_processors(
#                 repetition_penalty=self.repetition_penalty,
#                 repetition_context_size=self.repetition_context_size,
#             )
#             self._model, self._tokenizer = load(self.model_name)
#             return True
#         except Exception as e:
#             print(f"🔍 Error - {e}")
#             return False

#     def _ensure_loaded(self):
#         """
#         Garante que _model e _tokenizer estejam carregados no runtime.
#         """
#         if self._model is None or self._tokenizer is None:
#             ok = self.init()
#             if not ok:
#                 raise RuntimeError("Falha ao carregar modelo/tokenizer no ChatMLX.init()")

#     @property
#     def _llm_type(self) -> str:
#         """Return identifier of the LLM."""
#         return "ChatMLX"

#     # --------------------------------
#     # Núcleo de geração
#     # --------------------------------
#     def _generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         """
#         Generate chat completion seguindo o padrão LangChain.
#         """
#         api_messages = self._convert_messages(messages)

#         try:
#             response = self._call_api(api_messages, stop, **kwargs)
#             content = response.get("content", "Default response from custom LLM")
#             tool_calls = response.get("tool_calls", [])
#         except Exception as e:
#             content = f"Response from {self.model_name}: Hello from ChatMLX! (Error: {str(e)})"
#             tool_calls = []

#         if tool_calls:
#             message = AIMessage(content=content, tool_calls=tool_calls)
#         else:
#             message = AIMessage(content=content)

#         generation = ChatGeneration(message=message)
#         return ChatResult(generations=[generation])

#     async def _agenerate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         """
#         Versão assíncrona real de _generate.
#         """
#         api_messages = self._convert_messages(messages)

#         try:
#             response = await self._acall_api(api_messages, stop, **kwargs)
#             content = response.get("content", "Default async response from custom LLM")
#             tool_calls = response.get("tool_calls", [])
#         except Exception as e:
#             content = f"Async response from {self.model_name}: Hello from ChatMLX! (Error: {str(e)})"
#             tool_calls = []

#         message = AIMessage(content=content, tool_calls=tool_calls or None)
#         generation = ChatGeneration(message=message)
#         return ChatResult(generations=[generation])

#     # ---------- HELPERS DE STREAM ----------
#     def _iter_stream_tokens(self, prompt: str) -> Iterator[str]:
#         """
#         Itera tokens/trechos vindos do MLX de forma síncrona e normaliza para str.
#         Suporta múltiplos formatos de saída do stream_generate:
#         - str
#         - bytes
#         - objetos com .text e/ou .token
#         - dicts com 'text' e/ou 'token'
#         """
#         for piece in stream_generate(
#             self._model,
#             self._tokenizer,
#             prompt,
#             max_tokens=self.max_tokens,
#             sampler=self._mlx_sampler,
#             logits_processors=self._mlx_logits_processors,
#         ):
#             # 1) strings diretas
#             if isinstance(piece, str):
#                 if piece:
#                     yield piece
#                 continue

#             # 2) bytes
#             if isinstance(piece, (bytes, bytearray)):
#                 s = piece.decode("utf-8", errors="ignore")
#                 if s:
#                     yield s
#                 continue

#             # 3) dict: pode ter 'text' e/ou 'token'
#             if isinstance(piece, dict):
#                 s = ""
#                 if "text" in piece and piece["text"]:
#                     s = str(piece["text"])
#                 elif "token" in piece and piece["token"] is not None:
#                     try:
#                         s = self._tokenizer.decode([int(piece["token"])])
#                     except Exception:
#                         s = ""
#                 if s:
#                     yield s
#                 continue

#             # 4) objeto com atributos (ex.: GenerationResponse)
#             #    tenta .text; se vazio, tenta .token -> decode
#             text_attr = getattr(piece, "text", None)
#             if isinstance(text_attr, str) and text_attr:
#                 yield text_attr
#                 continue

#             token_attr = getattr(piece, "token", None)
#             if token_attr is not None:
#                 try:
#                     s = self._tokenizer.decode([int(token_attr)])
#                     if s:
#                         yield s
#                 except Exception:
#                     pass
#                 continue

#             # 5) fallback: repr como último recurso (evita sumir com algo inesperado)
#             try:
#                 s = str(piece)
#                 if s:
#                     yield s
#             except Exception:
#                 continue


#     # ---- dentro da classe ChatMLX ----

#     def _parse_stream_piece(
#         self,
#         scan_buffer: str,
#         piece: str,
#         in_tool: bool,
#         tool_buf: str,
#     ) -> tuple[str, bool, str, str, list]:
#         """
#         Parser incremental com buffer persistente de fronteira.
#         Retorna: clean_out, in_tool, tool_buf, scan_buffer, new_tools
#         """
#         scan_buffer += piece
#         clean_out = ""
#         new_tools = []

#         START = "<tool_call>"
#         END = "</tool_call>"

#         while True:
#             if in_tool:
#                 end_idx = scan_buffer.find(END)
#                 if end_idx == -1:
#                     tool_buf += scan_buffer
#                     scan_buffer = ""
#                     break
#                 else:
#                     tool_content = tool_buf + scan_buffer[:end_idx]
#                     scan_buffer = scan_buffer[end_idx + len(END):]
#                     tool_buf = ""
#                     in_tool = False
#                     try:
#                         tool_data = json.loads(tool_content.strip())
#                         new_tools.append({
#                             "name": tool_data["name"],
#                             "args": tool_data.get("arguments", tool_data.get("args", {})),
#                             "id": f"call_{uuid.uuid4().hex[:8]}",
#                             "type": "tool_call",
#                         })
#                     except Exception:
#                         pass
#                     # continua; pode haver mais ferramentas na cauda
#             else:
#                 start_idx = scan_buffer.find(START)
#                 if start_idx == -1:
#                     # não achou START completo; preserve a cauda p/ fronteira
#                     keep = len(START) - 1  # 10 chars
#                     if len(scan_buffer) > keep:
#                         emit_upto = len(scan_buffer) - keep
#                         clean_out += scan_buffer[:emit_upto]
#                         scan_buffer = scan_buffer[emit_upto:]
#                     break
#                 else:
#                     # achou START: tudo antes é texto normal
#                     clean_out += scan_buffer[:start_idx]
#                     # entra em modo tool; NÃO emita "<tool_call>"
#                     scan_buffer = scan_buffer[start_idx + len(START):]
#                     in_tool = True
#                     tool_buf = ""
#                     # volta ao loop para procurar o END

#         return clean_out, in_tool, tool_buf, scan_buffer, new_tools

#     # ---------- STREAM SÍNCRONO ----------
#     def _stream(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> Iterator[ChatGenerationChunk]:
#         """
#         Streaming síncrono: emite AIMessageChunk com texto incremental.
#         No final, anexa quaisquer tool_calls detectados ao último chunk.
#         """
#         try:
#             self._ensure_loaded()
#             api_messages = self._convert_messages(messages)
#             prompt = self._tokenizer.apply_chat_template(
#                 api_messages, add_generation_prompt=True, tools=self.bound_tools
#             )
#         except Exception as e:
#             yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream init error] {e}"))
#             return

#         # estados do parser de tool_call
#         in_tool = False
#         tool_buf = ""
#         scan_buffer = ""        # <-- NOVO: persistente entre chunks
#         collected_tools = []
#         stop = stop or []
#         cumulative_text = ""

#         def apply_stop(new_text: str) -> tuple[str, bool]:
#             if not stop:
#                 return new_text, False
#             probe = cumulative_text + new_text
#             first_hit_idx = None
#             for s in stop:
#                 idx = probe.find(s)
#                 if idx != -1 and (first_hit_idx is None or idx < first_hit_idx):
#                     first_hit_idx = idx
#             if first_hit_idx is None:
#                 return new_text, False
#             emit_len = max(0, first_hit_idx - len(cumulative_text))
#             return new_text[:emit_len], True

#         try:
#             for raw_piece in self._iter_stream_tokens(prompt):
#                 # 1) PARSE ANTES DE TUDO
#                 clean_piece, in_tool, tool_buf, scan_buffer, new_tools = self._parse_stream_piece(
#                     scan_buffer=scan_buffer,
#                     piece=raw_piece,
#                     in_tool=in_tool,
#                     tool_buf=tool_buf,
#                 )
#                 if new_tools:
#                     collected_tools.extend(new_tools)

#                 # 2) STOP
#                 to_emit, should_stop = apply_stop(clean_piece)

#                 # 3) EMITIR
#                 if to_emit:
#                     cumulative_text += to_emit
#                     chunk = AIMessageChunk(content=to_emit)
#                     if run_manager:
#                         try:
#                             run_manager.on_llm_new_token(to_emit)
#                         except Exception:
#                             pass
#                     yield ChatGenerationChunk(message=chunk)

#                 if should_stop:
#                     break

#             # Se sobrou algo no scan_buffer e não estamos dentro de tool, descarte (é cauda parcial)
#             if scan_buffer and not in_tool:
#                 scan_buffer = ""

#             # Emite chunk final só com tool_calls (se houver)
#             if collected_tools:
#                 yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=collected_tools))

#         except Exception as e:
#             yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {e}"))

#     # ---------- STREAM ASSÍNCRONO ----------
#     async def _astream(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> AsyncIterator[ChatGenerationChunk]:
#         """
#         Streaming assíncrono: consome o gerador síncrono em executor sem bloquear o event loop.
#         - Mantém scan_buffer persistente (fronteira de <tool_call>)
#         - Detecta e agrega tool_calls no chunk final
#         - Respeita stop sequences
#         - Emite on_llm_new_token
#         """
#         # 1) Preparação
#         try:
#             self._ensure_loaded()
#             api_messages = self._convert_messages(messages)
#             prompt = self._tokenizer.apply_chat_template(
#                 api_messages, add_generation_prompt=True, tools=self.bound_tools
#             )
#         except Exception as e:
#             yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream init error] {e}"))
#             return

#         loop = asyncio.get_running_loop()
#         queue: asyncio.Queue = asyncio.Queue(maxsize=256)

#         def producer():
#             try:
#                 for p in self._iter_stream_tokens(prompt):
#                     loop.call_soon_threadsafe(queue.put_nowait, p)
#             except Exception as ex:
#                 loop.call_soon_threadsafe(queue.put_nowait, {"__err__": str(ex)})
#             finally:
#                 loop.call_soon_threadsafe(queue.put_nowait, {"__eos__": True})

#         prod_fut = loop.run_in_executor(None, producer)

#         # 2) Estados do parser / stop
#         in_tool = False
#         tool_buf = ""
#         scan_buffer = ""              # <--- persistente entre chunks!
#         collected_tools: list = []
#         stop = stop or []
#         cumulative_text = ""

#         async def apply_stop_async(new_text: str) -> tuple[str, bool]:
#             if not stop:
#                 return new_text, False
#             probe = cumulative_text + new_text
#             first_hit_idx = None
#             for s in stop:
#                 idx = probe.find(s)
#                 if idx != -1 and (first_hit_idx is None or idx < first_hit_idx):
#                     first_hit_idx = idx
#             if first_hit_idx is None:
#                 return new_text, False
#             emit_len = max(0, first_hit_idx - len(cumulative_text))
#             return new_text[:emit_len], True

#         # 3) Consumo do stream
#         try:
#             while True:
#                 item = await queue.get()

#                 if isinstance(item, dict) and item.get("__eos__"):
#                     break
#                 if isinstance(item, dict) and "__err__" in item:
#                     yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {item['__err__']}"))
#                     return

#                 raw_piece = item  # string normalizada por _iter_stream_tokens
#                 # >>> usar scan_buffer persistente E o nome certo do argumento <<<
#                 clean_piece, in_tool, tool_buf, scan_buffer, new_tools = self._parse_stream_piece(
#                     scan_buffer=scan_buffer,
#                     piece=raw_piece,
#                     in_tool=in_tool,
#                     tool_buf=tool_buf,
#                 )
#                 if new_tools:
#                     collected_tools.extend(new_tools)

#                 to_emit, should_stop = await apply_stop_async(clean_piece)
#                 if to_emit:
#                     cumulative_text += to_emit
#                     chunk = AIMessageChunk(content=to_emit)
#                     if run_manager:
#                         try:
#                             await run_manager.on_llm_new_token(to_emit)
#                         except Exception:
#                             pass
#                     yield ChatGenerationChunk(message=chunk)

#                 if should_stop:
#                     # drena a fila até EOS para encerrar ordenadamente
#                     while True:
#                         itm = await queue.get()
#                         if isinstance(itm, dict) and itm.get("__eos__"):
#                             break
#                     break

#             # Se sobrou algo no scan_buffer e não estamos em tool, descarte (cauda parcial)
#             if scan_buffer and not in_tool:
#                 scan_buffer = ""

#             # Emite chunk final só com tool_calls (se houver)
#             if collected_tools:
#                 yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=collected_tools))

#             # garante encerramento do produtor
#             try:
#                 await prod_fut
#             except Exception:
#                 pass

#         except Exception as e:
#             yield ChatGenerationChunk(message=AIMessageChunk(content=f"[astream error] {e}"))


#     # --------------------------------
#     # Conversões / utilidades
#     # --------------------------------
#     def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
#         api_messages = []
#         for message in messages:
#             api_messages.append({"role": self._get_role(message.type), "content": message.content})
#         return api_messages

#     def _get_role(self, message_type: str) -> str:
#         role_mapping = {"human": "user", "ai": "assistant", "system": "system"}
#         return role_mapping.get(message_type, "user")

#     def _call_api(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
#         """
#         Chamada real ao modelo MLX (generate).
#         """
#         self._ensure_loaded()

#         try:
#             prompt = self._tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 tools=self.bound_tools,
#             )
#         except Exception as e:
#             print(f"🔍 Error - {e}")
#             return {"content": f"Error: {str(e)}", "tool_calls": []}

#         response = generate(
#             self._model,
#             self._tokenizer,
#             prompt,
#             max_tokens=self.max_tokens,
#             sampler=self._mlx_sampler,
#             logits_processors=self._mlx_logits_processors,
#         )

#         detected_tool_calls = self._detect_real_tool_calls(response)

#         if detected_tool_calls:
#             return {"content": "", "tool_calls": detected_tool_calls}
#         else:
#             return {"content": response, "tool_calls": []}

#     async def _acall_api(self, messages: List[Dict], stop: Optional[List[str]] = None, **kwargs) -> Dict:
#         """
#         Versão assíncrona de _call_api - chamada assíncrona ao modelo MLX.
#         """
#         import asyncio

#         self._ensure_loaded()

#         try:
#             prompt = self._tokenizer.apply_chat_template(
#                 messages,
#                 add_generation_prompt=True,
#                 tools=self.bound_tools,
#             )
#         except Exception as e:
#             print(f"🔍 Async Error - {e}")
#             return {"content": f"Async Error: {str(e)}", "tool_calls": []}

#         try:
#             response = await asyncio.to_thread(
#                 generate,
#                 self._model,
#                 self._tokenizer,
#                 prompt,
#                 max_tokens=self.max_tokens,
#                 sampler=self._mlx_sampler,
#                 logits_processors=self._mlx_logits_processors,
#             )
#         except Exception as e:
#             print(f"🔍 Async Generate Error - {e}")
#             return {"content": f"Async Generate Error: {str(e)}", "tool_calls": []}

#         detected_tool_calls = self._detect_real_tool_calls(response)

#         if detected_tool_calls:
#             return {"content": "", "tool_calls": detected_tool_calls}
#         else:
#             return {"content": response, "tool_calls": []}

#     # --------------------------------
#     # Ferramentas
#     # --------------------------------
#     def bind_tools(
#         self,
#         tools: Sequence[Union[Dict[str, Any], type, BaseTool]],
#         **kwargs: Any,
#     ) -> "ChatMLX":
#         """
#         Vincula ferramentas ao modelo.
#         Importante: NÃO criar nova instância aqui (para não perder _model/_tokenizer).
#         """
#         formatted_tools = []
#         tool_functions = {}

#         for tool in tools:
#             if hasattr(tool, "name"):  # BaseTool
#                 formatted_tools.append(convert_to_openai_tool(tool))
#                 tool_functions[tool.name] = tool.func
#             elif isinstance(tool, dict):  # já é dict no formato certo
#                 formatted_tools.append(tool)
#                 if "func" in tool and "name" in tool:
#                     tool_functions[tool["name"]] = tool["func"]
#             else:  # função/classe Pydantic
#                 formatted_tools.append(convert_to_openai_tool(tool))
#                 if hasattr(tool, "__name__"):
#                     tool_functions[tool.__name__] = tool

#         self.bound_tools = formatted_tools
#         self.tool_choice = kwargs.get("tool_choice", self.tool_choice)
#         self._tool_functions = tool_functions
#         return self


#     # --------------------------------
#     # Detecção de tool calls no texto completo (fallback)
#     # --------------------------------
#     def _detect_real_tool_calls(self, response: str) -> List[Dict]:
#         tool_calls = []
#         tool_pattern = r"<tool_call>(.*?)</tool_call>"
#         matches = re.findall(tool_pattern, response, re.DOTALL)
#         for match in matches:
#             try:
#                 tool_data = json.loads(match.strip())
#                 tool_calls.append(
#                     {
#                         "name": tool_data["name"],
#                         "args": tool_data.get("arguments", tool_data.get("args", {})),
#                         "id": f"call_{uuid.uuid4().hex[:8]}",
#                         "type": "tool_call",
#                     }
#                 )
#             except json.JSONDecodeError:
#                 continue
#         return tool_calls




    
