from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, Sequence

from pydantic import PrivateAttr, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import CallbackManagerForLLMRun

from render_harmony import RenderHarmony
import asyncio

# MLX
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm import load, generate, stream_generate

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-4bit"

class MyChatModel(BaseChatModel):
    # -----------------------------
    # Campos Pydantic
    # -----------------------------
    model_name: Optional[str] = DEFAULT_MODEL_ID
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4098
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.15
    repetition_context_size: int = 50
    use_gpt_harmony_response_format: bool = False

    # Evita estado compartilhado
    bound_tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # -----------------------------
    # Privados (runtime)
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
            print(f"[Init Error] {e}")
            return False

    def _ensure_loaded(self):
        if self._model is None or self._tokenizer is None or self._render_harmony is None:
            if not self.init():
                raise RuntimeError("Falha ao carregar modelo/tokenizer/RenderHarmony")

    @property
    def _llm_type(self) -> str:
        return "MyChatModel"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    # ---------------------------
    # Geração síncrona
    # ---------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_loaded()

        # HARMONY → TOKENS p/ MLX
        tokens = self._render_harmony.render_harmony_conversation(messages, bound_tools=self.bound_tools)

        try:
            resp_text = generate(
                self._model,
                self._tokenizer,
                tokens,  # MLX recebe TOKENS
                max_tokens=self.max_tokens,
                sampler=self._mlx_sampler,
                logits_processors=self._mlx_logits_processors,
            )
        except Exception as e:
            ai = AIMessage(content=f"[generate error] {e}")
            return ChatResult(generations=[ChatGeneration(message=ai)])

        # stop manual (se o backend não suportar)
        if stop:
            for s in stop:
                i = resp_text.find(s)
                if i != -1:
                    resp_text = resp_text[:i]
                    break

        finals = self._render_harmony.detect_harmony_final_messages(resp_text)
        final_text = finals[-1] if finals else resp_text

        detected = self._render_harmony.detect_harmony_tool_calls(resp_text)
        tool_calls: List[ToolCall] = self._render_harmony.to_lc_tool_calls(detected) if detected else []

        content = "" if tool_calls else (final_text or "")

        input_tokens = len(tokens)  # prompt já está em tokens Harmony
        # conte tokens de saída com o tokenizer do MLX
        try:
            output_tokens = len(self._tokenizer.encode(resp_text))
        except Exception:
            # fallback (menos preciso) se o tokenizer não expuser encode
            output_tokens = len(resp_text)

        total_tokens = input_tokens + output_tokens

        ai = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,      # <-- obrigatório no v1
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---------------------------
    # Geração assíncrona
    # ---------------------------
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_loaded()

        tokens = self._render_harmony.render_harmony_conversation(messages, bound_tools=self.bound_tools)

        try:
            resp_text = await asyncio.to_thread(
                generate,
                self._model, self._tokenizer, tokens,
                self.max_tokens,  # max_tokens como arg posicional é aceito por generate
                self._mlx_sampler,
                self._mlx_logits_processors,
            )
        except Exception as e:
            ai = AIMessage(content=f"[async generate error] {e}")
            return ChatResult(generations=[ChatGeneration(message=ai)])

        if stop:
            for s in stop:
                i = resp_text.find(s)
                if i != -1:
                    resp_text = resp_text[:i]
                    break

        finals = self._render_harmony.detect_harmony_final_messages(resp_text)
        final_text = finals[-1] if finals else resp_text

        detected = self._render_harmony.detect_harmony_tool_calls(resp_text)
        tool_calls: List[ToolCall] = self._render_harmony.to_lc_tool_calls(detected) if detected else []

        content = "" if tool_calls else (final_text or "")
        input_tokens = len(tokens)

        try:
            output_tokens = len(self._tokenizer.encode(resp_text))
        except Exception:
            # fallback (menos preciso) se o tokenizer não expuser encode
            output_tokens = len(resp_text)

        total_tokens = input_tokens + output_tokens

        ai = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,      # <-- obrigatório no v1
            },
            response_metadata={"model_name": self.model_name},
        )
        return ChatResult(generations=[ChatGeneration(message=ai)])

    # ---------------------------
    # Streaming síncrono → ChatGenerationChunk
    # ---------------------------
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Usa threading+queue (sem asyncio) no modo síncrono.
        """
        self._ensure_loaded()
        from queue import Queue
        from threading import Thread

        tokens = self._render_harmony.render_harmony_conversation(messages, bound_tools=self.bound_tools)
        q: "Queue[Union[str, dict]]" = Queue(maxsize=256)

        def producer():
            try:
                for part in stream_generate(
                    self._model, self._tokenizer, tokens,
                    max_tokens=self.max_tokens,
                    sampler=self._mlx_sampler,
                    logits_processors=self._mlx_logits_processors,
                ):
                    q.put(part.text)  # pedaço de texto
            except Exception as e:
                q.put({"__err__": str(e)})
            finally:
                q.put({"__eos__": True})

        Thread(target=producer, daemon=True).start()

        acc = ""
        while True:
            item = q.get()
            if isinstance(item, dict) and item.get("__eos__"):
                break
            if isinstance(item, dict) and "__err__" in item:
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {item['__err__']}"))
                return

            token_piece: str = item
            acc += token_piece

            detected = self._render_harmony.detect_harmony_tool_calls(acc)
            if detected:
                tool_calls: List[ToolCall] = self._render_harmony.to_lc_tool_calls(detected)
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=tool_calls))
                # drenar até EOS
                while True:
                    end_item = q.get()
                    if isinstance(end_item, dict) and end_item.get("__eos__"):
                        break
                break
            else:
                final_messages = self._render_harmony.detect_harmony_final_messages(acc)
                if final_messages:
                    # if run_manager is not None and token_piece:
                    #     run_manager.on_llm_new_token(token_piece)  # opcional (legacy-style)

    
                    yield ChatGenerationChunk(message=AIMessageChunk(content=token_piece))
                else:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=""))

    # ---------------------------
    # Streaming assíncrono → ChatGenerationChunk
    # ---------------------------
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        # run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self._ensure_loaded()
        tokens = self._render_harmony.render_harmony_conversation(messages, bound_tools=self.bound_tools)

        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue(maxsize=256)

        def producer():
            try:
                for part in stream_generate(
                    self._model, self._tokenizer, tokens,
                    max_tokens=self.max_tokens,
                    sampler=self._mlx_sampler,
                    logits_processors=self._mlx_logits_processors,
                ):
                    loop.call_soon_threadsafe(q.put_nowait, part.text)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, {"__err__": str(e)})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, {"__eos__": True})

        loop.run_in_executor(None, producer)

        acc = ""
        while True:
            item = await q.get()
            if isinstance(item, dict) and item.get("__eos__"):
                break
            if isinstance(item, dict) and "__err__" in item:
                yield ChatGenerationChunk(message=AIMessageChunk(content=f"[stream error] {item['__err__']}"))
                return

            token_piece: str = item
            acc += token_piece

            detected = self._render_harmony.detect_harmony_tool_calls(acc)
            if detected:
                tool_calls: List[ToolCall] = self._render_harmony.to_lc_tool_calls(detected)
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_calls=tool_calls))
                # drenar até EOS
                while True:
                    end_item = await q.get()
                    if isinstance(end_item, dict) and end_item.get("__eos__"):
                        break
                break
            else:
                final_messages = self._render_harmony.detect_harmony_final_messages(acc)
                if final_messages:
                    # if run_manager is not None and token_piece:
                    #     run_manager.on_llm_new_token(token_piece)  # opcional (legacy-style)

    
                    yield ChatGenerationChunk(message=AIMessageChunk(content=token_piece))
                else:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=""))


                

    # --------------------------------
    # Ferramentas (compat v1)
    # --------------------------------
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], type, BaseTool]],
        **kwargs: Any,
    ) -> "MyChatModel":
        """Vincula ferramentas no modelo atual (versão compat).
        Obs.: Esta implementação é mutável; considere uma variante imutável que retorna uma cópia.
        """
        formatted_tools: List[Dict[str, Any]] = []
        tool_functions: Dict[str, Any] = {}

        for tool in tools:
            if hasattr(tool, "name"):  # BaseTool
                formatted_tools.append(convert_to_openai_tool(tool))
                tool_functions[getattr(tool, "name")] = getattr(tool, "func", None)
            elif isinstance(tool, dict):  # já no formato correto
                formatted_tools.append(tool)
                if "func" in tool and "name" in tool:
                    tool_functions[tool["name"]] = tool["func"]
            else:  # função/classe Pydantic anotada
                formatted_tools.append(convert_to_openai_tool(tool))
                if hasattr(tool, "__name__"):
                    tool_functions[tool.__name__] = tool

        self.bound_tools = formatted_tools
        self.tool_choice = kwargs.get("tool_choice", self.tool_choice)
        self._tool_functions = tool_functions

        # Se Runnable.bind existir na sua versão, devolve um binding “declarativo”.
        if hasattr(self, "bind"):
            return self.bind(tools=formatted_tools, tool_choice=self.tool_choice)

        return self

