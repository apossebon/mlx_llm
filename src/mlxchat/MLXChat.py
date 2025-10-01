
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union, Sequence, ClassVar
import re
import json
import uuid

from pydantic import PrivateAttr, Field

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.callbacks import CallbackManagerForLLMRun

from src.mlxchat.render_harmony import RenderHarmony
import asyncio

# MLX
from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm import load, generate, stream_generate

DEFAULT_MODEL_ID = "mlx-community/gpt-oss-20b-MXFP4-Q8"
Qwen_MODEL_ID = "lmstudio-community/Qwen3-30B-A3B-Instruct-2507-MLX-4bit"

class MLXChat(BaseChatModel):
    _tool_re: ClassVar[re.Pattern] = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # -----------------------------
    # Campos Pydantic
    # -----------------------------
    model_name: str = DEFAULT_MODEL_ID
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4098
    top_p: float = 0.85
    top_k: int = 40
    repetition_penalty: float = 1.15
    repetition_context_size: int = 50
    use_gpt_harmony_response_format: bool = True
    use_prompt_cache: bool = False
    
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
    _mlx_prompt_cache: Any = PrivateAttr(default=None)
    _tool_functions: Dict[str, Any] = PrivateAttr(default_factory=dict)
    _render_harmony: RenderHarmony = PrivateAttr(default=None)

    # --------------------------------
    # Inicialização / carregamento
    # --------------------------------

    @property
    def _llm_type(self) -> str:
        return "mlx_chat"
    
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

    def init(self) -> bool:
        try:
            self._mlx_sampler = make_sampler(temp=self.temperature, top_p=self.top_p, top_k=self.top_k)
            self._mlx_logits_processors = make_logits_processors(
                repetition_penalty=self.repetition_penalty,
                repetition_context_size=self.repetition_context_size,
            )
            self._model, self._tokenizer = load(self.model_name)
            self._mlx_prompt_cache = make_prompt_cache(self._model) if self.use_prompt_cache else None
            self._render_harmony = RenderHarmony()
            return True
        except Exception as e:
            print(f"[Init Error] {e}")
            return False
        

    def _ensure_loaded(self):
        if self._model is None or self._tokenizer is None:
            if not self.init():
                raise RuntimeError("Falha ao carregar modelo/tokenizer")

    # --------------------------------
    # Generate / AGenerate (métodos principais)
    # --------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._ensure_loaded()

        # HARMONY → TOKENS p/ MLX
        if self.use_gpt_harmony_response_format:
            tokens = self._render_harmony.render_harmony_conversation(messages, bound_tools=self.bound_tools)
        else:
            converted_messages = self._convert_messages(messages)
            tokens = self._tokenizer.apply_chat_template(
                converted_messages,
                add_generation_prompt=True,
                tools=self.bound_tools,
            )

        try:
            resp_llm = generate(
                self._model,
                self._tokenizer,
                tokens,  
                max_tokens=self.max_tokens,
                sampler=self._mlx_sampler,
                logits_processors=self._mlx_logits_processors,
                prompt_cache=self._mlx_prompt_cache if self.use_prompt_cache else None,
                
            )
        except Exception as e:
            ai = AIMessage(content=f"[generate error] {e}")
            return ChatResult(generations=[ChatGeneration(message=ai)])

        # stop manual (se o backend não suportar)
        if stop:
            for s in stop:
                i = resp_llm.find(s)
                if i != -1:
                    resp_llm = resp_llm[:i]
                    break

        if self.use_gpt_harmony_response_format:

            thinking = self._render_harmony.detect_harmony_analysis_messages(resp_llm)
            finals = self._render_harmony.detect_harmony_final_messages(resp_llm)
            final_text = finals[-1] if finals else resp_llm

            detected = self._render_harmony.detect_harmony_tool_calls(resp_llm)
            tool_calls: List[ToolCall] = self._render_harmony.to_lc_tool_calls(detected) if detected else []

            content = "" if tool_calls else (final_text or "")

            input_tokens = len(tokens)  # prompt já está em tokens Harmony
            # conte tokens de saída com o tokenizer do MLX
            try:
                output_tokens = len(self._tokenizer.encode(resp_llm))
            except Exception:
                # fallback (menos preciso) se o tokenizer não expuser encode
                output_tokens = len(resp_llm)

            total_tokens = input_tokens + output_tokens

            ai = AIMessage(
                content=content,
                tool_calls=tool_calls,
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,     
                },
                # if thinking, add to metadata
                metadata={"thinking": thinking} if thinking else {},
                response_metadata={"model_name": self.model_name},
            )
            return ChatResult(generations=[ChatGeneration(message=ai)])
        else:
            print(f"[DEBUG] resp_llm: {resp_llm}")
            detected_tool_calls = self._detect_real_tool_calls(resp_llm)
            if detected_tool_calls:
                resp = {"content": "", "tool_calls": detected_tool_calls}
            else:
                resp = {"content": resp_llm, "tool_calls": []}

            content = resp.get("content", "Default response from custom LLM")
            tool_calls = resp.get("tool_calls", [])
            input_tokens = len(tokens)  # prompt já está em tokens Harmony
            # conte tokens de saída com o tokenizer do MLX
            try:
                output_tokens = len(self._tokenizer.encode(resp_llm))
            except Exception:
                # fallback (menos preciso) se o tokenizer não expuser encode
                output_tokens = len(resp_llm)

            total_tokens = input_tokens + output_tokens

            ai = AIMessage(
                content=content,
                tool_calls=tool_calls,
                usage_metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,     
                },
                # if thinking, add to metadata
                # metadata={"thinking": None},
                response_metadata={"model_name": self.model_name},
            )
            return ChatResult(generations=[ChatGeneration(message=ai)])
    # -----------------------------
    # Métodos auxiliares privados - detecção de tool calls tradicionais
    # -----------------------------
    def _convert_messages(self, messages: List[BaseMessage]) -> List[Dict]:
        api_messages = []
        for message in messages:
            api_messages.append({"role": self._get_role(message.type), "content": message.content})
        return api_messages

    def _get_role(self, message_type: str) -> str:
        role_mapping = {"human": "user", "ai": "assistant", "system": "system"}
        return role_mapping.get(message_type, "user")
    
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
