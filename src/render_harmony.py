import json
import re
from typing import List, Dict
import uuid

from langchain_core.messages import BaseMessage, ToolCall  # + ToolCall opcional

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
    ReasoningEffort
)

class RenderHarmony:
    def __init__(self):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def detect_harmony_tool_calls(self, response: str) -> List[Dict]:
        """
        Detecta tool calls no formato Harmony:

        <|start|>assistant<|channel|>commentary to=functions.function_name <|constrain|>json
        <|message|>{"param": "value"}<|call|>
        """
        tool_calls: List[Dict] = []
        pattern = (
            r'<\|start\|>assistant<\|channel\|>commentary\s+to=functions\.([^<\s]+)'
            r'(?:\s+<\|constrain\|>(\w+))?<\|message\|>(.*?)<\|call\|>'
        )
        for m in re.finditer(pattern, response, re.DOTALL):
            fn = m.group(1)
            constraint = (m.group(2) or "json").lower()
            content = m.group(3).strip()
            args = {}
            if constraint == "json":
                try:
                    args = json.loads(content)
                except json.JSONDecodeError:
                    j = re.search(r'\{.*\}', content, re.DOTALL)
                    if j:
                        try:
                            args = json.loads(j.group())
                        except json.JSONDecodeError:
                            args = {}
            else:
                args = {"content": content}
            tool_calls.append({
                "name": fn,
                "args": args,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            })
        return tool_calls

    def to_lc_tool_calls(self, harmony_calls: List[Dict]) -> List[ToolCall]:
        """Converte dicts -> ToolCall (LangChain)."""
        return [
            ToolCall(
                id=c["id"],
                name=c["name"],
                args=c.get("args", {}),
                type=c.get("type", "tool_call"),
            )
            for c in harmony_calls
        ]

    def detect_harmony_analysis_messages(self, response: str) -> List[str]:
        """Extrai mensagens de 'analysis' (não exibir ao usuário)."""
        msgs: List[str] = []
        pattern = r'<\|channel\|>analysis<\|message\|>(.*?)(?=<\|\w+\|>|<\|end\|>|\Z)'
        for m in re.finditer(pattern, response, re.DOTALL):
            msgs.append(m.group(1).strip())
        return msgs

    def detect_harmony_final_messages(self, response: str) -> List[str]:
        """Extrai mensagens do canal 'final' (mostrar ao usuário)."""
        msgs: List[str] = []
        pattern = r'<\|channel\|>final<\|message\|>.(.*?)(?=<\|\w+\|>|<\|end\|>|\Z)'
        for m in re.finditer(pattern, response, re.DOTALL):
            msgs.append(m.group(1).strip())
        return msgs

    def detect_harmony_commentary_messages(self, response: str) -> List[str]:
        """Extrai 'commentary' (exceto tool calls)."""
        msgs: List[str] = []
        pattern = r'<\|channel\|>commentary(?!\s+to=functions\.)<\|message\|>(.*?)(?=<\|\w+\|>|<\|end\|>|\Z)'
        for m in re.finditer(pattern, response, re.DOTALL):
            msgs.append(m.group(1).strip())
        return msgs

    def render_harmony_conversation(
        self,
        messages: List[BaseMessage],
        bound_tools: List[Dict],
        ReasoningEffort: ReasoningEffort = ReasoningEffort.LOW,
    ):
        """
        Renderiza a conversa Harmony e retorna TOKENS (para MLX).
        """
        harmony_msgs = []
        functions_definition = self.convert_tools_to_harmony_format(bound_tools)

        # SYSTEM com controle de "esforço de raciocínio"
        system_msg = Message.from_role_and_content(
            Role.SYSTEM, SystemContent.new().with_reasoning_effort(ReasoningEffort)
        )
        harmony_msgs.append(system_msg)

        for m in messages:
            content = m.content
            if m.type == "system":
                dev = Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new()
                    .with_instructions(content)
                    .with_function_tools(functions_definition)
                )
                harmony_msgs.append(dev)
            elif m.type == "human":
                harmony_msgs.append(Message.from_role_and_content(Role.USER, content))
            elif m.type == "ai":
                harmony_msgs.append(
                    Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")
                )
            elif m.type == "tool":
                # Para exibição/prompting. No LangChain, a resposta real da tool vira ToolMessage.
                harmony_msgs.append(
                    Message.from_author_and_content(Author.new(Role.TOOL, getattr(m, "name", "tool")), content)
                    .with_channel("commentary")
                )

        conversation = Conversation.from_messages(harmony_msgs)
        tokens = self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)
        return tokens

    def convert_tools_to_harmony_format(self, bound_tools: List[Dict]):
        """
        Converte bound_tools (OpenAI tool schema) para ToolDescription (Harmony).
        """
        from openai_harmony import ToolDescription
        harmony_tools = []
        if isinstance(bound_tools, dict):
            if "function" in bound_tools:
                f = bound_tools["function"]
                harmony_tools.append(ToolDescription.new(f.get("name", "unknown_function"),
                                                        f.get("description", f"Function {f.get('name', '')}"),
                                                        parameters=f.get("parameters", {})))
            else:
                for name, info in bound_tools.items():
                    harmony_tools.append(ToolDescription.new(name,
                                                             info.get("description", f"Function {name}"),
                                                             parameters=info.get("parameters", {})))
        elif isinstance(bound_tools, list):
            for item in bound_tools:
                if isinstance(item, dict) and "function" in item:
                    f = item["function"]
                    harmony_tools.append(ToolDescription.new(f.get("name", "unknown_function"),
                                                             f.get("description", f"Function {f.get('name','')}"),
                                                             parameters=f.get("parameters", {})))
        return harmony_tools