"""One chat turn: generate, run any tool calls, repeat until the model answers."""
from __future__ import annotations

import json
import re

# Qwen3 emits tool calls as <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> list[dict]:
    calls = []
    for raw in _TOOL_CALL.findall(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            args = obj.get("arguments") or {}
            calls.append({"name": obj["name"], "arguments": args if isinstance(args, dict) else {}})
    return calls


def run_turn(backend, messages: list[dict], toolbox=None, max_rounds: int = 4):
    """Return (reply_text, trace). `messages` is extended in place with tool traffic."""
    tools = toolbox.specs if toolbox else None
    trace: list[dict] = []
    for _ in range(max_rounds):
        text = backend.generate(messages, tools=tools)
        calls = parse_tool_calls(text) if toolbox else []
        if not calls:
            return text.strip(), trace
        messages.append({"role": "assistant", "content": text})
        for call in calls:
            result = toolbox.call(call["name"], call["arguments"])
            trace.append({**call, "result_chars": len(result)})
            messages.append({"role": "tool", "name": call["name"], "content": result})
    # Out of rounds: force a final answer with the tools detached.
    return backend.generate(messages, tools=None).strip(), trace
