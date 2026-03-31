"""
llama_cpp_adapter.py — LlamaCppAdapter for the HAIC grounding interviewer.

Calls a running llama.cpp server (OpenAI-compatible API) so the gateway can use
the fine-tuned Qwen GGUF model as a drop-in replacement for the MockAdapter.

Configuration (env vars):
    LLAMACPP_BASE_URL          Base URL of the llama.cpp server (required, e.g. http://localhost:<port>)
    LLAMACPP_MODEL             Model name string sent in API requests (required)
    LLAMACPP_MAX_TOKENS        Max tokens per response (default: 1500 — covers Qwen3 <think> block + answer)
    LLAMACPP_TEMPERATURE       Sampling temperature (default: 0.4)
    LLAMACPP_REPEAT_PENALTY    Repeat penalty to break Qwen3 thinking loops (default: 1.3)
    LLAMACPP_SYSTEM_PROMPT_PATH  Path to system prompt file (default: auto-detects sgt/prompts/base_interviewer.txt)
    LLAMACPP_TIMEOUT           HTTP timeout in seconds (default: 60)

The adapter:
- Injects the SGT grounding system prompt, overriding any client-supplied system message
- Detects session-ending signals and returns a graceful close without a model call
- Proxies SSE streaming chunks directly from the llama.cpp server
- Falls back to an embedded prompt if the SGT file is not found
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import httpx

from libs.adapters.base import BaseAdapter
from libs.schemas.api import (
    ChatCompletionResponse,
    ChatChoice,
    ChatMessage,
    Usage,
    MaestroMetadata,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Fallback system prompt (used if SGT file is not found)
# ──────────────────────────────────────────────────────────────────────────────

_FALLBACK_SYSTEM_PROMPT = (
    "You are an interviewer in the Human-AI Convention Grounding Protocol. "
    "You have exactly 4 turns. Ask one question per turn. "
    "Turn 1: ask about one specific recent moment. "
    "Turn 2: choose a pivot (adversarial/counterfactual/shadow/relational/temporal/sensory) "
    "that moves 180 degrees from how they answered. Emit [PIVOT: type] before your question. "
    "Turn 3: zoom into physical, embodied, or sequential texture. "
    "Turn 4: ask for one irreducible image or detail that holds the whole experience. "
    "After turn 4: one sentence of thanks, then stop. "
    "No interpretation, no advice, no affirmations. Mirror their exact words."
)

# Terminal signals — participant is ending the session
_TERMINAL_SIGNALS = frozenset([
    "that is all", "that's all", "thats all", "i'm done", "im done",
    "i am done", "nothing more", "nothing else", "that's it", "thats it",
    "that is it", "done", "finished", "end", "stop", "goodbye", "bye",
    "no more", "i think that covers it", "i think thats all",
])

_TERMINAL_REPLY = "Thank you for sharing that. Your responses have been recorded."


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _find_sgt_prompt() -> Path | None:
    """Walk up from this file to find sgt/prompts/base_interviewer.txt."""
    # adapter is at maestro/libs/adapters/ — repo root is 3 levels up
    here = Path(__file__).resolve()
    for parent in [here.parent.parent.parent, here.parent.parent.parent.parent]:
        candidate = parent / "sgt" / "prompts" / "base_interviewer.txt"
        if candidate.exists():
            return candidate
    return None


def _load_system_prompt() -> str:
    """Load the SGT base interviewer system prompt, falling back to embedded."""
    # Allow explicit override
    env_path = os.environ.get("LLAMACPP_SYSTEM_PROMPT_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists():
            text = p.read_text(encoding="utf-8").strip()
            logger.info(f"LlamaCppAdapter: loaded system prompt from {p}")
            return text
        else:
            logger.warning(f"LLAMACPP_SYSTEM_PROMPT_PATH={env_path} not found; using auto-detect")

    # Auto-detect sgt/prompts/base_interviewer.txt
    found = _find_sgt_prompt()
    if found:
        text = found.read_text(encoding="utf-8").strip()
        logger.info(f"LlamaCppAdapter: loaded system prompt from {found}")
        return text

    logger.warning("LlamaCppAdapter: SGT prompt not found; using fallback system prompt")
    return _FALLBACK_SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class LlamaCppAdapter(BaseAdapter):
    """
    Routes inference to a running llama.cpp server via its OpenAI-compatible API.

    The server must be started separately — typically:
        llama-server -m <model.gguf> --port <LLAMACPP_PORT> -c 4096 -np 1 --chat-template chatml
    """

    def __init__(self) -> None:
        self.base_url = os.environ.get("LLAMACPP_BASE_URL", "").rstrip("/")
        self.model_name = os.environ.get("LLAMACPP_MODEL", "")
        # 2000 tokens: safe budget for Qwen3 <think> block (can be long) + answer
        self.max_tokens = int(os.environ.get("LLAMACPP_MAX_TOKENS", "2000"))
        # 0.4 temperature — lower than default to reduce drift while allowing variety
        self.temperature = float(os.environ.get("LLAMACPP_TEMPERATURE", "0.4"))
        # repeat_penalty=1.3 breaks the Qwen3 reasoning loop so </think> is reliably reached
        self.repeat_penalty = float(os.environ.get("LLAMACPP_REPEAT_PENALTY", "1.3"))
        self.timeout = float(os.environ.get("LLAMACPP_TIMEOUT", "60"))
        self.system_prompt = _load_system_prompt()

        logger.info(
            f"LlamaCppAdapter: base_url={self.base_url} model={self.model_name} "
            f"max_tokens={self.max_tokens} temperature={self.temperature} "
            f"repeat_penalty={self.repeat_penalty}"
        )

    # ── Response post-processing ──────────────────────────────────────────────

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """
        Remove Qwen3 <think>...</think> reasoning blocks from the output.

        The fine-tuned model may produce internal reasoning wrapped in <think> tags
        before emitting the final answer. We strip these so the participant only
        sees the question.
        """
        # Remove complete <think>…</think> blocks (possibly multi-line)
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return cleaned.strip()

    # ── Message construction ──────────────────────────────────────────────────

    def _build_messages(self, request_envelope) -> list[dict]:
        """
        Build the message list for the llama.cpp server.

        - Strips any client-supplied system message and injects the SGT prompt.
        - Preserves user/assistant turns in order.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in request_envelope.messages:
            if m.role == "system":
                continue  # replaced by our prompt
            messages.append({"role": m.role, "content": m.content})
        return messages

    def _is_terminal(self, request_envelope) -> bool:
        """Return True if the last user message is a session-ending signal."""
        user_msgs = [m for m in request_envelope.messages if m.role == "user"]
        if not user_msgs:
            return False
        last = user_msgs[-1].content.strip().rstrip(".,!?").lower()
        return last in _TERMINAL_SIGNALS

    # ── BaseAdapter interface ─────────────────────────────────────────────────

    def prepare(self, request_envelope):
        return request_envelope

    async def invoke(self, prepared_request) -> ChatCompletionResponse:
        if self._is_terminal(prepared_request):
            return self._terminal_response(prepared_request)

        messages = self._build_messages(prepared_request)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.ConnectError:
                raise RuntimeError(
                    f"Cannot reach llama.cpp server at {self.base_url}. "
                    "Is it running? See CLAUDE.md for the start command."
                )

        choice = data["choices"][0]
        text = self._strip_thinking(choice["message"]["content"])
        usage = data.get("usage", {})
        req_id = data.get("id", f"llamacpp_{uuid.uuid4().hex[:12]}")

        return ChatCompletionResponse(
            id=req_id,
            created=int(time.time()),
            model=self.model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            ],
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
            maestro=MaestroMetadata(
                cache="miss",
                route="llama_cpp",
                request_id=req_id,
            ),
        )

    async def stream(self, prepared_request) -> AsyncGenerator[str, None]:
        if self._is_terminal(prepared_request):
            async for chunk in self._terminal_stream(prepared_request):
                yield chunk
            return

        messages = self._build_messages(prepared_request)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repeat_penalty": self.repeat_penalty,
            "stream": True,
        }

        req_id = f"llamacpp_{uuid.uuid4().hex[:12]}"

        try:
            # Accumulate the full response text so we can strip <think> blocks
            # before streaming back to the client. Qwen3 thinking tokens can
            # span many chunks, so we cannot strip inline.
            full_text = ""
            model_name = self.model_name
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        if raw == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(raw)
                            delta = chunk_data["choices"][0].get("delta", {})
                            full_text += delta.get("content", "")
                            if chunk_data.get("model"):
                                model_name = chunk_data["model"]
                        except (json.JSONDecodeError, KeyError, IndexError):
                            pass

            # Strip thinking blocks then re-emit word by word
            clean_text = self._strip_thinking(full_text)
            words = clean_text.split(" ")
            for i, word in enumerate(words):
                chunk_text = word if i == 0 else " " + word
                chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.02)

            final = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            return

        except httpx.ConnectError:
            # Emit an error chunk the frontend can handle
            error_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": "[Model server unavailable. Is llama.cpp running?]"},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    # ── Terminal helpers ──────────────────────────────────────────────────────

    def _terminal_response(self, prepared_request) -> ChatCompletionResponse:
        req_id = f"llamacpp_{uuid.uuid4().hex[:12]}"
        return ChatCompletionResponse(
            id=req_id,
            created=int(time.time()),
            model=self.model_name,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=_TERMINAL_REPLY),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            maestro=MaestroMetadata(cache="miss", route="llama_cpp_terminal", request_id=req_id),
        )

    async def _terminal_stream(self, prepared_request) -> AsyncGenerator[str, None]:
        req_id = f"llamacpp_{uuid.uuid4().hex[:12]}"
        words = _TERMINAL_REPLY.split(" ")
        for i, word in enumerate(words):
            chunk_text = word if i == 0 else " " + word
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{"index": 0, "delta": {"content": chunk_text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.02)
        final = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    def normalize_response(self, raw_response):
        return raw_response

    def normalize_error(self, raw_error):
        return raw_error
