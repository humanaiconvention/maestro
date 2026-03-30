import os
import time
import json
import logging
import uuid
from typing import AsyncGenerator, Any
import anthropic
from anthropic import AsyncAnthropic, APIError
from fastapi import HTTPException

from .base import BaseAdapter
from libs.schemas.api import ChatCompletionRequest, ChatCompletionResponse, ChatChoice, ChatMessage, Usage, MaestroMetadata

logger = logging.getLogger(__name__)

class AnthropicAdapter(BaseAdapter):
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

        self.client = AsyncAnthropic(api_key=self.api_key)

        # Model mapping
        self.model_mapping = {
            "maestro-default": "claude-3-5-haiku-20241022",
            "gpt-4": "claude-sonnet-4-20250514",
            "gpt-3.5-turbo": "claude-3-5-haiku-20241022"
        }

    def prepare(self, request_envelope: ChatCompletionRequest) -> dict:
        """
        Extracts messages, model alias, and temperature from the request.
        Maps model aliases to Anthropic model IDs.
        """
        model = self.model_mapping.get(request_envelope.model, request_envelope.model)

        # Anthropic expects 'user' and 'assistant' roles.
        # OpenAI 'system' messages are passed separately in Anthropic.
        system_prompt = ""
        messages = []
        for msg in request_envelope.messages:
            if msg.role == "system":
                system_prompt += msg.content + "\n"
            else:
                messages.append({"role": msg.role, "content": msg.content})

        prepared = {
            "model": model,
            "messages": messages,
            "max_tokens": 4096, # Default max tokens
        }

        if system_prompt.strip():
            prepared["system"] = system_prompt.strip()

        if request_envelope.temperature is not None:
            prepared["temperature"] = request_envelope.temperature

        return prepared

    async def invoke(self, prepared_request: dict) -> ChatCompletionResponse:
        try:
            response = await self.client.messages.create(**prepared_request)
            return self.normalize_response(response, prepared_request["model"])
        except APIError as e:
            raise self.normalize_error(e)

    async def stream(self, prepared_request: dict) -> AsyncGenerator[str, None]:
        try:
            async with self.client.messages.stream(**prepared_request) as stream:
                async for event in stream:
                    if event.type == "content_block_delta":
                        # Yield OpenAI-compatible SSE chunk
                        chunk = {
                            "id": f"ant_{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": prepared_request["model"],
                            "choices": [{
                                "index": 0,
                                "delta": {"content": event.delta.text},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                    elif event.type == "message_stop":
                        # End of stream chunk
                        final_chunk = {
                            "id": f"ant_{uuid.uuid4().hex}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": prepared_request["model"],
                            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                        }
                        yield f"data: {json.dumps(final_chunk)}\n\n"

            yield "data: [DONE]\n\n"
        except APIError as e:
            # SSE error format (simplified)
            error_msg = json.dumps({"error": {"message": str(e), "type": "anthropic_error"}})
            yield f"data: {error_msg}\n\n"
            yield "data: [DONE]\n\n"

    def normalize_response(self, raw: Any, model_id: str) -> ChatCompletionResponse:
        """Converts Anthropic Message to ChatCompletionResponse."""
        content = ""
        if raw.content and len(raw.content) > 0:
            content = raw.content[0].text

        return ChatCompletionResponse(
            id=raw.id,
            created=int(time.time()),
            model=model_id,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop" # Simplified
                )
            ],
            usage=Usage(
                prompt_tokens=raw.usage.input_tokens,
                completion_tokens=raw.usage.output_tokens,
                total_tokens=raw.usage.input_tokens + raw.usage.output_tokens
            ),
            maestro=MaestroMetadata(
                cache="miss",
                route="tier1",
                request_id=raw.id
            )
        )

    def normalize_error(self, raw: APIError) -> HTTPException:
        """Converts Anthropic APIError to HTTPException."""
        status_code = getattr(raw, "status_code", 500)
        message = str(raw)

        # Map specific codes to canonical ones if needed
        if status_code == 401:
            detail = "unauthorized"
        elif status_code == 429:
            detail = "rate_limited"
        elif status_code >= 500:
            detail = "provider_unavailable"
        else:
            detail = "invalid_request"

        return HTTPException(status_code=status_code, detail=detail)
