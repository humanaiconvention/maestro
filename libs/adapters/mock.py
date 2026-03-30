import asyncio
import time
import json
from .base import BaseAdapter
from libs.schemas.api import ChatCompletionResponse, ChatChoice, ChatMessage, Usage, MaestroMetadata

class MockAdapter(BaseAdapter):
    def prepare(self, request_envelope):
        return request_envelope

    async def invoke(self, prepared_request):
        # Simulate some latency
        await asyncio.sleep(0.5)
        
        return ChatCompletionResponse(
            id="mock_req_123",
            created=int(time.time()),
            model=prepared_request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="This is a mock response from Maestro Tier 1."),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
            maestro=MaestroMetadata(
                cache="miss",
                route="tier1",
                request_id="mock_req_123"
            )
        )

    async def stream(self, prepared_request):
        words = ["This", " is", " a", " mock", " streaming", " response", " from", " Maestro."]
        for i, word in enumerate(words):
            await asyncio.sleep(0.1)
            # OpenAI SSE chunk format (simplified)
            chunk = {
                "id": "mock_req_123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": prepared_request.model,
                "choices": [{"index": 0, "delta": {"content": word}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # End stream
        final_chunk = {
            "id": "mock_req_123",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": prepared_request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    def normalize_response(self, raw_response):
        return raw_response

    def normalize_error(self, raw_error):
        return raw_error
