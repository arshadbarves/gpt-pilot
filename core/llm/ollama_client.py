import datetime
from typing import Optional

import httpx
from core.llm.base import BaseLLMClient, APIError
from core.llm.request_log import LLMRequestLog, LLMRequestStatus

from core.config import LLMProvider
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)


class OllamaClient(BaseLLMClient):
    provider = LLMProvider.OLLAMA
    stream_options = None

    def _init_client(self):
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=httpx.Timeout(
                max(self.config.connect_timeout, self.config.read_timeout),
                connect=self.config.connect_timeout,
                read=self.config.read_timeout,
            ),
        )

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        completion_kwargs = {
            "model": self.config.model,
            "messages": [{"role": msg["role"], "content": msg["content"]} for msg in convo.messages],
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": True,
        }
        if json_mode:
            completion_kwargs["format"] = "json"

        response = []
        prompt_tokens = 0
        completion_tokens = 0

        async with self.client.stream("POST", "/api/chat", json=completion_kwargs) as stream:
            async for chunk in stream.aiter_text():
                response.append(chunk)
                if self.stream_handler:
                    await self.stream_handler(chunk)

        response_str = "".join(response)

        # Tell the stream handler we're done
        if self.stream_handler:
            await self.stream_handler(None)

        # Estimate tokens if not provided
        if prompt_tokens == 0 and completion_tokens == 0:
            prompt_tokens = sum(len(msg["content"].split()) for msg in convo.messages)
            completion_tokens = len(response_str.split())

        return response_str, prompt_tokens, completion_tokens

    def rate_limit_sleep(self, err: httpx.HTTPStatusError) -> Optional[datetime.timedelta]:
        """
        Ollama rate limits docs:
        https://docs.ollama.com/rate-limits
        """
        headers = err.response.headers
        if "Retry-After" not in headers:
            return None

        retry_after = int(headers["Retry-After"])
        return datetime.timedelta(seconds=retry_after)


__all__ = ["OllamaClient"]
