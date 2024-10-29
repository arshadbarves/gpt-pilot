import datetime
from lib2to3.pgen2.tokenize import generate_tokens
from typing import Optional

import httpx
from core.llm.base import BaseLLMClient, APIError
from core.llm.request_log import LLMRequestLog, LLMRequestStatus
import google.generativeai as genai

from core.config import LLMProvider
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)


class GoogleClient(BaseLLMClient):
    provider = LLMProvider.GOOGLE
    stream_options = None

    def _init_client(self):
        genai.configure(
            api_key=self.config.api_key,
            # client_options={
            #     "timeout": httpx.Timeout(
            #         max(self.config.connect_timeout, self.config.read_timeout),
            #         connect=self.config.connect_timeout,
            #         read=self.config.read_timeout,
            #     ),
            # },
        )

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:

        # https://github.com/google-gemini/cookbook/blob/main/quickstarts/
        generation_config = genai.GenerationConfig(
            temperature=self.config.temperature if temperature is None else temperature,
            top_p=1.0,
            top_k=40,
        )

        if json_mode:
            generation_config.response_mime_type = "application/json"

        self.client = genai.GenerativeModel(
            model_name= self.config.model,
            generation_config=genai.GenerationConfig(
                temperature=self.config.temperature if temperature is None else temperature,
                top_p=1.0,
                top_k=40,
            ),
        )

        # Process the conversation from Convo to Google's format {'role':'user', 'parts': ['How does quantum physics work?']}
        convo.messages = self._adapt_messages_to_google(convo)

        stream = await self.client.generate_content_async(contents=convo.messages, stream=True)
        response = []
        prompt_tokens = 0
        completion_tokens = 0

        async for chunk in stream:
            if chunk.text:
                prompt_tokens += chunk.usage_metadata.prompt_token_count
                completion_tokens += chunk.usage_metadata.candidates_token_count

                response.append(chunk.text)
                if self.stream_handler:
                    await self.stream_handler(chunk.text)

        response_str = "".join(response)

        # Tell the stream handler we're done
        if self.stream_handler:
            await self.stream_handler(None)

        return response_str, prompt_tokens, completion_tokens

    def rate_limit_sleep(self, err: APIError) -> Optional[datetime.timedelta]:
        """
        Google rate limits docs:
        https://cloud.google.com/ai-platform/training/docs/quota
        """
        return None

    def _adapt_messages_to_google(self, convo: Convo) -> list[dict]:
        """
        Adapt the conversation messages to Google's format.

        :param convo: Conversation object.
        :return: List of messages in Google's format.
        """
        role_mapping = {
            "system": "model",  # Replace 'system' with 'teacher'
            "user": "user",
            "assistant": "model",
            "function": "model"  # Replace 'function' with 'critic'
        }

        messages = []
        for msg in convo.messages:
            role = role_mapping.get(msg.get("role", "assistant"),
                                    "assistant")  # Default to 'assistant' if role not found
            content = msg.get("content", "")
            if content:
                parts = content.split("\n")
                for part in parts:
                    if part.strip():  # Ensure the part is not empty
                        messages.append({"role": role, "parts": [part]})
        return messages


__all__ = ["GoogleClient"]