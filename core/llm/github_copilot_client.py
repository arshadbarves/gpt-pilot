import httpx
import requests
from core.config import LLMProvider
import json
from typing import Optional
from core.llm.base import BaseLLMClient
from core.llm.convo import Convo
from core.log import get_logger

log = get_logger(__name__)

class GitHubCopilotClient(BaseLLMClient):
    provider = LLMProvider.GITHUB_COPILOT

    def _init_client(self):
        # First get the access token
        auth_response = requests.get(
            "https://api.github.com/copilot_internal/v2/token",
            headers={"Authorization": f"token {self.config.api_key}"}
        )
        if auth_response.status_code != 200:
            raise Exception(f"Failed to get Copilot token: {auth_response.text}")
            
        token = auth_response.json()["token"]
        
        # Initialize AsyncClient with correct base URL and headers
        self.client = httpx.AsyncClient(
            base_url="https://api.githubcopilot.com",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "OpenAI-Organization": "github-copilot",
                "Copilot-Integration-Id": "vscode-chat"
            }
        )

    async def _make_request(
        self,
        convo: Convo,
        temperature: Optional[float] = None,
        json_mode: bool = False,
    ) -> tuple[str, int, int]:
        payload = {
            "model": self.config.model,
            "messages": self._adapt_messages_to_github_copilot(convo),
            "stream": True,
        }
    
        response = []
        prompt_tokens = 0
        completion_tokens = 0
    
        async with self.client.stream("POST", "/chat/completions", json=payload) as r:
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    data = line[len("data: "):]
                    if data == "[DONE]":
                        break
                    else:
                        try:
                            chunk = json.loads(data)
                            print("chunk", chunk)
                            if 'usage' in chunk:
                                prompt_tokens += chunk['usage'].get('prompt_tokens', 0)
                                completion_tokens += chunk['usage'].get('completion_tokens', 0)
    
                            if not chunk.get('choices'):
                                continue
    
                            content = chunk['choices'][0]['delta'].get('content')
                            if not content:
                                continue
    
                            response.append(content)
                            if self.stream_handler:
                                await self.stream_handler(content)
                        except json.JSONDecodeError as e:
                            print(f"Failed to decode JSON: {e}")
    
            response_str = "".join(response)
            return response_str, prompt_tokens, completion_tokens

    def _adapt_messages_to_github_copilot(self, convo):
        adapted_messages = []
        for msg in convo:
            role = "user" if msg["role"] in ["user", "system"] else "assistant"
            adapted_messages.append({
                "role": role,
                "content": msg["content"]
            })
        
        return adapted_messages
