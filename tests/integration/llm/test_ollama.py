from json import loads
from os import getenv

import pytest

from core.config import LLMConfig, LLMProvider
from core.llm.base import APIError
from core.llm.convo import Convo
from core.llm.ollama_client import OllamaClient

run_integration_tests = getenv("INTEGRATION_TESTS", "").lower()
if run_integration_tests not in ["true", "yes", "1", "on"]:
    pytest.skip("Skipping integration tests", allow_module_level=True)


@pytest.mark.asyncio
async def test_unknown_model():
    cfg = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama-3.6-nonexistent",
        temperature=0.5,
    )

    llm = OllamaClient(cfg)
    convo = Convo("you're a friendly assistant").user("tell me joke")

    with pytest.raises(APIError, match="does not exist"):
        await llm(convo)


@pytest.mark.asyncio
async def test_ollama_success():
    cfg = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama-3.2",
        temperature=0.5,
    )

    streamed_response = []

    async def stream_handler(content: str):
        if content:
            streamed_response.append(content)

    llm = OllamaClient(cfg, stream_handler=stream_handler)
    convo = Convo("you're a friendly assistant").user("tell me joke")

    response, req_log = await llm(convo)
    assert response == "".join(streamed_response)

    assert req_log.messages == convo.messages
    assert req_log.prompt_tokens > 0
    assert req_log.completion_tokens > 0


@pytest.mark.asyncio
async def test_ollama_json_mode():
    cfg = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama-3.2",
        temperature=0.5,
    )

    llm = OllamaClient(cfg)
    convo = Convo("you're a friendly assistant")
    convo.user('tell me a q/a joke. output it in a JSON format like: {"q": "...", "a": "..."}')

    response, req_log = await llm(convo)

    data = loads(response)
    assert "q" in data
    assert "a" in data


@pytest.mark.asyncio
async def test_context_too_large():
    cfg = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model="llama-3.2",
        temperature=0.5,
    )

    streamed_response = []

    async def stream_handler(content: str):
        if content:
            streamed_response.append(content)

    llm = OllamaClient(cfg, stream_handler=stream_handler)
    convo = Convo("you're a friendly assistant")
    large_convo = " ".join(["lorem ipsum dolor sit amet"] * 30000)
    convo.user(large_convo)
    with pytest.raises(APIError, match="We sent too large request to the LLM"):
        await llm(convo)
