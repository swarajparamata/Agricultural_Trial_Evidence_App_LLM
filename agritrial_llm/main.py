"""HTTP API of the LLM service. The AgriEvidence backend is its only intended client."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage

from . import __version__
from .agent import agent_graph_mermaid, run_agent, stream_agent
from .config import ServiceSettings, get_settings
from .extraction import extraction_graph_mermaid, run_extraction
from .llm import build_chat_model, describe_error, message_text, probe_ollama, resolve_options
from .schemas import (
    AgentOptions,
    ChatRequest,
    ChatResponse,
    ExtractRequest,
    ExtractResponse,
    LlmTestRequest,
    OllamaStatusOut,
)

app = FastAPI(
    title="AgriEvidence LLM Service",
    version=__version__,
    description=(
        "LangGraph agent (LangChain + Ollama/Qwen 3.6) that answers questions about agricultural trial evidence, "
        "plus an extraction loop for free-text trial reports. Falls back to a rule-based planner when the model "
        "is unavailable."
    ),
)


def require_api_key(
    x_api_key: str | None = Header(default=None), settings: ServiceSettings = Depends(get_settings)
) -> None:
    if settings.llm_service_api_key and x_api_key != settings.llm_service_api_key:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key header.")


@app.get("/health", tags=["service"])
async def health(
    model: str | None = None, ollama_base_url: str | None = None, settings: ServiceSettings = Depends(get_settings)
) -> dict:
    """Service status, plus whether Ollama has `model` (default: the configured model) at `ollama_base_url`."""
    options = resolve_options(AgentOptions(model=model, ollama_base_url=ollama_base_url), settings)
    ollama = await probe_ollama(options.base_url, options.model)
    return {
        "status": "ok",
        "service": "agritrial-llm",
        "version": __version__,
        "ollama": OllamaStatusOut(**ollama.__dict__).model_dump(),
        "defaults": {
            "model": settings.ollama_model,
            "max_tool_iterations": settings.agent_max_tool_iterations,
            "max_revisions": settings.agent_max_revisions,
            "enable_reflection": settings.agent_enable_reflection,
            "offline_fallback": settings.agent_offline_fallback,
            "think": settings.llm_think,
        },
    }


@app.get("/v1/agent/graph", tags=["agent"], dependencies=[Depends(require_api_key)])
async def agent_graph() -> dict:
    """Mermaid diagrams of the LangGraph loops, for documentation and the settings page."""
    return {"agent": agent_graph_mermaid(), "extraction": extraction_graph_mermaid()}


@app.post("/v1/agent/chat", tags=["agent"], response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(request: ChatRequest, settings: ServiceSettings = Depends(get_settings)) -> ChatResponse:
    try:
        return await run_agent(request, settings)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@app.post("/v1/agent/stream", tags=["agent"], dependencies=[Depends(require_api_key)])
async def chat_stream(request: ChatRequest, settings: ServiceSettings = Depends(get_settings)) -> StreamingResponse:
    """Server-sent events: `log` events while the agent works, then `answer` (or `error`) and `done`."""

    async def events() -> AsyncIterator[str]:
        async for event in stream_agent(request, settings):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post(
    "/v1/extract/report", tags=["extraction"], response_model=ExtractResponse, dependencies=[Depends(require_api_key)]
)
async def extract_report(request: ExtractRequest, settings: ServiceSettings = Depends(get_settings)) -> ExtractResponse:
    return await run_extraction(request, settings)


@app.post("/v1/llm/test", tags=["service"], dependencies=[Depends(require_api_key)])
async def test_llm(request: LlmTestRequest, settings: ServiceSettings = Depends(get_settings)) -> dict:
    """Sends a one-line prompt to the model and reports whether it answered and how long it took."""
    options = resolve_options(
        AgentOptions(model=request.model, ollama_base_url=request.ollama_base_url, think=False), settings
    )
    ollama = await probe_ollama(options.base_url, options.model, max_age=0)
    result = {
        "ok": False,
        "model": options.model,
        "base_url": options.base_url,
        "ollama": OllamaStatusOut(**ollama.__dict__).model_dump(),
        "latency_ms": None,
        "reply": None,
        "error": None if ollama.ready else ollama.reason,
    }
    if not ollama.ready:
        return result
    started = time.perf_counter()
    try:
        reply = await build_chat_model(options).ainvoke([HumanMessage(content="Reply with the single word: ready")])
        result.update(ok=True, reply=message_text(reply.content).strip()[:200])
    except Exception as exc:
        result["error"] = describe_error(exc)
    result["latency_ms"] = int((time.perf_counter() - started) * 1000)
    return result
