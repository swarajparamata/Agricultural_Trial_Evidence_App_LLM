"""Ollama access: per-request options, the model availability probe and the LangChain chat model."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import httpx
from langchain_ollama import ChatOllama

from .config import ServiceSettings
from .schemas import AgentOptions


@dataclass(frozen=True)
class ResolvedOptions:
    """Request options merged over the service defaults."""

    model: str
    base_url: str
    temperature: float
    num_ctx: int
    think: bool
    keep_alive: str
    timeout_s: float
    max_tool_iterations: int
    max_revisions: int
    enable_reflection: bool
    offline_fallback: bool
    force_offline: bool
    history_turns: int
    extra_instructions: str


def resolve_options(options: AgentOptions, settings: ServiceSettings) -> ResolvedOptions:
    def pick(value, default):
        return default if value is None else value

    return ResolvedOptions(
        model=(options.model or settings.ollama_model).strip(),
        base_url=(options.ollama_base_url or settings.ollama_base_url).strip().rstrip("/"),
        temperature=pick(options.temperature, settings.llm_temperature),
        num_ctx=pick(options.num_ctx, settings.ollama_num_ctx),
        think=pick(options.think, settings.llm_think),
        keep_alive=settings.ollama_keep_alive,
        timeout_s=settings.ollama_timeout_s,
        max_tool_iterations=pick(options.max_tool_iterations, settings.agent_max_tool_iterations),
        max_revisions=pick(options.max_revisions, settings.agent_max_revisions),
        enable_reflection=pick(options.enable_reflection, settings.agent_enable_reflection),
        offline_fallback=pick(options.offline_fallback, settings.agent_offline_fallback),
        force_offline=options.force_offline,
        history_turns=options.history_turns,
        extra_instructions=options.extra_instructions.strip(),
    )


@dataclass
class OllamaStatus:
    base_url: str
    model: str
    reachable: bool
    model_available: bool
    available_models: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.reachable and self.model_available

    @property
    def reason(self) -> str:
        if not self.reachable:
            return f"Ollama is not reachable at {self.base_url}"
        if not self.model_available:
            return f"model '{self.model}' is not installed in Ollama (run: ollama pull {self.model})"
        return "ready"


_PROBE_CACHE: dict[tuple[str, str], tuple[float, OllamaStatus]] = {}
# A refused connection takes ~2 s per address on Windows, so failures are remembered longer than successes.
_READY_TTL_S = 15.0
_FAILED_TTL_S = 30.0
_PROBE_TIMEOUT = httpx.Timeout(5.0, connect=1.0)


def model_installed(model: str, installed: list[str]) -> bool:
    """'qwen3.6' matches 'qwen3.6:latest'; a tagged name must match exactly."""
    wanted = {model, model if ":" in model else f"{model}:latest"}
    return any(name in wanted for name in installed)


async def probe_ollama(base_url: str, model: str, *, max_age: float | None = None) -> OllamaStatus:
    """Checks that Ollama answers and has the model. Results are cached so chats don't pay for the check;
    pass `max_age=0` to force a fresh check."""
    key = (base_url, model)
    now = time.monotonic()
    if cached := _PROBE_CACHE.get(key):
        stamp, status = cached
        ttl = max_age if max_age is not None else (_READY_TTL_S if status.ready else _FAILED_TTL_S)
        if now - stamp < ttl:
            return status

    try:
        async with httpx.AsyncClient(timeout=_PROBE_TIMEOUT) as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
            names = [entry.get("name") or entry.get("model") or "" for entry in response.json().get("models", [])]
        status = OllamaStatus(base_url, model, True, model_installed(model, names), sorted(names))
    except (httpx.HTTPError, ValueError) as exc:
        status = OllamaStatus(base_url, model, False, False, error=describe_error(exc))
    _PROBE_CACHE[key] = (now, status)
    return status


def clear_probe_cache() -> None:
    _PROBE_CACHE.clear()


def build_chat_model(options: ResolvedOptions) -> ChatOllama:
    return ChatOllama(
        model=options.model,
        base_url=options.base_url,
        temperature=options.temperature,
        num_ctx=options.num_ctx,
        # False switches Qwen's thinking off; True returns the reasoning separately from the answer.
        reasoning=options.think,
        keep_alive=options.keep_alive,
        client_kwargs={"timeout": options.timeout_s},
    )


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Removes inline <think> blocks that some model templates leave in the answer."""
    return _THINK_BLOCK.sub("", text).strip()


def message_text(content: object) -> str:
    """Text of a message whose content may be a string or a list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


def describe_error(exc: BaseException) -> str:
    detail = str(exc).strip()
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__
