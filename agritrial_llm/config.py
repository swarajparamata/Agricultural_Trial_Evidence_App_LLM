"""Service configuration, read from environment variables or a `.env` file (see `.env.example`)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVICE_DIR = Path(__file__).resolve().parent.parent


class ServiceSettings(BaseSettings):
    """Defaults for the service. Callers can override the agent options per request."""

    # The repo's .env, whatever the working directory.
    model_config = SettingsConfigDict(env_file=SERVICE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    llm_service_host: str = "127.0.0.1"
    llm_service_port: int = 8100
    # When set, every /v1 request must send it in the `X-API-Key` header.
    llm_service_api_key: str = ""

    # 127.0.0.1 rather than localhost: on Windows, localhost tries IPv6 first and doubles the wait when Ollama is down.
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.6"
    ollama_timeout_s: float = 180.0
    ollama_num_ctx: int = 16384
    ollama_keep_alive: str = "10m"
    llm_temperature: float = 0.1
    # Qwen 3.6 is a thinking model; thinking improves hard questions but is much slower on CPU.
    llm_think: bool = False

    agent_max_tool_iterations: int = 6
    agent_max_revisions: int = 2
    agent_enable_reflection: bool = True
    # Answer with the rule-based planner when Ollama or the model is unavailable (keeps demos working).
    agent_offline_fallback: bool = True
    extraction_max_attempts: int = 3


@lru_cache
def get_settings() -> ServiceSettings:
    return ServiceSettings()
