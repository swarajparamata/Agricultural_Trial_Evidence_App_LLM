import json
import time
from pathlib import Path

import pytest

from agritrial_llm import llm
from agritrial_llm.config import ServiceSettings
from agritrial_llm.schemas import Dataset
from agritrial_llm.store import EvidenceStore

FIXTURES = Path(__file__).parent / "fixtures"
OLLAMA_DOWN_URL = "http://127.0.0.1:9"


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.model_validate(json.loads((FIXTURES / "sample_dataset.json").read_text(encoding="utf-8")))


@pytest.fixture
def backend_dataset() -> Dataset:
    """What the backend sends for the sample data, including its full default vocabulary (e.g. Rice)."""
    return Dataset.model_validate(json.loads((FIXTURES / "backend_dataset.json").read_text(encoding="utf-8")))


@pytest.fixture
def store(dataset: Dataset) -> EvidenceStore:
    return EvidenceStore(dataset)


@pytest.fixture
def settings() -> ServiceSettings:
    # Nothing listens on port 9, so the agent falls back to the offline planner.
    return ServiceSettings(_env_file=None, ollama_base_url=OLLAMA_DOWN_URL, llm_service_api_key="")


@pytest.fixture(autouse=True)
def ollama_is_down(settings: ServiceSettings):
    """Records a failed probe up front, so tests don't each wait for a refused connection."""
    llm.clear_probe_cache()
    status = llm.OllamaStatus(
        base_url=OLLAMA_DOWN_URL, model=settings.ollama_model, reachable=False, model_available=False,
        error="ConnectError: connection refused",
    )
    llm._PROBE_CACHE[(OLLAMA_DOWN_URL, settings.ollama_model)] = (time.monotonic(), status)
    yield
    llm.clear_probe_cache()
