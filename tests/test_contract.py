"""The dataset exactly as the AgriEvidence backend sends it (captured from the backend's sample data)."""

import asyncio
import json

from agritrial_llm.agent import run_agent
from agritrial_llm.schemas import AgentOptions, ChatRequest, Dataset
from tests.conftest import FIXTURES


def test_backend_payload_is_understood(settings):
    dataset = Dataset.model_validate(json.loads((FIXTURES / "backend_dataset.json").read_text(encoding="utf-8")))
    assert [trial.id for trial in dataset.trials] == ["T01", "T02", "T03", "T04", "T05", "T06"]
    assert dataset.trials[2].conflicts[0].candidates[1].display == "44 t/ha"

    request = ChatRequest(question="Which trials have conflicting data?", dataset=dataset, options=AgentOptions(force_offline=True))
    response = asyncio.run(run_agent(request, settings))
    assert "42 t/ha (trials_a.csv) vs 44 t/ha (T03_summary.txt)" in response.answer
    assert response.verification.passed
