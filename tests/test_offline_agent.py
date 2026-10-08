"""The full LangGraph agent with the offline planner (no Ollama needed)."""

import asyncio

import pytest

from agritrial_llm.agent import run_agent, stream_agent
from agritrial_llm.schemas import AgentOptions, ChatRequest, ChatTurn, Dataset


def ask(dataset: Dataset, settings, question: str, history: list[ChatTurn] | None = None, **options):
    request = ChatRequest(question=question, history=history or [], dataset=dataset, options=AgentOptions(**options))
    return asyncio.run(run_agent(request, settings))


def test_falls_back_to_offline_planner_when_ollama_is_down(dataset, settings):
    response = ask(dataset, settings, "What is the average yield uplift of Harvest Plus on wheat?")
    assert response.mode == "offline"
    assert response.model == "offline-planner"
    assert "not reachable" in response.fallback_reason
    assert "+6.3%" in response.answer
    assert "T05" in response.answer and "missing" in response.answer
    assert response.verification.passed, response.verification.issues
    assert response.tool_calls == 2
    assert response.trial_ids == ["T01", "T02", "T05", "T06"]
    assert "T06_archive_note.txt" in response.sources
    messages = [event.message for event in response.trace]
    assert any(message.startswith("[Tool] trial_statistics(") for message in messages)
    assert any(message.startswith("[Verify] Grounding checks passed") for message in messages)
    assert messages[-1] == "[Agent] Response generated successfully."


def test_conflicts_question(dataset, settings):
    response = ask(dataset, settings, "Which trials have conflicting data?")
    assert "T03" in response.answer
    assert "42 t/ha (trials_a.csv)" in response.answer and "44 t/ha (T03_summary.txt)" in response.answer
    assert "+5.0% to +10.0%" in response.answer
    assert "T05" in response.answer  # missing control is listed as a data gap
    assert response.verification.passed, response.verification.issues


def test_compare_two_trials(dataset, settings):
    response = ask(dataset, settings, "Compare T01 and T04")
    assert response.answer.startswith("**Comparing T01 and T04**")
    assert "Different crops" in response.answer
    assert response.verification.passed, response.verification.issues


def test_tied_trials_share_a_place(dataset, settings):
    response = ask(dataset, settings, "Compare T01, T03 and T04")
    assert "Ranking by uplift: T01 and T03 (+5.0% each), then T04 (0.0%)." in response.answer
    assert response.verification.passed, response.verification.issues

    response = ask(dataset, settings, "What is the average yield uplift in 2023?")
    assert "across 2 trials with a control yield (+5.0% in each)" in response.answer
    assert response.verification.passed, response.verification.issues


def test_trial_details(dataset, settings):
    response = ask(dataset, settings, "Tell me about T03")
    assert response.answer.startswith("**T03 – Root Boost on Potato (Spain, 2023, Demonstration)**")
    assert "Sources conflict" in response.answer
    assert "Neither source indicates which result is final" in response.answer
    assert response.verification.passed, response.verification.issues


def test_best_product_ranking(dataset, settings):
    response = ask(dataset, settings, "Which product performs best?")
    assert response.answer.startswith("**Harvest Plus** has the highest mean yield uplift: **+6.3%**")
    assert "Root Boost (T03 and T04): +2.5%" in response.answer
    assert response.verification.passed, response.verification.issues


def test_follow_up_question_keeps_the_subject(dataset, settings):
    history = [
        ChatTurn(role="user", content="What is the average uplift of Harvest Plus on wheat?"),
        ChatTurn(role="assistant", content="(previous answer)"),
    ]
    response = ask(dataset, settings, "What about in Germany?", history)
    assert "Harvest Plus on Wheat in Germany" in response.answer
    assert "T02" in response.answer and "T05" in response.answer
    assert "T01" not in response.answer


def test_unknown_product(dataset, settings):
    response = ask(dataset, settings, "How does GrowMax perform on wheat?")
    assert "no product called 'GrowMax'" in response.answer
    assert "Harvest Plus, Root Boost" in response.answer


def test_greeting_needs_no_tools(dataset, settings):
    response = ask(dataset, settings, "hello")
    assert response.tool_calls == 0
    assert "For example" in response.answer


def test_dataset_overview(dataset, settings):
    response = ask(dataset, settings, "What products and crops are available?")
    assert "6 trials" in response.answer
    assert "Harvest Plus (4)" in response.answer


def test_force_offline_skips_ollama(dataset, settings):
    response = ask(dataset, settings, "List trials in France", force_offline=True)
    assert response.fallback_reason == "offline mode is switched on in the settings"
    assert "T01" in response.answer and "T04" in response.answer


def test_reflection_can_be_switched_off(dataset, settings):
    response = ask(dataset, settings, "Which trials are in Spain?", enable_reflection=False)
    assert response.verification.skipped


def test_stream_emits_logs_then_answer_then_done(dataset, settings):
    request = ChatRequest(question="Which trials have conflicting data?", dataset=dataset)

    async def collect():
        return [event async for event in stream_agent(request, settings)]

    events = asyncio.run(collect())
    types = [event.type for event in events]
    assert types[-2:] == ["answer", "done"]
    assert set(types[:-2]) == {"log"}
    answer = events[-2]
    assert "T03" in answer.message
    assert answer.data["mode"] == "offline"
    assert answer.data["verification"]["passed"] is True


def test_no_fallback_raises_when_model_is_missing(dataset, settings):
    with pytest.raises(RuntimeError, match="unavailable"):
        ask(dataset, settings, "Which trials are in Spain?", offline_fallback=False)
