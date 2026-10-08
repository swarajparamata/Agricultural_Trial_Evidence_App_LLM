"""The LangGraph loops, driven by a scripted chat model standing in for Qwen."""

import asyncio
from collections.abc import Callable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agritrial_llm.agent import run_agent
from agritrial_llm.schemas import AgentOptions, ChatRequest, Dataset

Step = AIMessage | Exception | Callable[[list[BaseMessage]], AIMessage]


class ScriptedChatModel(BaseChatModel):
    """Plays back one scripted turn per call and records what it was shown."""

    script: list[Any]
    calls: int = 0
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        step: Step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        self.seen.append(list(messages))
        if isinstance(step, Exception):
            raise step
        message = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=message)])


def tool_call(name: str, call_id: str = "call-1", **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def run(dataset: Dataset, settings, model: ScriptedChatModel, question: str, **options):
    request = ChatRequest(question=question, dataset=dataset, options=AgentOptions(**options))
    return asyncio.run(run_agent(request, settings, chat_model=model))


QUESTION = "What is the average yield uplift of Harvest Plus on wheat?"
GOOD_ANSWER = (
    "Harvest Plus lifts wheat yields by +6.3% on average across T01, T02 and T06; T05 is excluded because its "
    "control yield is missing."
)


def test_reflection_loop_revises_an_ungrounded_answer(dataset, settings):
    model = ScriptedChatModel(
        script=[
            tool_call("trial_statistics", product="HarvestPlus", crop="wheat"),
            AIMessage(content="Harvest Plus lifts wheat yields by 9.9% on average (T01, T02, T06). T05 has no control yield."),
            AIMessage(content=GOOD_ANSWER),
        ]
    )
    response = run(dataset, settings, model, QUESTION)

    assert response.mode == "llm"
    assert response.answer == GOOD_ANSWER
    assert (response.iterations, response.tool_calls, response.revisions) == (3, 1, 1)
    assert response.verification.passed
    feedback = model.seen[2][-1]
    assert isinstance(feedback, HumanMessage) and "9.9%" in feedback.content
    messages = [event.message for event in response.trace]
    assert any(message.startswith("[Reflect] Revision 1/2") for message in messages)


def test_tool_loop_stops_at_the_iteration_budget(dataset, settings):
    def keep_searching(messages: list[BaseMessage]) -> AIMessage:
        if "You have used all the tool calls" in str(messages[-1].content):
            return AIMessage(content="T01 shows +5.0% (trials_a.csv, trials_b.csv).")
        return tool_call("search_trials", call_id=f"call-{len(messages)}", country="France")

    model = ScriptedChatModel(script=[keep_searching])
    response = run(dataset, settings, model, "How did trials in France do?", max_tool_iterations=2)

    assert response.tool_calls == 2
    assert response.iterations == 3
    assert response.answer.startswith("T01 shows +5.0%")
    assert any("Tool budget used up" in event.message for event in response.trace)


def test_tool_errors_are_returned_to_the_model(dataset, settings):
    model = ScriptedChatModel(
        script=[
            tool_call("trial_statistics", group_by="planet"),
            tool_call("trial_statistics", call_id="call-2", group_by="product"),
            AIMessage(content="Harvest Plus leads with +6.3% (T01, T02, T06; T05 lacks a control yield)."),
        ]
    )
    response = run(dataset, settings, model, "Which product performs best?")

    errors = [event for event in response.trace if event.status == "error" and event.message.startswith("[Tool]")]
    assert len(errors) == 1 and "ValidationError" in errors[0].message
    assert response.tool_calls == 2
    assert response.verification.passed, response.verification.issues


def test_llm_failure_mid_run_falls_back_to_the_offline_planner(dataset, settings):
    model = ScriptedChatModel(script=[ConnectionError("connection refused")])
    response = run(dataset, settings, model, QUESTION)

    assert response.mode == "offline"
    assert response.fallback_reason.startswith("LLM call failed (ConnectionError")
    assert "+6.3%" in response.answer
    assert response.verification.passed


def test_unfixable_answers_stop_after_max_revisions(dataset, settings):
    model = ScriptedChatModel(script=[AIMessage(content="Harvest Plus adds 12.7% to wheat yields.")])
    response = run(dataset, settings, model, QUESTION, max_revisions=1)

    assert response.revisions == 1
    assert not response.verification.passed
    assert "_Automatic fact-check could not confirm everything:" in response.answer


def test_guessed_trial_ids_are_sent_back_to_search_first(dataset, settings):
    answer = (
        "France: T01 shows +5.0%. Germany: T02 shows +4.0%; T05 has no control yield, so its uplift can't be "
        "calculated."
    )
    model = ScriptedChatModel(
        script=[
            # Guesses the IDs instead of searching (what a small model did for "France vs Germany").
            tool_call("compare_trials", trial_ids=["T01", "T02"]),
            tool_call("search_trials", "call-2", product="Harvest Plus", crop="wheat", country=["France", "Germany"]),
            tool_call("compare_trials", "call-3", trial_ids=["T01", "T02", "T05"]),
            AIMessage(content=answer),
        ]
    )
    response = run(dataset, settings, model, "Compare Harvest Plus wheat results in France and Germany.")

    tools = [event for event in response.trace if event.message.startswith("[Tool]")]
    assert tools[0].status == "error" and "search_trials first" in tools[0].message
    assert tools[1].message.endswith("3 trial(s): T01, T02, T05")
    assert tools[2].status == "success"  # the IDs came from the search, so the comparison runs
    assert response.answer == answer and response.verification.passed


def test_invented_trial_ids_are_rejected_but_typed_ones_reach_the_tool(dataset, settings):
    model = ScriptedChatModel(
        script=[
            tool_call("compare_trials", trial_ids=["TR-HPW-FR", "TR-HPW-DE"]),
            tool_call("get_trial_details", "call-2", trial_id="T99"),
            AIMessage(content="There is no trial T99 in the data."),
        ]
    )
    response = run(dataset, settings, model, "Compare the French and German trials, and tell me about T99.")
    tools = [event for event in response.trace if event.message.startswith("[Tool]")]
    assert tools[0].status == "error" and "TR-HPW-FR, TR-HPW-DE were not in the question" in tools[0].message
    assert "No trial with ID 'T99'" in tools[1].message  # typed by the user, so the tool answers


def test_the_prompt_carries_the_parsers_reading_of_the_question(dataset, settings):
    model = ScriptedChatModel(script=[AIMessage(content="Harvest Plus was not tested on rice.")])
    run(dataset, settings, model, "Compare Harvest Plus wheat results in France and Germany.")
    system = model.seen[0][0].content
    assert "suggests these first tool calls" in system
    assert "country=['France', 'Germany']" in system and "group_by='country'" in system


def test_made_up_filters_are_sent_back(backend_dataset, settings):
    model = ScriptedChatModel(
        script=[
            # A small model added a country and years the user never mentioned.
            tool_call("search_trials", product="Harvest Plus", crop="Rice", country="India", year_from=2015),
            tool_call("search_trials", "call-2", product="HarvestPlus", crop="rice"),
            AIMessage(content="No trial in the data tests Harvest Plus on rice, so there is no evidence either way."),
        ]
    )
    response = run(backend_dataset, settings, model, "Is there evidence for Harvest Plus improving rice yields?")
    tools = [event for event in response.trace if event.message.startswith("[Tool]")]
    assert tools[0].status == "error"
    assert "country='India' (not in the question); year_from=2015 (not in the question)" in tools[0].message
    assert "crop='Rice', product='Harvest Plus'" in tools[0].message  # what the question does name
    assert tools[1].status == "success"  # spelling variants of mentioned values are fine
    assert response.verification.passed


def test_a_name_in_the_wrong_field_is_corrected(backend_dataset, settings):
    model = ScriptedChatModel(
        script=[
            tool_call("search_trials", product="Harvest Plus", country="Rice"),
            AIMessage(content="No trial in the data tests Harvest Plus on rice."),
        ]
    )
    response = run(backend_dataset, settings, model, "Is there evidence for Harvest Plus improving rice yields?")
    first = next(event for event in response.trace if event.message.startswith("[Tool]"))
    assert first.status == "error" and "Rice is a crop, not a country: use crop='Rice'" in first.message
