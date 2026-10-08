"""Offline planner: a deterministic stand-in for the LLM, used when Ollama or the model is unavailable.

It implements LangChain's chat-model interface and speaks the same tool-calling protocol as Qwen, so the
LangGraph loop runs unchanged: it plans tool calls from the question, looks at the results to decide whether
a second round is needed (for example, widening a search that found nothing), then composes the answer from
the tool results. This keeps demos and presentations working without a GPU.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import count
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict

from .composer import compose
from .llm import message_text
from .query import QueryAnalysis, analyze
from .store import EvidenceStore

QUESTION_ROLE = "question"
FEEDBACK_ROLE = "feedback"
ROLE_KEY = "agritrial_role"


def plan_tool_calls(analysis: QueryAnalysis) -> list[tuple[str, dict[str, Any]]]:
    """The tools to call for a question, in order."""
    filters = analysis.filters
    match analysis.intent:
        case "help":
            return []
        case "issues":
            return [("find_data_issues", {})]
        case "detail":
            return [("get_trial_details", {"trial_id": trial_id}) for trial_id in analysis.trial_ids]
        case "compare" if len(analysis.trial_ids) >= 2:
            return [("compare_trials", {"trial_ids": analysis.trial_ids})]
        case "reference":
            return [("list_reference_data", {})]
        case "overview":
            return [("list_reference_data", {}), ("trial_statistics", {"group_by": "product"})]
        case "list":
            return [("search_trials", dict(filters))]
        case _:
            group_by = analysis.group_by or ("product" if analysis.intent == "compare" else None)
            stats_args = {**filters, **({"group_by": group_by} if group_by else {})}
            return [("trial_statistics", stats_args), ("search_trials", dict(filters))]


def plan_follow_up(analysis: QueryAnalysis, results: dict[str, list[dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    """Second-round tool calls that depend on what the first round returned."""
    first = (results.get("search_trials") or results.get("trial_statistics") or [None])[0]
    if not first or first.get("unresolved_terms"):
        return []
    applied = first.get("filters", {})
    found = first.get("count", first.get("trial_count", 0))

    # Nothing matches the combination (e.g. Harvest Plus on rice): look up each part on its own, so the
    # answer can say what the data does contain instead of only "nothing found".
    if found == 0 and len(applied) >= 2:
        return [("search_trials", {name: applied[name]}) for name in ("product", "crop") if name in applied]

    # Evidence questions: read the full record of every trial whose sources conflict, to quote the sources.
    if analysis.evidence:
        trials = (results.get("search_trials") or [{}])[0].get("trials", [])
        return [("get_trial_details", {"trial_id": trial["id"]}) for trial in trials if trial.get("open_conflicts")][:3]
    return []


class RuleBasedChatModel(BaseChatModel):
    """Chat model that plans with rules instead of a neural network."""

    store: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def _llm_type(self) -> str:
        return "agritrial-offline-planner"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"planner": "rule-based"}

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> RuleBasedChatModel:
        # The planner knows the tool names already; binding is a no-op.
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.respond(messages))])

    def respond(self, messages: list[BaseMessage]) -> AIMessage:
        store: EvidenceStore = self.store
        question_index = _last_question_index(messages)
        question = message_text(messages[question_index].content) if question_index is not None else ""
        previous_questions = [
            message_text(m.content)
            for m in messages[: question_index or 0]
            if isinstance(m, HumanMessage) and m.additional_kwargs.get(ROLE_KEY) != FEEDBACK_ROLE
        ]
        previous = analyze(previous_questions[-1], store) if previous_questions else None
        analysis = analyze(question, store, previous)

        after_question = messages[(question_index or 0) + 1 :]
        tool_messages = [m for m in after_question if isinstance(m, ToolMessage)]
        if not tool_messages:
            calls = plan_tool_calls(analysis)
            if calls:
                ids = count(1)
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": f"offline-{next(ids)}", "type": "tool_call"}
                        for name, args in calls
                    ],
                )

        results: dict[str, list[dict[str, Any]]] = {}
        for message in tool_messages:
            try:
                output = json.loads(message_text(message.content))
            except ValueError:
                continue
            if isinstance(output, dict):
                results.setdefault(message.name or "", []).append(output)

        # One more round when the first results call for it, decided from those results.
        rounds = sum(1 for message in after_question if isinstance(message, AIMessage) and message.tool_calls)
        if rounds == 1 and isinstance(messages[-1], ToolMessage):
            if follow_up := plan_follow_up(analysis, results):
                ids = count(1)
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": name, "args": args, "id": f"offline-2-{next(ids)}", "type": "tool_call"}
                        for name, args in follow_up
                    ],
                )
        return AIMessage(content=compose(analysis, results, store))


def _last_question_index(messages: list[BaseMessage]) -> int | None:
    """Index of the user's current question (feedback from the reviewer is not a question)."""
    fallback = None
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, HumanMessage):
            continue
        role = message.additional_kwargs.get(ROLE_KEY)
        if role == QUESTION_ROLE:
            return index
        if role != FEEDBACK_ROLE and fallback is None:
            fallback = index
    return fallback
