"""The evidence agent: a LangGraph state machine with two loops.

    START → agent ⇄ tools                 tool loop: the model calls tools until it has the evidence it needs
            agent → verify → revise → agent   reflection loop: failed checks go back to the model as feedback
                    verify → finalize → END

The model is Qwen 3.6 via Ollama, or the offline planner when Ollama is unavailable (or fails mid-run).
Per-request dependencies (model, tools, dataset, event log) travel in the run config, so the compiled
graph is shared by all requests.
"""

from __future__ import annotations

import asyncio
import json
import operator
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from .config import ServiceSettings
from .events import EventListener, EventLog
from .llm import (
    ResolvedOptions,
    build_chat_model,
    describe_error,
    message_text,
    probe_ollama,
    resolve_options,
    strip_thinking,
)
from .offline import FEEDBACK_ROLE, QUESTION_ROLE, ROLE_KEY, RuleBasedChatModel, plan_tool_calls
from .prompts import AGENT_SYSTEM_PROMPT, FORCE_ANSWER_PROMPT, PARSER_HINT, REVISION_PROMPT
from .query import analyze
from .schemas import AgentEvent, ChatRequest, ChatResponse, Verification
from .store import FILTER_FIELDS, EvidenceStore, filter_terms, natural_key, ngram_keys, norm_key
from .tools import build_tools, format_call, summarize_output
from .verification import verify_answer

OFFLINE_MODEL_NAME = "offline-planner"


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    mode: str  # "llm" | "offline"
    fallback_reason: str | None
    iterations: int  # model calls in total
    round_iterations: int  # model calls since the last revision (bounded by max_tool_iterations)
    tool_calls: int
    revisions: int
    max_tool_iterations: int
    max_revisions: int
    observations: Annotated[list[dict[str, Any]], operator.add]
    draft: str
    verification: dict[str, Any]
    final_answer: str


@dataclass
class AgentRuntime:
    """Everything a node needs for one request."""

    store: EvidenceStore
    tools: dict[str, BaseTool]
    llm: Runnable | None  # tool-bound chat model
    llm_plain: BaseChatModel | None  # same model without tools, for the forced final answer
    offline: RuleBasedChatModel
    options: ResolvedOptions
    events: EventLog


def _runtime(config: RunnableConfig) -> AgentRuntime:
    return config["configurable"]["runtime"]


# --- nodes ----------------------------------------------------------------------------------


async def agent_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    iteration = state.get("iterations", 0) + 1
    round_iteration = state.get("round_iterations", 0) + 1
    budget_spent = round_iteration > state["max_tool_iterations"]
    mode = state.get("mode", "offline")
    update: dict[str, Any] = {"iterations": iteration, "round_iterations": round_iteration}

    messages = list(state["messages"])
    if budget_spent:
        messages.append(HumanMessage(content=FORCE_ANSWER_PROMPT, additional_kwargs={ROLE_KEY: FEEDBACK_ROLE}))
        rt.events.log("[Agent] Tool budget used up, asking for the final answer", "info", node="agent")

    model_name = rt.options.model if mode == "llm" else OFFLINE_MODEL_NAME
    rt.events.log(f"[Agent] Step {iteration}: reasoning with {model_name}", "processing", node="agent")
    try:
        if mode == "llm":
            model = rt.llm_plain if budget_spent else rt.llm
            response = await model.ainvoke(messages, config)
        else:
            response = await rt.offline.ainvoke(messages, config)
    except Exception as exc:  # network errors, timeouts, model errors
        if mode != "llm" or not rt.options.offline_fallback:
            raise
        reason = f"LLM call failed ({describe_error(exc)})"
        rt.events.log(f"[LLM] {reason}; switching to the offline planner", "error", node="agent")
        update.update(mode="offline", fallback_reason=reason)
        response = await rt.offline.ainvoke(messages, config)

    if not isinstance(response, AIMessage):
        response = AIMessage(content=message_text(getattr(response, "content", response)))
    text = strip_thinking(message_text(response.content))
    tool_calls = [] if budget_spent else list(response.tool_calls)
    response = AIMessage(content=text, tool_calls=tool_calls, id=response.id)

    if tool_calls:
        planned = ", ".join(format_call(call["name"], call["args"]) for call in tool_calls)
        rt.events.log(f"[Agent] Decided to call: {planned}", "info", node="agent")
    else:
        rt.events.log(f"[Agent] Drafted an answer ({len(text)} characters)", "success", node="agent")
        update["draft"] = text
    update["messages"] = [response]
    return update


_ID_ARGS = {"compare_trials": "trial_ids", "get_trial_details": "trial_id"}


def _unseen_trial_ids(state: AgentState, store: EvidenceStore, name: str, args: dict[str, Any]) -> list[str]:
    """Existing trial IDs the model passes without having seen them in the question or an earlier tool result.

    Small models sometimes guess IDs (e.g. compare T01 and T02 for "France vs Germany") and miss trials; making
    them search first keeps answers complete.
    """
    if name not in _ID_ARGS:
        return []
    requested = args.get(_ID_ARGS[name]) or []
    requested = [requested] if isinstance(requested, str) else list(requested)
    question = state.get("question", "")
    seen = set(store.mentioned_trial_ids(question))
    for observation in state.get("observations", []):
        seen |= set(store.mentioned_trial_ids(str(observation.get("output", ""))))
    unseen = []
    for trial_id in map(str, requested):
        trial = store.get(trial_id)
        if trial is not None:
            if trial.id not in seen:
                unseen.append(trial_id)
        # An ID that doesn't exist is fine if the user typed it (the tool says so); otherwise it was invented.
        elif not re.search(rf"(?<![A-Za-z0-9]){re.escape(trial_id)}(?![A-Za-z0-9])", question, re.IGNORECASE):
            unseen.append(trial_id)
    return unseen


_FILTERED_TOOLS = {"search_trials", "trial_statistics"}


def _user_text(state: AgentState) -> str:
    """Everything the user wrote in this conversation (not the reviewer's feedback)."""
    return " ".join(
        message_text(message.content)
        for message in state["messages"]
        if isinstance(message, HumanMessage) and message.additional_kwargs.get(ROLE_KEY) != FEEDBACK_ROLE
    )


def _ungrounded_filters(state: AgentState, store: EvidenceStore, name: str, args: dict[str, Any]) -> list[str]:
    """Filter values the user never mentioned, such as a country or year the model made up.

    A value counts as mentioned when its words are in the conversation or it names the same thing as words that
    are (an alias or code: 'France' for 'FR', 'Potato' for 'potatoes'). Years may be one off ('after 2022').
    """
    if name not in _FILTERED_TOOLS:
        return []
    text = _user_text(state)
    grams = ngram_keys(text)
    keys = {key for key, _ in grams}
    problems = []
    for field in FILTER_FIELDS:
        terms = filter_terms(args.get(field))
        if not terms:
            continue
        mentioned = {store.resolve(field, original) for _, original in grams} - {None}
        for term in terms:
            # A known name in the wrong field (country='Rice'): say where it belongs.
            if store.resolve(field, term) is None:
                owner = next((other for other in FILTER_FIELDS if store.resolve(other, term) is not None), None)
                if owner is not None:
                    right = store.resolve(owner, term)
                    problems.append(
                        f"{field}={term!r} ({right} is a {owner.replace('_', ' ')}, not a {field.replace('_', ' ')}: "
                        f"use {owner}={right!r})"
                    )
                    continue
            if norm_key(term) not in keys and store.resolve(field, term) not in mentioned:
                problems.append(f"{field}={term!r} (not in the question)")
    years = {int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", text)}
    for field in ("year", "year_from", "year_to"):
        for value in filter_terms(args.get(field)):
            match = re.search(r"\d{4}", value)
            if match and not any(abs(int(match.group()) - year) <= 1 for year in years):
                problems.append(f"{field}={value} (not in the question)")
    return problems


def _filters_named(state: AgentState, store: EvidenceStore) -> str:
    filters = analyze(state.get("question", ""), store).filters
    return ", ".join(f"{name}={value!r}" for name, value in filters.items()) or "none"


async def tools_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    last = state["messages"][-1]
    messages, observations = [], []
    for call in last.tool_calls:
        name, args = call["name"], call.get("args") or {}
        tool = rt.tools.get(name)
        if tool is None:
            output = json.dumps({"error": f"Unknown tool {name!r}. Available tools: {', '.join(rt.tools)}."})
        elif unseen := _unseen_trial_ids(state, rt.store, name, args):
            output = json.dumps(
                {
                    "error": (
                        f"Trial IDs {', '.join(unseen)} were not in the question or in an earlier result. Find the "
                        "trials with search_trials first, then use the IDs it returns."
                    )
                }
            )
        elif ungrounded := _ungrounded_filters(state, rt.store, name, args):
            output = json.dumps(
                {
                    "error": (
                        f"Wrong filters: {'; '.join(ungrounded)}. Use only the filters the question names "
                        f"({_filters_named(state, rt.store)})."
                    )
                }
            )
        else:
            try:
                output = await asyncio.to_thread(tool.invoke, args)
            except Exception as exc:  # invalid arguments: the model sees the error and can retry
                output = json.dumps({"error": describe_error(exc)[:400]})
        summary = summarize_output(name, output)
        status = "error" if summary.startswith("error") else "success"
        rt.events.log(f"[Tool] {format_call(name, args)} → {summary}", status, node="tools")
        messages.append(ToolMessage(content=output, tool_call_id=call.get("id") or name, name=name))
        observations.append({"tool": name, "args": args, "output": output})
    return {
        "messages": messages,
        "observations": observations,
        "tool_calls": state.get("tool_calls", 0) + len(last.tool_calls),
    }


async def verify_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    if not rt.options.enable_reflection:
        rt.events.log("[Verify] Reflection is switched off; skipping the checks", "info", node="verify")
        return {"verification": {"passed": True, "issues": [], "skipped": True}}

    result = verify_answer(state.get("draft", ""), state["question"], state.get("observations", []), rt.store)
    if result.passed:
        rt.events.log("[Verify] Grounding checks passed: numbers, trial IDs and caveats match the data", "success", node="verify")
    else:
        rt.events.log(f"[Verify] {len(result.issues)} problem(s): {' | '.join(result.issues)}", "error", node="verify")
    return {"verification": result.as_dict()}


async def revise_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    revision = state.get("revisions", 0) + 1
    issues = state["verification"]["issues"]
    rt.events.log(
        f"[Reflect] Revision {revision}/{state['max_revisions']}: sending the problems back to the agent",
        "processing",
        node="revise",
    )
    feedback = REVISION_PROMPT.format(issues="\n".join(f"- {issue}" for issue in issues))
    return {
        "messages": [HumanMessage(content=feedback, additional_kwargs={ROLE_KEY: FEEDBACK_ROLE})],
        "revisions": revision,
        "round_iterations": 0,
    }


async def finalize_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    answer = state.get("draft", "").strip() or "I couldn't produce an answer from the trial data."
    verification = state.get("verification") or {}
    if verification and not verification.get("passed"):
        answer += "\n\n_Automatic fact-check could not confirm everything: " + " ".join(verification["issues"][:2]) + "_"
    rt.events.log("[Agent] Response generated successfully.", "success", node="finalize")
    return {"final_answer": answer}


# --- routing --------------------------------------------------------------------------------


def route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "verify"


def route_after_verify(state: AgentState) -> str:
    if state["verification"].get("passed"):
        return "finalize"
    if state.get("revisions", 0) < state["max_revisions"]:
        return "revise"
    return "finalize"


def build_agent_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("verify", verify_node)
    graph.add_node("revise", revise_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"tools": "tools", "verify": "verify"})
    graph.add_edge("tools", "agent")  # tool loop
    graph.add_conditional_edges("verify", route_after_verify, {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "agent")  # reflection loop
    graph.add_edge("finalize", END)
    return graph.compile(name="agritrial-evidence-agent")


AGENT_GRAPH = build_agent_graph()


# --- running --------------------------------------------------------------------------------


def _parser_hint(request: ChatRequest, history: list, store: EvidenceStore) -> str:
    """The rule-based parser's reading of the question as suggested first tool calls.

    Small models tend to copy values they see; giving them correctly resolved filters (canonical names, year
    ranges, lists) keeps their tool calls on the question. The model still decides what to call and when.
    """
    previous_question = next((turn.content for turn in reversed(history) if turn.role == "user"), None)
    previous = analyze(previous_question, store) if previous_question else None
    calls = plan_tool_calls(analyze(request.question, store, previous))
    if not calls:
        return ""
    return PARSER_HINT.format(calls="; ".join(format_call(name, args) for name, args in calls))


def _initial_messages(request: ChatRequest, options: ResolvedOptions, store: EvidenceStore) -> list[AnyMessage]:
    extra = f"\n\nAdditional instructions from the administrator:\n{options.extra_instructions}" if options.extra_instructions else ""
    history = request.history[-options.history_turns :] if options.history_turns else []
    extra += _parser_hint(request, history, store)
    messages: list[AnyMessage] = [SystemMessage(content=AGENT_SYSTEM_PROMPT.format(unit=store.unit, extra=extra))]
    for turn in history:
        messages.append(HumanMessage(content=turn.content) if turn.role == "user" else AIMessage(content=turn.content))
    messages.append(HumanMessage(content=request.question, additional_kwargs={ROLE_KEY: QUESTION_ROLE}))
    return messages


async def run_agent(
    request: ChatRequest,
    settings: ServiceSettings,
    listener: EventListener | None = None,
    *,
    chat_model: BaseChatModel | None = None,
) -> ChatResponse:
    """Runs the agent graph. `chat_model` replaces Ollama (tests use it to script the model's turns)."""
    started = time.perf_counter()
    events = EventLog(listener)
    options = resolve_options(request.options, settings)
    store = EvidenceStore(request.dataset)
    tool_list = build_tools(store)
    offline = RuleBasedChatModel(store=store)

    preview = request.question if len(request.question) <= 60 else f"{request.question[:60]}…"
    events.log(f'[Intent] Received request: "{preview}"', "processing", node="start")
    events.log(f"[Data] {len(store.trials)} reconciled trial record(s) in context, yields in {store.unit}", "info", node="start")

    llm = llm_plain = None
    mode, fallback_reason = "offline", None
    if chat_model is not None:
        llm_plain, llm, mode = chat_model, chat_model.bind_tools(tool_list), "llm"
    elif options.force_offline:
        fallback_reason = "offline mode is switched on in the settings"
    else:
        status = await probe_ollama(options.base_url, options.model)
        if status.ready:
            llm_plain = build_chat_model(options)
            llm = llm_plain.bind_tools(tool_list)
            mode = "llm"
            events.log(f"[LLM] Using {options.model} via Ollama at {options.base_url}", "info", node="start")
        else:
            fallback_reason = status.reason
            if not options.offline_fallback:
                raise RuntimeError(f"The language model is unavailable: {status.reason}.")
    if mode == "offline":
        events.log(f"[LLM] {fallback_reason}; using the offline rule-based planner", "info", node="start")

    runtime = AgentRuntime(
        store=store,
        tools={tool.name: tool for tool in tool_list},
        llm=llm,
        llm_plain=llm_plain,
        offline=offline,
        options=options,
        events=events,
    )
    initial: AgentState = {
        "messages": _initial_messages(request, options, store),
        "question": request.question,
        "mode": mode,
        "fallback_reason": fallback_reason,
        "iterations": 0,
        "round_iterations": 0,
        "tool_calls": 0,
        "revisions": 0,
        "max_tool_iterations": options.max_tool_iterations,
        "max_revisions": options.max_revisions,
        "observations": [],
    }
    # Each loop pass is a few graph steps; leave room for every allowed iteration and revision.
    recursion_limit = (options.max_tool_iterations + 2) * 2 * (options.max_revisions + 1) + 10
    final = await AGENT_GRAPH.ainvoke(
        initial, config={"configurable": {"runtime": runtime}, "recursion_limit": recursion_limit}
    )

    answer = final["final_answer"]
    trial_ids = sorted(store.mentioned_trial_ids(answer), key=natural_key)
    sources = sorted({source.file for trial_id in trial_ids if (trial := store.get(trial_id)) for source in trial.sources})
    verification = final.get("verification") or {"passed": True, "issues": [], "skipped": True}
    final_mode = final.get("mode", mode)
    return ChatResponse(
        answer=answer,
        mode=final_mode,
        model=options.model if final_mode == "llm" else OFFLINE_MODEL_NAME,
        fallback_reason=final.get("fallback_reason"),
        trial_ids=trial_ids,
        sources=sources,
        iterations=final.get("iterations", 0),
        revisions=final.get("revisions", 0),
        tool_calls=final.get("tool_calls", 0),
        verification=Verification(**verification),
        trace=events.events,
        duration_ms=int((time.perf_counter() - started) * 1000),
    )


async def stream_agent(request: ChatRequest, settings: ServiceSettings) -> AsyncIterator[AgentEvent]:
    """Yields step events while the agent runs, then the answer (or an error) and a final `done` event."""
    queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    async def worker() -> None:
        try:
            response = await run_agent(request, settings, listener=queue.put_nowait)
            data = response.model_dump(exclude={"trace", "answer"})
            queue.put_nowait(AgentEvent(type="answer", status="success", message=response.answer, data=data))
        except Exception as exc:
            queue.put_nowait(AgentEvent(type="error", status="error", message=describe_error(exc)))
        finally:
            queue.put_nowait(AgentEvent(type="done"))
            queue.put_nowait(None)

    task = asyncio.create_task(worker())
    try:
        while (event := await queue.get()) is not None:
            yield event
    finally:
        if not task.done():
            task.cancel()


def agent_graph_mermaid() -> str:
    return AGENT_GRAPH.get_graph().draw_mermaid()
