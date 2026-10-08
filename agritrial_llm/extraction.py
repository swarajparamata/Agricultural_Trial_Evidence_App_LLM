"""Report extraction: a second LangGraph loop that turns a free-text trial report into structured fields.

    START → extract → validate ─(problems left and attempts remaining)→ extract …
                               └→ END

The backend calls it when its rule-based parser can't read an uploaded report. Validation rejects values
that don't appear in the text, so the model has to copy numbers rather than invent them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .config import ServiceSettings
from .events import EventLog
from .llm import build_chat_model, describe_error, probe_ollama, resolve_options
from .prompts import EXTRACTION_RETRY_PROMPT, EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_PROMPT
from .schemas import ExtractRequest, ExtractResponse, ReportExtraction
from .store import norm_key

REQUIRED_FIELDS = ("trial_id", "crop", "product", "treatment_yield")

_TEXT_NUMBER = re.compile(r"\d{1,3}(?:[ ,.]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?")


class ExtractionState(TypedDict, total=False):
    messages: list[AnyMessage]
    text: str
    filename: str
    attempt: int
    max_attempts: int
    fields: dict[str, Any] | None
    issues: list[str]


@dataclass
class ExtractionRuntime:
    model: Runnable
    model_name: str
    units: list[str]
    events: EventLog


def _numbers_in_text(text: str) -> set[float]:
    """Numbers as written, reading '7,800' and '7 800' as thousands and '7,8' as a decimal comma."""
    numbers = set()
    for raw in _TEXT_NUMBER.findall(text):
        candidates = {raw.replace(",", "").replace(" ", ""), raw.replace(" ", "").replace(",", ".")}
        if raw.count(".") > 1:
            candidates.add(raw.replace(".", "").replace(",", "."))
        for candidate in candidates:
            try:
                numbers.add(float(candidate))
            except ValueError:
                continue
    return numbers


def validate_extraction(fields: dict[str, Any], text: str, filename: str, units: list[str]) -> list[str]:
    issues = []
    for name in REQUIRED_FIELDS:
        if fields.get(name) in (None, ""):
            issues.append(f"{name} is missing; look for it in the text or the file name '{filename}'.")

    trial_id = fields.get("trial_id")
    if trial_id and norm_key(trial_id) not in norm_key(f"{text} {filename}"):
        issues.append(f"trial_id '{trial_id}' does not appear in the report text or file name.")

    year = fields.get("year")
    if year is not None and not 1900 <= int(year) <= 2100:
        issues.append(f"year {year} is not a plausible trial year.")

    written = _numbers_in_text(text)
    for name in ("treatment_yield", "control_yield"):
        value = fields.get(name)
        if value is None:
            continue
        if value <= 0:
            issues.append(f"{name} must be a positive number.")
        elif not any(abs(value - number) < 1e-6 for number in written):
            issues.append(f"{name} {value} does not appear in the report text; copy the number exactly as written.")

    unit = fields.get("yield_unit")
    if fields.get("treatment_yield") is not None and not unit:
        issues.append("yield_unit is missing; give the unit written next to the yields.")
    elif unit and units and norm_key(unit) not in {norm_key(known) for known in units}:
        issues.append(f"yield_unit '{unit}' is not a known unit ({', '.join(units)}).")
    return issues


def _runtime(config: RunnableConfig) -> ExtractionRuntime:
    return config["configurable"]["runtime"]


async def extract_node(state: ExtractionState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    attempt = state.get("attempt", 0) + 1
    messages = list(state["messages"])
    if state.get("issues"):
        messages.append(
            HumanMessage(content=EXTRACTION_RETRY_PROMPT.format(issues="\n".join(f"- {i}" for i in state["issues"])))
        )
    rt.events.log(
        f"[Extract] Attempt {attempt}/{state['max_attempts']}: asking {rt.model_name} for structured fields",
        "processing",
        node="extract",
    )
    try:
        result = await rt.model.ainvoke(messages, config)
        fields = result.model_dump() if isinstance(result, ReportExtraction) else dict(result or {})
        issues: list[str] = []
    except Exception as exc:  # unparseable output counts as a failed attempt
        fields, issues = None, [f"The reply was not valid JSON for the schema ({describe_error(exc)[:200]})."]
        rt.events.log(f"[Extract] Attempt {attempt} returned unusable output", "error", node="extract")
    reply = ReportExtraction(**fields).model_dump_json(exclude_none=True) if fields is not None else "(invalid output)"
    return {"attempt": attempt, "fields": fields, "messages": [*messages, AIMessage(content=reply)], "issues": issues}


async def validate_node(state: ExtractionState, config: RunnableConfig) -> dict[str, Any]:
    rt = _runtime(config)
    if state.get("fields") is None:
        return {}
    issues = validate_extraction(state["fields"], state["text"], state["filename"], rt.units)
    if issues:
        rt.events.log(f"[Validate] {len(issues)} problem(s): {' | '.join(issues)}", "error", node="validate")
    else:
        rt.events.log("[Validate] Extraction passed all checks", "success", node="validate")
    return {"issues": issues}


def route_after_validate(state: ExtractionState) -> str:
    if state.get("issues") and state.get("attempt", 0) < state["max_attempts"]:
        return "extract"
    return END


def build_extraction_graph():
    graph = StateGraph(ExtractionState)
    graph.add_node("extract", extract_node)
    graph.add_node("validate", validate_node)
    graph.add_edge(START, "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges("validate", route_after_validate, {"extract": "extract", END: END})  # retry loop
    return graph.compile(name="agritrial-report-extractor")


EXTRACTION_GRAPH = build_extraction_graph()


def _names(entries) -> str:
    return ", ".join(entry.name for entry in entries) or "(none yet)"


async def run_extraction(
    request: ExtractRequest, settings: ServiceSettings, *, model: Runnable | None = None
) -> ExtractResponse:
    """Extracts the trial fields from a report. `model` lets tests inject a fake structured-output model."""
    options = resolve_options(request.options, settings)
    events = EventLog()
    if model is None:
        status = await probe_ollama(options.base_url, options.model)
        if options.force_offline or not status.ready:
            reason = "offline mode is switched on" if options.force_offline else status.reason
            events.log(f"[Extract] LLM unavailable: {reason}", "error", node="extract")
            return ExtractResponse(
                ok=False, mode="unavailable", model=options.model, issues=[reason], trace=events.events
            )
        model = build_chat_model(options).with_structured_output(ReportExtraction, method="json_schema")

    reference = request.reference
    system = EXTRACTION_SYSTEM_PROMPT.format(
        crops=_names(reference.crops),
        products=_names(reference.products),
        countries=_names(reference.countries),
        trial_types=_names(reference.trial_types),
        units=", ".join(request.units) or "t/ha, kg/ha",
    )
    initial: ExtractionState = {
        "messages": [
            SystemMessage(content=system),
            HumanMessage(content=EXTRACTION_USER_PROMPT.format(filename=request.filename, text=request.text)),
        ],
        "text": request.text,
        "filename": request.filename,
        "attempt": 0,
        "max_attempts": settings.extraction_max_attempts,
        "fields": None,
        "issues": [],
    }
    runtime = ExtractionRuntime(model=model, model_name=options.model, units=request.units, events=events)
    final = await EXTRACTION_GRAPH.ainvoke(
        initial, config={"configurable": {"runtime": runtime}, "recursion_limit": settings.extraction_max_attempts * 3 + 5}
    )
    fields = final.get("fields")
    issues = final.get("issues", [])
    return ExtractResponse(
        ok=fields is not None and not issues,
        mode="llm",
        model=options.model,
        fields=ReportExtraction(**fields) if fields else None,
        issues=issues,
        attempts=final.get("attempt", 0),
        trace=events.events,
    )


def extraction_graph_mermaid() -> str:
    return EXTRACTION_GRAPH.get_graph().draw_mermaid()
