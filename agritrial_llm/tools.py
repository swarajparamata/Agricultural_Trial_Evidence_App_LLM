"""LangChain tools the agent can call. Each one answers from the request's dataset snapshot."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from .store import EvidenceStore, fmt_pct

GroupBy = Literal["product", "crop", "country", "year", "trial_type"]


# Descriptions avoid example values: small models copy them into calls the question never asked for.
_ONLY_IF_ASKED = "Leave empty unless the question names it."


class TrialFilters(BaseModel):
    crop: str | list[str] | None = Field(
        None, description=f"Crop name as the user wrote it, or a list of crops. {_ONLY_IF_ASKED}"
    )
    product: str | list[str] | None = Field(
        None, description=f"Product name as the user wrote it, or a list of products. {_ONLY_IF_ASKED}"
    )
    country: str | list[str] | None = Field(
        None, description=f"Country name or code, or a list of countries. {_ONLY_IF_ASKED}"
    )
    year: int | list[int] | None = Field(None, description=f"One trial year, or a list of years. {_ONLY_IF_ASKED}")
    year_from: int | None = Field(None, description=f"Earliest trial year of a span, inclusive. {_ONLY_IF_ASKED}")
    year_to: int | None = Field(None, description=f"Latest trial year of a span, inclusive. {_ONLY_IF_ASKED}")
    trial_type: str | list[str] | None = Field(
        None, description=f"Trial type (scientific, demonstration, ...), or a list. {_ONLY_IF_ASKED}"
    )


class SearchTrialsInput(TrialFilters):
    status: str | None = Field(
        None, description="Data status: consistent, single_source, incomplete, conflict or manual."
    )
    text: str | None = Field(None, description="Words to look for in notes, caveats, source files and custom fields.")


class TrialDetailsInput(BaseModel):
    trial_id: str = Field(description="A trial ID from the question or from an earlier tool result.")


class StatisticsInput(TrialFilters):
    group_by: GroupBy | None = Field(None, description="Break the statistics down (and rank) by this field.")


class CompareInput(BaseModel):
    trial_ids: list[str] = Field(
        description="Two or more trial IDs from the question or from earlier tool results.", min_length=1, max_length=12
    )


class NoInput(BaseModel):
    pass


def _dump(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)


def build_tools(store: EvidenceStore) -> list[BaseTool]:
    """Tools bound to one request's snapshot."""

    def search_trials(
        crop: str | list[str] | None = None,
        product: str | list[str] | None = None,
        country: str | list[str] | None = None,
        year: int | list[int] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        trial_type: str | list[str] | None = None,
        status: str | None = None,
        text: str | None = None,
    ) -> str:
        return _dump(
            store.search(
                crop=crop,
                product=product,
                country=country,
                year=year,
                year_from=year_from,
                year_to=year_to,
                trial_type=trial_type,
                status=status,
                text=text,
            )
        )

    def get_trial_details(trial_id: str) -> str:
        return _dump(store.trial_detail(trial_id))

    def trial_statistics(
        crop: str | list[str] | None = None,
        product: str | list[str] | None = None,
        country: str | list[str] | None = None,
        year: int | list[int] | None = None,
        year_from: int | None = None,
        year_to: int | None = None,
        trial_type: str | list[str] | None = None,
        group_by: str | None = None,
    ) -> str:
        return _dump(
            store.statistics(
                crop=crop,
                product=product,
                country=country,
                year=year,
                year_from=year_from,
                year_to=year_to,
                trial_type=trial_type,
                group_by=group_by,
            )
        )

    def compare_trials(trial_ids: list[str]) -> str:
        return _dump(store.compare(trial_ids))

    def find_data_issues() -> str:
        return _dump(store.data_issues())

    def list_reference_data() -> str:
        return _dump(store.reference_summary())

    return [
        StructuredTool.from_function(
            func=search_trials,
            name="search_trials",
            description=(
                "Find trials matching filters (crop, product, country, year, trial type, data status, free text). "
                "Returns each trial's treated and control yields, uplift %, data status, caveats and source files."
            ),
            args_schema=SearchTrialsInput,
        ),
        StructuredTool.from_function(
            func=get_trial_details,
            name="get_trial_details",
            description=(
                "Full record of one trial: every source file with the values it reported, conflicts between "
                "sources, notes and caveats."
            ),
            args_schema=TrialDetailsInput,
        ),
        StructuredTool.from_function(
            func=trial_statistics,
            name="trial_statistics",
            description=(
                "Yield response of the trials matching the filters: mean, median, min and max uplift %, trials "
                "excluded because a yield is missing, and the range caused by conflicting source values. Use "
                "group_by to rank products, crops, countries, years or trial types."
            ),
            args_schema=StatisticsInput,
        ),
        StructuredTool.from_function(
            func=compare_trials,
            name="compare_trials",
            description=(
                "Side-by-side comparison of two or more trials: fields that differ, ranking by uplift, and warnings "
                "(different crops, conflicting values, missing controls)."
            ),
            args_schema=CompareInput,
        ),
        StructuredTool.from_function(
            func=find_data_issues,
            name="find_data_issues",
            description=(
                "Data-quality report for the whole dataset: conflicting values between sources, missing values, "
                "single-source trials and how often each caveat occurs."
            ),
            args_schema=NoInput,
        ),
        StructuredTool.from_function(
            func=list_reference_data,
            name="list_reference_data",
            description=(
                "What the dataset contains: products, crops, countries, years and trial types with their spelling "
                "variants and trial counts, plus the trial IDs and custom fields."
            ),
            args_schema=NoInput,
        ),
    ]


def format_call(name: str, args: dict[str, Any]) -> str:
    """`search_trials(product='Harvest Plus', crop='Wheat')` for the inspection log."""
    shown = ", ".join(f"{key}={value!r}" for key, value in args.items() if value not in (None, "", []))
    return f"{name}({shown})"


def summarize_output(name: str, output: str) -> str:
    """One-line description of a tool result for the inspection log."""
    try:
        data = json.loads(output)
    except (TypeError, ValueError):
        return "returned a non-JSON result"
    if not isinstance(data, dict):
        return "done"
    if "error" in data:
        return f"error: {data['error']}"

    summary = "done"
    match name:
        case "search_trials":
            ids = [trial["id"] for trial in data.get("trials", [])]
            summary = f"{data.get('count', 0)} trial(s)" + (f": {', '.join(ids)}" if ids else "")
        case "get_trial_details":
            summary = f"{data.get('id')} ({data.get('status')}), {len(data.get('sources', []))} source(s)"
        case "trial_statistics":
            summary = f"{data.get('trial_count', 0)} trial(s), mean uplift {fmt_pct(data.get('mean_uplift_pct'))}"
            if groups := data.get("groups"):
                summary += f", {len(groups)} group(s) by {data.get('group_by')}"
        case "compare_trials":
            differ = data.get("fields_that_differ") or []
            summary = f"{len(data.get('trials', []))} trial(s)" + (f", differ on {', '.join(differ)}" if differ else "")
        case "find_data_issues":
            summary = (
                f"{len(data.get('conflicts', []))} conflict(s), {len(data.get('missing_values', []))} trial(s) with "
                f"missing values, {len(data.get('single_source_trials', []))} single-source"
            )
        case "list_reference_data":
            summary = (
                f"{data.get('trial_count', 0)} trials, {len(data.get('products', []))} product(s), "
                f"{len(data.get('crops', []))} crop(s)"
            )
    if unresolved := data.get("unresolved_terms"):
        summary += f" (unknown: {', '.join(f'{k}={v!r}' for k, v in unresolved.items())})"
    return summary
