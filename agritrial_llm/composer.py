"""Answer templates for the offline planner. Every number comes from a tool result."""

from __future__ import annotations

from itertools import groupby
from typing import Any

from .query import QueryAnalysis
from .store import CORE_FIELD_LABELS, EvidenceStore, fmt_num, fmt_pct

ToolResults = dict[str, list[dict[str, Any]]]

# Caveats shown inline next to the trial, so they are left out of the caveat list.
_INLINE_CAVEAT_PREFIXES = ("Conflicting", "Control yield missing", "Treated yield missing")

_MISSING_LABELS = {"control_yield": "control yield", "treatment_yield": "treated yield"}


def compose(analysis: QueryAnalysis, results: ToolResults, store: EvidenceStore) -> str:
    if analysis.intent == "help":
        return _help(store)
    if "find_data_issues" in results:
        return _issues(results["find_data_issues"][0])
    if "compare_trials" in results:
        return _compare(results["compare_trials"][0])
    searches = results.get("search_trials", [])
    details = results.get("get_trial_details", [])
    # Details fetched as a follow-up to a search or statistics call support that answer; alone, they are the answer.
    if details and not searches and "trial_statistics" not in results:
        return "\n\n".join(_detail(detail) for detail in details)
    if analysis.intent == "list" and searches:
        return _list(searches[0], analysis, related=searches[1:])
    if "trial_statistics" in results:
        reference = results.get("list_reference_data", [None])[0]
        search = searches[0] if searches else None
        answer = _statistics(
            results["trial_statistics"][0], search, analysis, store, related=searches[1:], details=details
        )
        return f"{_reference(reference)}\n\n{answer}" if reference else answer
    if "list_reference_data" in results:
        return _reference(results["list_reference_data"][0])
    if searches:
        return _list(searches[0], analysis, related=searches[1:])
    return "I couldn't find anything in the trial data for that question."


# --- formatting helpers ---------------------------------------------------------------------


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _and_join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def _uplift_span(stats: dict[str, Any]) -> str:
    """'range +4.0% in T02 to +10.0% in T06', naming every tied trial; '+5.0% in each' when all are equal."""
    low, high = stats["min_uplift"], stats["max_uplift"]
    if low["uplift_pct"] == high["uplift_pct"]:
        return f"{fmt_pct(high['uplift_pct'])} in each"
    return (
        f"range {fmt_pct(low['uplift_pct'])} in {_and_join(low['trial_ids'])} "
        f"to {fmt_pct(high['uplift_pct'])} in {_and_join(high['trial_ids'])}"
    )


def _sentence_case(text: str) -> str:
    return text[:1].lower() + text[1:] if text[:2] != text[:2].upper() else text


def _names(value: Any) -> str:
    """A filter value as text: 'France', or 'France and Germany' for a list."""
    return _and_join([str(item) for item in value]) if isinstance(value, list) else str(value)


def _years(filters: dict[str, Any]) -> str:
    """'2024', '2021 and 2023', '2021 to 2025', '2021 onwards', 'up to 2025', or ''."""
    if (year := filters.get("year")) is not None:
        return _names(year)
    low, high = filters.get("year_from"), filters.get("year_to")
    if low and high:
        return f"{low} to {high}"
    if low:
        return f"{low} onwards"
    return f"up to {high}" if high else ""


def _trial_phrase(filters: dict[str, Any]) -> str:
    """' of Harvest Plus on Wheat in Germany from 2021 to 2025 (Scientific)' for list headings."""
    phrase = ""
    if product := filters.get("product"):
        phrase += f" of {_names(product)}"
    if crop := filters.get("crop"):
        phrase += f" on {_names(crop)}"
    if country := filters.get("country"):
        phrase += f" in {_names(country)}"
    if years := _years(filters):
        phrase += f" {years}" if years.startswith("up to") else f" from {years}"
    if trial_type := filters.get("trial_type"):
        phrase += f" ({_names(trial_type)})"
    return phrase


def _subject(filters: dict[str, Any]) -> str:
    """'Harvest Plus on Wheat in Germany (2024, Scientific trials)' or 'All trials'."""
    subject = _names(filters["product"]) if filters.get("product") else ""
    if crop := filters.get("crop"):
        subject = f"{subject} on {_names(crop)}" if subject else f"{_names(crop)} trials"
    if country := filters.get("country"):
        subject = f"{subject} in {_names(country)}" if subject else f"Trials in {_names(country)}"
    qualifiers = [value for value in (_years(filters), filters.get("trial_type") and _names(filters["trial_type"])) if value]
    if filters.get("trial_type"):
        qualifiers[-1] = f"{qualifiers[-1]} trials"
    if not subject:
        subject = "All trials"
    return f"{subject} ({', '.join(qualifiers)})" if qualifiers else subject


def _none_found(filters: dict[str, Any]) -> str:
    """'I found no Rice trials.' or 'I found no trials for Harvest Plus on Rice.'"""
    subject = _subject(filters)
    if subject.endswith("trials"):
        return f"I found no **{subject}**."
    return f"I found no trials for **{subject}**."


def _no_match(filters: dict[str, Any], related: list[dict[str, Any]]) -> str:
    """No trial matches the combination; say what the data holds for each part of it (from follow-up searches)."""
    lines = [f"No trials in the data test **{_subject(filters)}**, so there is no evidence either way."]
    for output in related:
        applied = output.get("filters", {})
        trials = output.get("trials", [])
        # Counts rather than trial IDs: naming a trial would require its caveats, which belong to another question.
        found = plural(output.get("count", len(trials)), "trial")
        if "product" in applied:
            crops = sorted({trial["crop"] for trial in trials if trial.get("crop")})
            lines.append(
                f"{_names(applied['product'])} has only been trialled on {_and_join(crops)} ({found})."
                if crops
                else f"{_names(applied['product'])} does not appear in any trial."
            )
        elif "crop" in applied:
            products = sorted({trial["product"] for trial in trials if trial.get("product")})
            lines.append(
                f"{_names(applied['crop'])} has only been trialled with {_and_join(products)} ({found})."
                if products
                else f"No trial in the data involves {_names(applied['crop'])}."
            )
    return " ".join(lines)


def _id_list(ids: list[str], total: int, limit: int = 6) -> str:
    """'T01, T02 and T05', or 'T01, T02, … (40 trials)' when there are many."""
    if total <= limit and len(ids) == total:
        return _and_join(ids) if ids else "none"
    return f"{', '.join(ids[:limit])}, … ({total} trials)"


def _evidence_summary(stats: dict[str, Any], trials: list[dict[str, Any]]) -> str:
    """How the trials split between increases, no difference and decreases, naming the trials behind each."""
    measured = stats.get("trials_with_uplift", 0)
    if not measured:
        return "**What the evidence shows:** none of these trials has both yields, so none of them can show an effect."
    with_uplift = [trial for trial in trials if trial.get("uplift_pct") is not None]

    def named(selected: list[dict[str, Any]]) -> str:
        shown = [f"{t['id']} ({t.get('trial_type') or 'type not recorded'}, {fmt_pct(t['uplift_pct'])})" for t in selected[:4]]
        return _and_join(shown) + (" and others" if len(selected) > 4 else "")

    increase = stats.get("trials_with_increase", 0)
    parts = []
    if increase:
        verb = "shows" if increase == 1 else "show"
        parts.append(
            f"{increase} of {plural(measured, 'trial')} with a control {verb} a yield increase: "
            f"{named([t for t in with_uplift if t['uplift_pct'] > 0])}."
        )
    else:
        parts.append(f"None of the {plural(measured, 'trial')} with a control shows a yield increase.")
    if flat := [t for t in with_uplift if t["uplift_pct"] == 0]:
        parts.append(f"No difference in {named(flat)}.")
    if lower := [t for t in with_uplift if t["uplift_pct"] < 0]:
        parts.append(f"Lower yield than the control in {named(lower)}.")
    if conflicted := [trial["id"] for trial in trials if trial.get("open_conflicts")]:
        parts.append(f"The sources of {_and_join(conflicted)} disagree, so that result is uncertain.")
    return "**What the evidence shows:** " + " ".join(parts)


def _source_notes(details: list[dict[str, Any]]) -> str:
    """What the sources themselves say about trials fetched in full (e.g. conflicting ones)."""
    lines = [f"- {detail['id']} · {note}" for detail in details if detail.get("id") for note in detail.get("notes", [])]
    return "**From the sources**\n" + "\n".join(lines) if lines else ""


def _where(trial: dict[str, Any]) -> str:
    parts = [str(part) for part in (trial.get("country"), trial.get("year"), trial.get("trial_type")) if part]
    return f" ({', '.join(parts)})" if parts else ""


def _what(trial: dict[str, Any]) -> str:
    if trial.get("product") and trial.get("crop"):
        return f"{trial['product']} on {trial['crop']}"
    return trial.get("product") or trial.get("crop") or "Trial"


def _yields(trial: dict[str, Any]) -> str:
    unit = trial.get("unit", "")
    treated, control = trial.get("treatment_yield"), trial.get("control_yield")
    if treated is not None and control is not None:
        text = f"{fmt_num(treated)} vs {fmt_num(control)} {unit} → {fmt_pct(trial.get('uplift_pct'))}"
        if trial.get("uplift_range_pct"):
            low, high = trial["uplift_range_pct"]
            text += f" ({fmt_pct(low)} to {fmt_pct(high)} given the conflicting values)"
        return text
    missing = ", ".join(_MISSING_LABELS.get(name, name.replace("_", " ")) for name in trial.get("missing_fields", []))
    reported = f"treated {fmt_num(treated)} {unit}, " if treated is not None else ""
    return f"{reported}{missing or 'yield'} missing, so uplift can't be calculated"


def _trial_line(trial: dict[str, Any], *, with_what: bool = False) -> str:
    label = f"{trial['id']} – {_what(trial)}" if with_what else trial["id"]
    line = f"- {label}{_where(trial)}: {_yields(trial)}"
    if sources := trial.get("sources"):
        line += f" · {', '.join(sources)}"
    for conflict in trial.get("open_conflicts", []):
        line += f"\n  - ⚠ Sources conflict on {_sentence_case(conflict)}"
    return line


def _conflict_warnings(trials: list[dict[str, Any]]) -> list[str]:
    return [
        f"- ⚠ {trial['id']}: sources conflict on {_sentence_case(conflict)}"
        for trial in trials
        for conflict in trial.get("open_conflicts", [])
    ]


def _caveat_lines(trials: list[dict[str, Any]]) -> list[str]:
    lines = []
    for trial in trials:
        caveats = [c for c in trial.get("caveats", []) if not c.startswith(_INLINE_CAVEAT_PREFIXES)]
        if caveats:
            lines.append(f"- {trial['id']}: {'; '.join(caveats)}")
    return lines


def _no_significance_note(store: EvidenceStore) -> str:
    reports_significance = any(
        "signific" in ref.label.lower() or "p-value" in ref.label.lower() for ref in store.custom_fields.values()
    )
    return "" if reports_significance else "No statistical significance is reported in the data, so treat these uplifts as indicative."


def _unresolved(output: dict[str, Any]) -> str:
    unresolved = output.get("unresolved_terms") or {}
    known = output.get("known_values") or {}
    lines = []
    for name, term in unresolved.items():
        label = CORE_FIELD_LABELS.get(name, name).lower()
        options = ", ".join(known.get(name, [])) or "none"
        lines.append(f"There is no {label} called '{term}' in the data. Known {label} values: {options}.")
    return " ".join(lines)


# --- intents --------------------------------------------------------------------------------


def _statistics(
    stats: dict[str, Any],
    search: dict[str, Any] | None,
    analysis: QueryAnalysis,
    store: EvidenceStore,
    *,
    related: list[dict[str, Any]] = (),
    details: list[dict[str, Any]] = (),
) -> str:
    filters = stats.get("filters", {})
    subject = _subject(filters)
    if stats.get("unresolved_terms"):
        if analysis.evidence:
            asked = _subject({**filters, **stats["unresolved_terms"]})
            return f"No trials in the data test **{asked}**, so there is no evidence either way. {_unresolved(stats)}"
        return _unresolved(stats)
    if stats.get("trial_count", 0) == 0:
        return _no_match(filters, list(related)) if related else _none_found(filters)

    blocks: list[str] = []
    groups = stats.get("groups")
    if groups:
        blocks.append(_ranking(stats, groups))
    elif "mean_uplift_pct" in stats:
        headline = (
            f"**{subject}**: mean yield uplift **{fmt_pct(stats['mean_uplift_pct'])}** across "
            f"{plural(stats['trials_with_uplift'], 'trial')} with a control yield"
        )
        if stats.get("trials_with_uplift", 0) > 1:
            headline += f" ({_uplift_span(stats)})"
        headline += "."
        if spread := stats.get("mean_uplift_range_pct"):
            headline += (
                f" Because some sources conflict, the mean could be anywhere from {fmt_pct(spread[0])} "
                f"to {fmt_pct(spread[1])}."
            )
        blocks.append(headline)
    else:
        blocks.append(
            f"**{subject}**: {plural(stats['trial_count'], 'trial')} found, but none has both a treated and a "
            "control yield, so uplift can't be calculated."
        )

    trials = (search or {}).get("trials", [])
    if analysis.evidence and "mean_uplift_pct" in stats:
        blocks.append(_evidence_summary(stats, trials))
    with_what = not (filters.get("product") and filters.get("crop"))
    if groups and analysis.intent != "compare":
        # Rankings already name the extremes; only flag conflicts so the answer stays short.
        if warnings := _conflict_warnings(trials):
            blocks.append("\n".join(warnings))
    elif trials:
        # A comparison lists the trials behind each group, so the reader sees what is being compared.
        blocks.append("\n".join(_trial_line(trial, with_what=with_what) for trial in trials))
    if notes := _source_notes(list(details)):
        blocks.append(notes)
    if caveats := _caveat_lines(trials):
        blocks.append("**Caveats**\n" + "\n".join(caveats))
    if note := stats.get("note"):
        blocks.append(note)
    if "mean_uplift_pct" in stats or groups:
        if note := _no_significance_note(store):
            blocks.append(note)
    return "\n\n".join(blocks)


def _ranking(stats: dict[str, Any], groups: list[dict[str, Any]]) -> str:
    field = CORE_FIELD_LABELS.get(stats.get("group_by", ""), "group").lower()
    ranked = [group for group in groups if "mean_uplift_pct" in group]
    lines = []
    if ranked:
        # Groups are sorted by mean uplift (rounded to 0.1 %), so groups tied for the top come first.
        top = ranked[0]["mean_uplift_pct"]
        leaders = [group for group in ranked if group["mean_uplift_pct"] == top]
        names = _and_join([f"**{group['group']}**" for group in leaders])
        verb = "share" if len(leaders) > 1 else "has"
        headline = f"{names} {verb} the highest mean yield uplift: **{fmt_pct(top)}**"
        if len(ranked) > len(leaders):
            runner_up = ranked[len(leaders)]
            headline += f", ahead of {runner_up['group']} at {fmt_pct(runner_up['mean_uplift_pct'])}"
        lines.append(headline + ".")
    else:
        lines.append(f"None of the {field} groups has trials with both treated and control yields.")

    lines.append(f"\nMean uplift by {field}:")
    for position, group in enumerate(groups, start=1):
        count = f"{group['trials_with_uplift']} of {plural(group['trial_count'], 'trial')} with a control"
        name = f"{group['group']} ({_id_list(group.get('trial_ids', []), group['trial_count'], limit=4)})"
        if "mean_uplift_pct" not in group:
            lines.append(f"{position}. {name}: n/a ({count})")
            continue
        text = f"{position}. {name}: {fmt_pct(group['mean_uplift_pct'])} ({count}"
        if group["trials_with_uplift"] > 1:
            text += f"; {_uplift_span(group)}"
        text += ")"
        if spread := group.get("mean_uplift_range_pct"):
            text += f" – {fmt_pct(spread[0])} to {fmt_pct(spread[1])} given conflicting source values"
        if excluded := group.get("excluded_from_uplift"):
            text += "; excluded: " + ", ".join(f"{item['trial_id']} ({item['reason']})" for item in excluded)
        lines.append(text)
    return "\n".join(lines)


def _list(search: dict[str, Any], analysis: QueryAnalysis, *, related: list[dict[str, Any]] = ()) -> str:
    if search.get("unresolved_terms"):
        return _unresolved(search)
    filters = search.get("filters", {})
    trials = search.get("trials", [])
    if not trials:
        return _no_match(filters, list(related)) if related else _none_found(filters)
    blocks = [
        f"**{plural(search['count'], 'trial')}{_trial_phrase(filters)}**:",
        "\n".join(_trial_line(t, with_what=True) for t in trials),
    ]
    if caveats := _caveat_lines(trials):
        blocks.append("**Caveats**\n" + "\n".join(caveats))
    return "\n\n".join(blocks)


def _issues(issues: dict[str, Any]) -> str:
    conflicts = [c for c in issues.get("conflicts", []) if c.get("status") == "open"]
    resolved = [c for c in issues.get("conflicts", []) if c.get("status") != "open"]
    missing = issues.get("missing_values", [])
    single = issues.get("single_source_trials", [])

    blocks = [
        f"**Data quality across {plural(issues.get('trial_count', 0), 'trial')}**: "
        f"{plural(len(conflicts), 'open conflict')}, {plural(len(missing), 'trial')} with missing values and "
        f"{plural(len(single), 'single-source trial')}."
    ]
    if conflicts:
        lines = []
        for conflict in conflicts:
            line = f"- {conflict['trial_id']} ({conflict.get('trial', '')}) – {conflict['field']}: " + " vs ".join(
                conflict.get("values", [])
            )
            if in_use := conflict.get("value_in_use"):
                line += f". Value currently shown: {in_use}"
            if spread := conflict.get("uplift_range_pct"):
                line += f". Uplift is {fmt_pct(spread[0])} to {fmt_pct(spread[1])} depending on which value is right"
            line += "."
            for note in conflict.get("notes", []):
                line += f'\n  - Note: "{note}"'
            lines.append(line)
        blocks.append("**Conflicting values**\n" + "\n".join(lines))
    if resolved:
        blocks.append(
            "**Resolved conflicts**\n"
            + "\n".join(f"- {c['trial_id']} – {c['field']}: using {c.get('value_in_use')}" for c in resolved)
        )
    if missing:
        lines = []
        for item in missing:
            labels = ", ".join(_MISSING_LABELS.get(name, name.replace("_", " ")) for name in item["missing"])
            line = f"- {item['trial_id']}: {labels} missing ({', '.join(item['sources'])})"
            if any(name in _MISSING_LABELS for name in item["missing"]):
                line += ", so its uplift can't be calculated"
            lines.append(line + ".")
        blocks.append("**Missing values**\n" + "\n".join(lines))
    if single:
        blocks.append(
            "**Single-source trials**\n" + "\n".join(f"- {item['trial_id']} ({item['source']})" for item in single)
        )
    if counts := issues.get("caveat_counts"):
        structural = ("Conflicting", "Control yield missing", "Treated yield missing", "Reported by a single source")
        lines = [
            f"- {caveat}: {plural(count, 'trial')}" for caveat, count in counts.items() if not caveat.startswith(structural)
        ]
        if lines:
            blocks.append("**Other caveats**\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _compare(comparison: dict[str, Any]) -> str:
    trials = comparison.get("trials", [])
    if len(trials) < 2:
        missing = comparison.get("not_found") or []
        text = comparison.get("note", "I need at least two existing trials to compare.")
        return f"{text} Not found: {', '.join(missing)}." if missing else text

    names = _and_join([trial["id"] for trial in trials])
    differ = [label.lower() for label in comparison.get("fields_that_differ", [])]
    shared = [f"{label.lower()} {value}" for label, value in comparison.get("shared_values", {}).items()]
    headline = f"**Comparing {names}**"
    if differ:
        headline += f": they differ in {_and_join(differ)}"
    if shared:
        headline += (", and share " if differ else ": they share ") + _and_join(shared)
    blocks = [headline + ".", "\n".join(_trial_line(trial, with_what=True) for trial in trials)]

    ranking = comparison.get("ranking_by_uplift", [])
    if ranking:
        # Trials with the same uplift share a place: "T01 and T03 (+5.0% each), then T04 (0.0%)".
        places = []
        for uplift, tied in groupby(ranking, key=lambda item: item["uplift_pct"]):
            ids = [item["trial_id"] for item in tied]
            places.append(f"{_and_join(ids)} ({fmt_pct(uplift)}{' each' if len(ids) > 1 else ''})")
        blocks.append("Ranking by uplift: " + ", then ".join(places) + ".")
    if warnings := comparison.get("warnings"):
        blocks.append("**Watch out**\n" + "\n".join(f"- {warning}" for warning in warnings))
    if caveats := _caveat_lines(trials):
        blocks.append("**Caveats**\n" + "\n".join(caveats))
    return "\n\n".join(blocks)


def _detail(detail: dict[str, Any]) -> str:
    if "error" in detail:
        known = ", ".join(detail.get("known_trial_ids", []))
        return f"{detail['error']} Known trials: {known}."

    title = f"**{detail['id']} – {_what(detail)}{_where(detail)}**"
    lines = [f"- Yields: {_yields(detail)}", f"- Data status: {detail.get('status', 'unknown').replace('_', ' ')}"]
    for conflict in detail.get("conflicts", []):
        values = " vs ".join(conflict.get("values", []))
        state = "⚠ Sources conflict" if conflict.get("status") == "open" else "Resolved conflict"
        line = f"- {state} on {conflict['field'].lower()}: {values}"
        if in_use := conflict.get("value_in_use"):
            line += f" (value shown: {in_use})"
        lines.append(line)
    for source in detail.get("sources", []):
        reported = source.get("reported_values", {})
        values = ", ".join(f"{CORE_FIELD_LABELS.get(k, k).lower()} {v}" for k, v in reported.items() if "yield" in k)
        locator = f", {source['locator']}" if source.get("locator") else ""
        lines.append(f"- Source {source['file']}{locator}" + (f": {values}" if values else ""))
    for note in detail.get("notes", []):
        lines.append(f"- Note from {note}")
    caveats = [c for c in detail.get("caveats", []) if not c.startswith(_INLINE_CAVEAT_PREFIXES)]
    if caveats:
        lines.append(f"- Caveats: {'; '.join(caveats)}")
    if custom := detail.get("custom_fields"):
        lines.append("- " + "; ".join(f"{label}: {value}" for label, value in custom.items()))
    return f"{title}\n\n" + "\n".join(lines)


def _reference(reference: dict[str, Any] | None) -> str:
    if not reference:
        return ""

    def names(key: str) -> str:
        return ", ".join(f"{row['name']} ({row['trials']})" for row in reference.get(key, [])) or "none"

    ids = reference.get("trial_ids", [])
    span = f" ({ids[0]}–{ids[-1]})" if len(ids) > 1 else ""
    lines = [
        f"The dataset has **{plural(reference.get('trial_count', 0), 'trial')}**{span}, with yields in "
        f"{reference.get('unit', 't/ha')}.",
        f"- Products: {names('products')}",
        f"- Crops: {names('crops')}",
        f"- Countries: {names('countries')}",
        f"- Years: {names('years')}",
        f"- Trial types: {names('trial_types')}",
    ]
    if status := reference.get("status_counts"):
        lines.append("- Data status: " + ", ".join(f"{name.replace('_', ' ')} {count}" for name, count in status.items()))
    if custom := reference.get("custom_fields"):
        lines.append("- Custom fields: " + ", ".join(field["label"] for field in custom))
    return "\n".join(lines)


def _help(store: EvidenceStore) -> str:
    example_product = next(iter(store.values_in_data("product")), "a product")
    example_ids = store.trial_ids[:2]
    compare = f"Compare {example_ids[0]} and {example_ids[1]}" if len(example_ids) == 2 else "Compare two trials"
    return (
        f"I answer questions about the {len(store.trials)} trial records, citing the trials and source files I use. "
        "For example:\n"
        f'- "What is the average yield uplift of {example_product}?"\n'
        '- "Which trials have conflicting data?"\n'
        f'- "{compare}"\n'
        '- "Which product performs best?"'
    )
