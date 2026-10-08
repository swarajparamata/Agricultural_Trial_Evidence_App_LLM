"""Deterministic checks of a draft answer against the tool results.

They drive the agent's reflection loop: when a check fails, the issues go back to the model as reviewer
feedback and it revises the answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .schemas import DatasetTrial
from .store import EvidenceStore, ngram_keys, norm_key

# A number not glued to a word (so 'T03' and 'trials_a' don't count), with optional thousands separators.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])([-+−]?)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)(?![0-9])")
_UNIT_AFTER = re.compile(r"\s?(%|t/ha|kg/ha|q/ha|t\b|kg\b|tonnes?\b|percent)", re.IGNORECASE)

_CONFLICT_WORDS = re.compile(
    r"conflict|discrepan|inconsisten|disagree|differ|mismatch|neither source|not final|two values|uncertain",
    re.IGNORECASE,
)
_MISSING_WORDS = re.compile(
    r"missing|not available|unavailable|not recorded|not reported|no control|no untreated|without|lacks?|absent|"
    r"cannot|can't|could not|not possible|n/a|excluded|blank|empty",
    re.IGNORECASE,
)
_DOMAIN_WORDS = re.compile(
    r"trial|yield|uplift|product|crop|countr|data|evidence|treat|control|compar|average|mean|conflict|source|"
    r"result|perform|effect|increase|scientific|demonstration|t/ha|kg/ha|harvest|boost",
    re.IGNORECASE,
)


@dataclass
class VerificationResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    skipped: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "issues": self.issues, "skipped": self.skipped}


@dataclass(frozen=True)
class _Claim:
    value: float
    decimals: int
    text: str


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _checkable_numbers(answer: str) -> list[_Claim]:
    """Numbers that state data: decimals, percentages, yields, and integers too large to be counts or ordinals."""
    claims = []
    for match in _NUMBER.finditer(answer):
        sign, digits = match.groups()
        value = _to_float(digits) * (-1 if sign in ("-", "−") else 1)
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        unit = _UNIT_AFTER.match(answer, match.end())
        if decimals == 0 and not unit:
            if abs(value) <= 12 or 1900 <= abs(value) <= 2100:
                continue
        text = match.group(0).strip() + (unit.group(0) if unit else "")
        claims.append(_Claim(value=value, decimals=decimals, text=text))
    return claims


def _walk_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, str):
        for match in _NUMBER.finditer(value):
            sign, digits = match.groups()
            yield _to_float(digits) * (-1 if sign in ("-", "−") else 1)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _walk_numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_numbers(item)


def evidence_numbers(observations: list[dict[str, Any]]) -> set[float]:
    """Every number the tools returned, plus unit-converted (x1000, /1000) and absolute variants."""
    numbers: set[float] = set()
    for observation in observations:
        try:
            data = json.loads(observation.get("output", ""))
        except (TypeError, ValueError):
            continue
        for number in _walk_numbers(data):
            numbers.update({number, abs(number), number * 1000, number / 1000})
    return numbers


def _grounded(claim: _Claim, evidence: set[float]) -> bool:
    # The answer's own precision sets the allowed rounding: '6%' covers 6.3, '6.33' must be within 0.005.
    tolerance = 0.5 * 10 ** (-claim.decimals) + 1e-9
    for number in evidence:
        if abs(number - claim.value) <= max(tolerance, abs(number) * 0.005):
            return True
        if abs(number - abs(claim.value)) <= max(tolerance, abs(number) * 0.005):
            return True
    return False


def _sentences_mentioning(answer: str, trial_id: str) -> list[str]:
    chunks = [chunk for line in answer.splitlines() for chunk in re.split(r"(?<=[.!?])\s+", line)]
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(trial_id)}(?![A-Za-z0-9])", re.IGNORECASE)
    return [chunk for chunk in chunks if pattern.search(chunk)]


def _all_numbers(text: str) -> set[float]:
    numbers = set()
    for match in _NUMBER.finditer(text):
        value = _to_float(match.group(2))
        numbers.update({value, value * 1000, value / 1000})
    return numbers


def _discloses_conflicts(trial: DatasetTrial, answer: str) -> bool:
    """A conflict word next to the trial ID, or every conflicting value, counts as disclosure."""
    if _CONFLICT_WORDS.search(" ".join(_sentences_mentioning(answer, trial.id))):
        return True
    numbers = _all_numbers(answer)
    for conflict in trial.conflicts:
        if conflict.status != "open":
            continue
        for candidate in conflict.candidates:
            match = _NUMBER.search(candidate.display)
            if match is None:
                if norm_key(candidate.display) not in norm_key(answer):
                    return False
            elif not any(abs(number - _to_float(match.group(2))) < 1e-6 for number in numbers):
                return False
    return True


def is_data_question(question: str, store: EvidenceStore) -> bool:
    if _DOMAIN_WORDS.search(question) or store.mentioned_trial_ids(question):
        return True
    known = {key for name in ("crop", "product", "country") for key in store.lookup_table(name)}
    return any(key in known for key, _ in ngram_keys(question))


def verify_answer(
    answer: str, question: str, observations: list[dict[str, Any]], store: EvidenceStore
) -> VerificationResult:
    """Returns the problems a reviewer would point out; an empty list means the answer is grounded."""
    issues: list[str] = []
    text = answer.strip()
    if not text:
        return VerificationResult(passed=False, issues=["The answer is empty. Write a complete answer."])

    data_question = is_data_question(question, store)
    if data_question and not observations:
        issues.append(
            "You answered without looking at the data. Call the tools (e.g. search_trials or trial_statistics) "
            "and base the answer on their results."
        )

    if unknown := store.unknown_trial_ids(text):
        issues.append(
            f"The answer mentions trial ID(s) that do not exist: {', '.join(unknown)}. "
            f"Known trials: {', '.join(store.trial_ids)}."
        )

    if observations:
        evidence = evidence_numbers(observations)
        unsupported = []
        for claim in _checkable_numbers(text):
            if not _grounded(claim, evidence) and claim.text not in unsupported:
                unsupported.append(claim.text)
        if unsupported:
            issues.append(
                f"These numbers do not appear in the tool results: {', '.join(unsupported[:6])}. Use only numbers "
                "returned by the tools (call trial_statistics for averages) or remove them."
            )

    mentioned = [trial for trial_id in store.mentioned_trial_ids(text) if (trial := store.get(trial_id))]
    for trial in mentioned:
        if any(conflict.status == "open" for conflict in trial.conflicts) and not _discloses_conflicts(trial, text):
            details = "; ".join(
                f"{c.label or c.field}: " + " vs ".join(f"{cand.display} ({', '.join(cand.sources)})" for cand in c.candidates)
                for c in trial.conflicts
                if c.status == "open"
            )
            issues.append(f"{trial.id} has conflicting source values ({details}). Say so and give both values.")
        if "control_yield" in trial.missing_fields:
            sentences = " ".join(_sentences_mentioning(text, trial.id))
            if not _MISSING_WORDS.search(sentences):
                issues.append(f"{trial.id} has no control yield. Say that its uplift cannot be calculated.")

    if data_question and observations and not mentioned:
        cited_sources = any(source.file in text for trial in store.trials for source in trial.sources)
        if _checkable_numbers(text) and not cited_sources:
            issues.append("Cite the trial IDs (and source files) behind the numbers in the answer.")

    issues += _misattributed_caveats(text, store)
    return VerificationResult(passed=not issues, issues=issues)


_NEGATED = re.compile(r"\b(no|not|non|none|without)\b", re.IGNORECASE)
_SINGLE_SOURCE = re.compile(r"single[- ]source", re.IGNORECASE)
_CONFLICT = re.compile(r"\bconflict", re.IGNORECASE)


def _trial_blocks(text: str, store: EvidenceStore) -> list[tuple[DatasetTrial, str]]:
    """Lines that talk about exactly one trial: its own line plus the indented lines under it."""
    blocks: list[tuple[DatasetTrial, str]] = []
    current: DatasetTrial | None = None
    for line in text.splitlines():
        ids = store.mentioned_trial_ids(line)
        if len(ids) == 1:
            current = store.get(ids[0])
        elif ids or not line.startswith((" ", "\t")):
            current = None  # several trials, or a new top-level line: no single owner
        if current is not None and line.strip():
            blocks.append((current, line))
    return blocks


def _misattributed_caveats(text: str, store: EvidenceStore) -> list[str]:
    """Caveats the answer gives a trial that the data does not record for it (small models invent them)."""
    labels = sorted({caveat for trial in store.trials for caveat in trial.caveats}, key=len, reverse=True)
    wrong: dict[str, list[str]] = {}
    for trial, line in _trial_blocks(text, store):
        own = {norm_key(caveat) for caveat in trial.caveats}
        line_key = norm_key(line)
        claimed = [label for label in labels if norm_key(label) in line_key and norm_key(label) not in own]
        if _SINGLE_SOURCE.search(line) and trial.status != "single_source" and not _NEGATED.search(line):
            if not any("singlesource" in key for key in own):
                claimed.append("single source")
        if _CONFLICT.search(line) and not trial.conflicts and not _NEGATED.search(line):
            claimed.append("conflicting values")
        for label in claimed:
            if label not in wrong.setdefault(trial.id, []):
                wrong[trial.id].append(label)
    issues = []
    for trial_id, claimed in wrong.items():
        trial = store.get(trial_id)
        recorded = "; ".join(trial.caveats) if trial and trial.caveats else "none"
        issues.append(
            f"{trial_id} does not have these caveats: {', '.join(claimed)}. Its recorded caveats are: {recorded}. "
            "Give each trial only the caveats the tool results list for it."
        )

    # Blanket claims in lines that name no trial ("some trials are single-source") must fit the trials cited.
    cited = [trial for trial_id in store.mentioned_trial_ids(text) if (trial := store.get(trial_id))]
    general = [line for line in text.splitlines() if not store.mentioned_trial_ids(line) and not _NEGATED.search(line)]
    if cited and not any(_single_source(trial) for trial in cited) and any(_SINGLE_SOURCE.search(line) for line in general):
        issues.append("None of the trials in the answer is single-source. Remove that claim.")
    if cited and not any(trial.conflicts for trial in cited) and any(_CONFLICT.search(line) for line in general):
        issues.append("None of the trials in the answer has conflicting sources. Remove that claim.")
    return issues


def _single_source(trial: DatasetTrial) -> bool:
    return trial.status == "single_source" or any("single source" in caveat.lower() for caveat in trial.caveats)
