"""Read-only queries over the dataset snapshot. Every tool the agent calls is answered from here."""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .schemas import Dataset, DatasetTrial, TrialConflict

FILTER_FIELDS = ("crop", "product", "country", "trial_type")
GROUP_FIELDS = ("product", "crop", "country", "year", "trial_type")
STATUSES = ("consistent", "single_source", "incomplete", "conflict", "manual")

CORE_FIELD_LABELS = {
    "crop": "Crop",
    "product": "Product",
    "country": "Country",
    "year": "Year",
    "trial_type": "Trial type",
    "treatment_yield": "Treated yield",
    "control_yield": "Control yield",
    "yield_difference": "Difference",
    "uplift_pct": "Uplift",
}

# Placeholders models sometimes send for optional tool arguments instead of omitting them.
_EMPTY_ARG_VALUES = {"", "none", "null", "any", "all", "*", "n/a", "na"}

_STATUS_SYNONYMS = {
    "conflicting": "conflict",
    "conflicts": "conflict",
    "missing": "incomplete",
    "single": "single_source",
    "singlesourced": "single_source",
    "verified": "consistent",
    "agreeing": "consistent",
    "manualentry": "manual",
}


def norm_key(value: object) -> str:
    """Case-, space- and punctuation-insensitive key: 'Harvest Plus', 'HarvestPlus' and 'HARVESTPLUS' match."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def clean_arg(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in _EMPTY_ARG_VALUES:
        return None
    return value


def filter_terms(value: Any) -> list[str]:
    """A filter argument as a list of terms: one value, a list, or a comma-separated string ('France, Germany')."""
    items = value if isinstance(value, (list, tuple, set)) else [value]
    terms: list[str] = []
    for item in items:
        if clean_arg(item) is None:
            continue
        for part in str(item).split(",") if isinstance(item, str) else [str(item)]:
            if clean_arg(part.strip()) is not None and part.strip() not in terms:
                terms.append(part.strip())
    return terms


def ngram_keys(text: str, max_words: int = 3) -> list[tuple[str, str]]:
    """(normalized key, original words) for every run of 1-3 words, so 'harvest plus' is found as 'harvestplus'."""
    words = re.findall(r"[A-Za-z0-9]+", text)
    grams = []
    for size in range(max_words, 0, -1):
        for start in range(len(words) - size + 1):
            original = " ".join(words[start : start + size])
            grams.append((norm_key(original), original))
    return grams


def resolve_status(term: str) -> str | None:
    key = norm_key(term)
    for status in STATUSES:
        if norm_key(status) == key:
            return status
    return _STATUS_SYNONYMS.get(key)


def fmt_num(value: float | None, decimals: int = 2) -> str:
    """12.50 -> '12.5', 45.0 -> '45', 7800.0 -> '7,800'."""
    if value is None:
        return "n/a"
    text = f"{value:,.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def fmt_pct(value: float | None) -> str:
    """+5.0%, -2.5%, 0.0%."""
    if value is None:
        return "n/a"
    rounded = round(value, 1)
    return "0.0%" if rounded == 0 else f"{rounded:+.1f}%"


def natural_key(trial_id: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", trial_id))


def conflict_text(conflict: TrialConflict) -> str:
    """'Treated yield: 42 t/ha (trials_a.csv) vs 44 t/ha (T03_summary.txt)'."""
    values = " vs ".join(f"{c.display} ({', '.join(c.sources)})" for c in conflict.candidates)
    return f"{conflict.label or conflict.field}: {values}"


def _compact(data: dict[str, Any]) -> dict[str, Any]:
    """Drops empty collections so tool outputs stay small for the model."""
    return {key: value for key, value in data.items() if value not in ([], {}, "")}


@dataclass
class FilterResult:
    trials: list[DatasetTrial]
    applied: dict[str, Any] = field(default_factory=dict)
    unresolved: dict[str, str] = field(default_factory=dict)


class EvidenceStore:
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.unit = dataset.display_unit
        self.trials = sorted(dataset.trials, key=lambda trial: natural_key(trial.id))
        self._by_id = {norm_key(trial.id): trial for trial in self.trials}
        self._lookups = {name: self._build_lookup(name) for name in FILTER_FIELDS}
        self.custom_fields = {ref.key: ref for ref in dataset.reference.custom_fields}
        id_prefixes = {match.group(1) for trial in self.trials if (match := re.match(r"([A-Za-z]+)\d", trial.id))}
        self._id_prefixes = {prefix.upper() for prefix in id_prefixes}

    # --- vocabulary -------------------------------------------------------------------------

    def _reference_entries(self, name: str):
        ref = self.dataset.reference
        return {"crop": ref.crops, "product": ref.products, "country": ref.countries, "trial_type": ref.trial_types}[name]

    def _build_lookup(self, name: str) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for entry in self._reference_entries(name):
            for alias in (entry.name, *entry.aliases, entry.code or ""):
                if key := norm_key(alias):
                    lookup.setdefault(key, entry.name)
        for trial in self.trials:
            if value := getattr(trial, name):
                lookup.setdefault(norm_key(value), value)
        return lookup

    def resolve(self, name: str, term: str | None) -> str | None:
        """Canonical value for a user-supplied spelling ('potatoes' -> 'Potato', 'FR' -> 'France')."""
        key = norm_key(term) if term is not None else ""
        if not key:
            return None
        lookup = self._lookups[name]
        if key in lookup:
            return lookup[key]
        for suffix in ("es", "s"):
            if key.endswith(suffix) and key[: -len(suffix)] in lookup:
                return lookup[key[: -len(suffix)]]
        return None

    def lookup_table(self, name: str) -> dict[str, str]:
        """Normalized spelling -> canonical value, used by the offline planner to spot entities in questions."""
        return dict(self._lookups[name])

    def values_in_data(self, name: str) -> list[str]:
        values = {str(getattr(trial, name)) for trial in self.trials if getattr(trial, name) is not None}
        return sorted(values, key=natural_key)

    def get(self, trial_id: str) -> DatasetTrial | None:
        return self._by_id.get(norm_key(trial_id))

    @property
    def trial_ids(self) -> list[str]:
        return [trial.id for trial in self.trials]

    def mentioned_trial_ids(self, text: str) -> list[str]:
        """Known trial IDs in the order they first appear in `text`."""
        positions = []
        for trial in self.trials:
            match = re.search(rf"(?<![A-Za-z0-9]){re.escape(trial.id)}(?![A-Za-z0-9])", text, re.IGNORECASE)
            if match:
                positions.append((match.start(), trial.id))
        return [trial_id for _, trial_id in sorted(positions)]

    def unknown_trial_ids(self, text: str) -> list[str]:
        """Tokens that look like trial IDs of this dataset (same letter prefix) but don't exist, e.g. 'T07'."""
        unknown = []
        for match in re.finditer(r"(?<![A-Za-z0-9])([A-Z]{1,4})(\d{1,5})(?![A-Za-z0-9])", text):
            token = match.group(0)
            if match.group(1) in self._id_prefixes and self.get(token) is None and token not in unknown:
                unknown.append(token)
        return unknown

    # --- filtering --------------------------------------------------------------------------

    def apply_filters(
        self,
        *,
        crop: str | list[str] | None = None,
        product: str | list[str] | None = None,
        country: str | list[str] | None = None,
        year: int | str | list[int | str] | None = None,
        year_from: int | str | None = None,
        year_to: int | str | None = None,
        trial_type: str | list[str] | None = None,
        status: str | None = None,
        text: str | None = None,
        trial_ids: list[str] | None = None,
    ) -> FilterResult:
        """Each vocabulary field takes one value or several (any of them matches); years take a list or a range."""
        result = FilterResult(trials=list(self.trials))
        for name, term in (("crop", crop), ("product", product), ("country", country), ("trial_type", trial_type)):
            terms = filter_terms(term)
            if not terms:
                continue
            canonical: list[str] = []
            unknown: list[str] = []
            for item in terms:
                value = self.resolve(name, item)
                if value is None:
                    unknown.append(item)
                elif value not in canonical:
                    canonical.append(value)
            if unknown:
                result.unresolved[name] = ", ".join(unknown)
            if not canonical:
                result.trials = []
                continue
            result.applied[name] = canonical[0] if len(canonical) == 1 else canonical
            keys = {norm_key(value) for value in canonical}
            result.trials = [trial for trial in result.trials if norm_key(getattr(trial, name) or "") in keys]

        year_terms = filter_terms(year)
        if year_terms:
            years = sorted({int(match.group()) for item in year_terms if (match := re.search(r"\d{4}", item))})
            if not years:
                result.unresolved["year"] = ", ".join(year_terms)
                result.trials = []
            else:
                result.applied["year"] = years[0] if len(years) == 1 else years
                result.trials = [trial for trial in result.trials if trial.year in years]
        for bound, value in (("year_from", year_from), ("year_to", year_to)):
            value = clean_arg(value)
            if value is None:
                continue
            match = re.search(r"\d{4}", str(value))
            if match is None:
                result.unresolved[bound] = str(value)
                result.trials = []
                continue
            limit = result.applied[bound] = int(match.group())
            result.trials = [
                trial
                for trial in result.trials
                if trial.year is not None and (trial.year >= limit if bound == "year_from" else trial.year <= limit)
            ]

        status = clean_arg(status)
        if status is not None:
            canonical_status = resolve_status(str(status))
            if canonical_status is None:
                result.unresolved["status"] = str(status)
                result.trials = []
            else:
                result.applied["status"] = canonical_status
                result.trials = [trial for trial in result.trials if trial.status == canonical_status]

        text = clean_arg(text)
        if text:
            needle = str(text).lower()
            result.applied["text"] = str(text)
            result.trials = [trial for trial in result.trials if needle in self._haystack(trial)]

        if trial_ids:
            wanted_ids = {norm_key(trial_id) for trial_id in trial_ids}
            result.applied["trial_ids"] = list(trial_ids)
            result.trials = [trial for trial in result.trials if norm_key(trial.id) in wanted_ids]
        return result

    def _haystack(self, trial: DatasetTrial) -> str:
        parts = [trial.id, trial.crop, trial.product, trial.country, trial.trial_type, str(trial.year or "")]
        parts += trial.caveats + [note.text for note in trial.notes] + [source.file for source in trial.sources]
        parts += [str(value) for value in trial.custom_fields.values()]
        return " ".join(part for part in parts if part).lower()

    def suggestions(self, unresolved: dict[str, str]) -> dict[str, list[str]]:
        """Known values for each field the caller used an unknown term for."""
        return {name: self.values_in_data(name) for name in unresolved if name in GROUP_FIELDS}

    # --- views ------------------------------------------------------------------------------

    def custom_values(self, trial: DatasetTrial) -> dict[str, Any]:
        values = {}
        for key, value in trial.custom_fields.items():
            if value in (None, ""):
                continue
            ref = self.custom_fields.get(key)
            label = ref.label if ref else key
            values[label] = f"{value} {ref.unit}" if ref and ref.unit and ref.type == "number" else value
        return values

    def summary(self, trial: DatasetTrial) -> dict[str, Any]:
        data = _compact(
            {
                "id": trial.id,
                "crop": trial.crop,
                "product": trial.product,
                "country": trial.country,
                "year": trial.year,
                "trial_type": trial.trial_type,
                "treatment_yield": trial.treatment_yield,
                "control_yield": trial.control_yield,
                "unit": trial.yield_unit,
                "difference": trial.yield_difference,
                "uplift_pct": trial.uplift_pct,
                "uplift_range_pct": trial.uplift_range or [],
                "status": trial.status,
                "missing_fields": trial.missing_fields,
                "open_conflicts": [conflict_text(c) for c in trial.conflicts if c.status == "open"],
                "caveats": trial.caveats,
                "sources": [source.file for source in trial.sources],
                "custom_fields": self.custom_values(trial),
            }
        )
        # Always present, even when empty: a missing list invites a model to fill it with made-up caveats.
        data["caveats"] = list(trial.caveats)
        return data

    def detail(self, trial: DatasetTrial) -> dict[str, Any]:
        data = self.summary(trial)
        data["sources"] = [
            _compact({"file": s.file, "kind": s.kind, "locator": s.locator, "reported_values": s.values})
            for s in trial.sources
        ]
        data["conflicts"] = [
            _compact(
                {
                    "field": c.label or c.field,
                    "status": c.status,
                    "values": [f"{cand.display} ({', '.join(cand.sources)})" for cand in c.candidates],
                    "value_in_use": c.chosen_display,
                }
            )
            for c in trial.conflicts
        ]
        data["notes"] = [f"{note.source}: {note.text}" for note in trial.notes]
        data.pop("open_conflicts", None)
        return _compact(data)

    # --- tool queries -----------------------------------------------------------------------

    def search(self, *, limit: int = 25, **filters: Any) -> dict[str, Any]:
        result = self.apply_filters(**filters)
        output: dict[str, Any] = {
            "count": len(result.trials),
            "filters": result.applied,
            "unit": self.unit,
            "trials": [self.summary(trial) for trial in result.trials[:limit]],
        }
        if len(result.trials) > limit:
            output["truncated"] = f"showing {limit} of {len(result.trials)}"
        self._add_unresolved(output, result)
        output.update(self._found_separately(result))
        return output

    def trial_detail(self, trial_id: str) -> dict[str, Any]:
        trial = self.get(trial_id)
        if trial is None:
            return {"error": f"No trial with ID '{trial_id}'.", "known_trial_ids": self.trial_ids}
        return self.detail(trial)

    def statistics(self, *, group_by: str | None = None, **filters: Any) -> dict[str, Any]:
        result = self.apply_filters(**filters)
        output: dict[str, Any] = {"filters": result.applied, "unit": self.unit, **self._aggregate(result.trials)}
        self._add_unresolved(output, result)
        output.update(self._found_separately(result))

        group_by = clean_arg(group_by)
        if group_by:
            if group_by not in GROUP_FIELDS:
                output["group_by_error"] = f"Unknown group_by '{group_by}'. Use one of: {', '.join(GROUP_FIELDS)}."
            else:
                groups: dict[str, list[DatasetTrial]] = defaultdict(list)
                for trial in result.trials:
                    groups[str(getattr(trial, group_by) or "Unknown")].append(trial)
                rows = [{"group": name, **self._aggregate(trials, brief=True)} for name, trials in groups.items()]
                rows.sort(key=lambda row: (row.get("mean_uplift_pct") is None, -(row.get("mean_uplift_pct") or 0)))
                output["group_by"] = group_by
                output["groups"] = rows
        return output

    def _aggregate(self, trials: list[DatasetTrial], *, brief: bool = False) -> dict[str, Any]:
        with_uplift = [trial for trial in trials if trial.uplift_pct is not None]
        output: dict[str, Any] = {
            "trial_count": len(trials),
            "trial_ids": [trial.id for trial in trials],
            "trials_with_uplift": len(with_uplift),
        }
        if with_uplift:
            uplifts = [trial.uplift_pct for trial in with_uplift]
            output["mean_uplift_pct"] = round(statistics.fmean(uplifts), 1)
            if len(uplifts) > 2:
                output["median_uplift_pct"] = round(statistics.median(uplifts), 1)
            # Every trial at the extreme, so a tie doesn't hide behind whichever trial came first.
            for key, extreme in (("max_uplift", max(uplifts)), ("min_uplift", min(uplifts))):
                output[key] = {
                    "trial_ids": [trial.id for trial in with_uplift if trial.uplift_pct == extreme],
                    "uplift_pct": extreme,
                }
            # How the evidence splits, for questions such as "what supports this claim?".
            output["trials_with_increase"] = sum(1 for value in uplifts if value > 0)
            output["trials_without_difference"] = sum(1 for value in uplifts if value == 0)
            output["trials_with_decrease"] = sum(1 for value in uplifts if value < 0)
            if any(trial.uplift_range for trial in with_uplift):
                lows = [trial.uplift_range[0] if trial.uplift_range else trial.uplift_pct for trial in with_uplift]
                highs = [trial.uplift_range[1] if trial.uplift_range else trial.uplift_pct for trial in with_uplift]
                output["mean_uplift_range_pct"] = [round(statistics.fmean(lows), 1), round(statistics.fmean(highs), 1)]
            crops = {trial.crop for trial in with_uplift}
            differences = [trial.yield_difference for trial in with_uplift if trial.yield_difference is not None]
            # Absolute differences only make sense within one crop.
            if len(crops) == 1 and differences:
                output["mean_difference"] = round(statistics.fmean(differences), 3)
            if not brief:
                output["per_trial"] = [
                    _compact(
                        {
                            "trial_id": trial.id,
                            "uplift_pct": trial.uplift_pct,
                            "uplift_range_pct": trial.uplift_range or [],
                            "difference": trial.yield_difference,
                            "treatment_yield": trial.treatment_yield,
                            "control_yield": trial.control_yield,
                        }
                    )
                    for trial in with_uplift
                ]
        excluded = [
            {"trial_id": trial.id, "reason": self.no_uplift_reason(trial)} for trial in trials if trial.uplift_pct is None
        ]
        if excluded:
            output["excluded_from_uplift"] = excluded
        conflicted = [trial.id for trial in trials if any(c.status == "open" for c in trial.conflicts)]
        if conflicted:
            output["trials_with_open_conflicts"] = conflicted
        crops = sorted({trial.crop for trial in trials if trial.crop})
        if len(crops) > 1 and not brief:
            output["note"] = (
                f"Trials span several crops ({', '.join(crops)}); compare uplift % rather than absolute yields."
            )
        return output

    @staticmethod
    def no_uplift_reason(trial: DatasetTrial) -> str:
        if trial.treatment_yield is None:
            return "treated yield missing"
        if trial.control_yield is None:
            return "control yield missing"
        if trial.control_yield == 0:
            return "control yield is zero"
        return "uplift unavailable"

    def compare(self, trial_ids: list[str]) -> dict[str, Any]:
        found = [trial for trial_id in trial_ids if (trial := self.get(trial_id)) is not None]
        missing = [trial_id for trial_id in trial_ids if self.get(trial_id) is None]
        output: dict[str, Any] = {"unit": self.unit, "trials": [self.summary(trial) for trial in found]}
        if missing:
            output["not_found"] = missing
        if len(found) < 2:
            output["note"] = "Need at least two existing trials to compare."
            return output

        differing, shared = [], {}
        for name in ("product", "crop", "country", "year", "trial_type"):
            values = {getattr(trial, name) for trial in found}
            if len(values) > 1:
                differing.append(CORE_FIELD_LABELS[name])
            else:
                shared[CORE_FIELD_LABELS[name]] = next(iter(values))
        for key, ref in self.custom_fields.items():
            values = {str(trial.custom_fields.get(key)) for trial in found if trial.custom_fields.get(key) not in (None, "")}
            if len(values) > 1:
                differing.append(ref.label)
        output["fields_that_differ"] = differing
        output["shared_values"] = {label: value for label, value in shared.items() if value is not None}

        ranked = sorted((trial for trial in found if trial.uplift_pct is not None), key=lambda t: -t.uplift_pct)
        output["ranking_by_uplift"] = [{"trial_id": trial.id, "uplift_pct": trial.uplift_pct} for trial in ranked]

        warnings = []
        crops = sorted({trial.crop for trial in found if trial.crop})
        if len(crops) > 1:
            warnings.append(f"Different crops ({', '.join(crops)}): compare uplift % rather than absolute yields.")
        for trial in found:
            warnings += [f"{trial.id}: conflicting {conflict_text(c)}" for c in trial.conflicts if c.status == "open"]
            if trial.uplift_pct is None:
                warnings.append(f"{trial.id}: {self.no_uplift_reason(trial)}, so uplift can't be calculated.")
        if warnings:
            output["warnings"] = warnings
        return output

    def data_issues(self) -> dict[str, Any]:
        conflicts = [
            _compact(
                {
                    "trial_id": trial.id,
                    "trial": f"{trial.product} on {trial.crop}, {trial.country} {trial.year}",
                    "field": c.label or c.field,
                    "status": c.status,
                    "values": [f"{cand.display} ({', '.join(cand.sources)})" for cand in c.candidates],
                    "value_in_use": c.chosen_display,
                    "uplift_range_pct": trial.uplift_range or [],
                    "notes": [note.text for note in trial.notes],
                }
            )
            for trial in self.trials
            for c in trial.conflicts
        ]
        missing = [
            {
                "trial_id": trial.id,
                "missing": trial.missing_fields,
                "sources": [source.file for source in trial.sources],
            }
            for trial in self.trials
            if trial.missing_fields
        ]
        single_source = [
            {"trial_id": trial.id, "source": trial.sources[0].file}
            for trial in self.trials
            if len({source.file for source in trial.sources}) == 1 and trial.status != "manual"
        ]
        caveats = Counter(caveat for trial in self.trials for caveat in trial.caveats)
        return {
            "trial_count": len(self.trials),
            "status_counts": dict(Counter(trial.status for trial in self.trials)),
            "conflicts": conflicts,
            "missing_values": missing,
            "single_source_trials": single_source,
            "caveat_counts": dict(caveats.most_common()),
        }

    def reference_summary(self) -> dict[str, Any]:
        def counted(name: str) -> list[dict[str, Any]]:
            counts = Counter(getattr(trial, name) for trial in self.trials if getattr(trial, name) is not None)
            entries = {entry.name: entry for entry in self._reference_entries(name)} if name != "year" else {}
            rows = []
            for value, count in sorted(counts.items(), key=lambda item: natural_key(str(item[0]))):
                entry = entries.get(value)
                rows.append(
                    _compact(
                        {
                            "name": value,
                            "trials": count,
                            "code": entry.code if entry else None,
                            "aliases": entry.aliases if entry else [],
                        }
                    )
                )
            return rows

        unused = {
            name: sorted({entry.name for entry in self._reference_entries(name)} - set(self.values_in_data(name)))
            for name in FILTER_FIELDS
        }
        return _compact(
            {
                "trial_count": len(self.trials),
                "unit": self.unit,
                "trial_ids": self.trial_ids,
                "products": counted("product"),
                "crops": counted("crop"),
                "countries": counted("country"),
                "years": counted("year"),
                "trial_types": counted("trial_type"),
                "status_counts": dict(Counter(trial.status for trial in self.trials)),
                "custom_fields": [
                    _compact({"key": ref.key, "label": ref.label, "type": ref.type, "unit": ref.unit})
                    for ref in self.custom_fields.values()
                ],
                "vocabulary_without_trials": {name: values for name, values in unused.items() if values},
            }
        )

    def _found_separately(self, result: FilterResult) -> dict[str, Any]:
        """When a combination matches nothing (Harvest Plus on rice): what each of its filters finds on its own."""
        named = {name: value for name, value in result.applied.items() if name in FILTER_FIELDS}
        if result.trials or result.unresolved or len(named) < 2:
            return {}
        parts = {}
        for name, value in named.items():
            alone = self.apply_filters(**{name: value}).trials
            label = f"{name}={', '.join(value) if isinstance(value, list) else value}"
            parts[label] = _compact(
                {
                    "count": len(alone),
                    "crops": sorted({trial.crop for trial in alone if trial.crop}),
                    "products": sorted({trial.product for trial in alone if trial.product}),
                    "countries": sorted({trial.country for trial in alone if trial.country}),
                }
            )
        return {"found_separately": parts}

    def _add_unresolved(self, output: dict[str, Any], result: FilterResult) -> None:
        if result.unresolved:
            output["unresolved_terms"] = result.unresolved
            output["known_values"] = self.suggestions(result.unresolved)
