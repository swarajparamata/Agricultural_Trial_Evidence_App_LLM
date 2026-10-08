"""Rule-based reading of a question, used by the offline planner when no LLM is available."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .store import FILTER_FIELDS, EvidenceStore, ngram_keys, norm_key

Intent = Literal["help", "issues", "compare", "detail", "stats", "list", "reference", "overview"]

_HELP = re.compile(r"^\s*(hi|hello|hey|thanks|thank you|help|what can you do|who are you|good (morning|afternoon))\b", re.I)
_FOLLOW_UP = re.compile(r"^\s*(and|what about|how about|same for|also|what if)\b", re.I)

# Checked in order; the first match wins.
_INTENT_RULES: tuple[tuple[Intent, re.Pattern[str]], ...] = (
    (
        "issues",
        re.compile(
            r"conflict|discrepan|inconsisten|disagree|mismatch|data quality|quality issue|missing|incomplete|"
            r"\bgaps?\b|problem|reliab|trustworth|caveat|\bissues?\b|single[- ]source|contradict",
            re.I,
        ),
    ),
    ("compare", re.compile(r"compar|versus|\bvs\.?(?=\s)|difference between|side[- ]by[- ]side|head[- ]to[- ]head", re.I)),
    (
        "stats",
        re.compile(
            r"average|\bmean\b|uplift|increase|improv|effect|perform|benefit|\bgain|\bbest\b|worst|highest|lowest|"
            r"\btop\b|strongest|weakest|how much|summar|overall|evidence|impact|response|\bwork|rank|outperform",
            re.I,
        ),
    ),
    (
        "list",
        re.compile(
            r"\blist\b|\bshow\b|which trials|what trials|how many|\bcount\b|\bfind\b|trials (in|for|from|on|with)|"
            r"give me|\bdisplay\b|\bwhich\b.*\btrials?\b",
            re.I,
        ),
    ),
    (
        "reference",
        re.compile(r"(what|which) (products|crops|countries|years|trial types)|\bavailable\b|what data|what'?s in", re.I),
    ),
)

_GROUP_BY_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("product", re.compile(r"\b(by|per|each|across|which|what|best|top|worst|between|all)\s+(the\s+)?products?\b", re.I)),
    ("crop", re.compile(r"\b(by|per|each|across|which|what|best|top|worst|between|all)\s+(the\s+)?crops?\b", re.I)),
    (
        "country",
        re.compile(r"\b(by|per|each|across|which|what|best|top|worst|between|all)\s+(the\s+)?countr(y|ies)\b|\bwhere\b", re.I),
    ),
    ("year", re.compile(r"\b(by|per|each|across|which|what)\s+(the\s+)?(years?|seasons?)\b|over time|\btrend", re.I)),
    (
        "trial_type",
        re.compile(
            r"\b(by|per|each|across|which|what)\s+(the\s+)?(trial\s+)?types?\b|"
            r"scientific\s+(vs\.?|versus|or|and)\s+demonstration|demonstration\s+(vs\.?|versus|or|and)\s+scientific",
            re.I,
        ),
    ),
)

# Words that make a trial-type mention mean "trials of that type" rather than e.g. "scientific evidence".
_TRIAL_NOUNS = {"trial", "trials", "study", "studies", "plot", "plots", "site", "sites"}
# How far after a trial type the trial noun may come: "scientific wheat trials", "demonstration Harvest Plus trials".
_TRIAL_NOUN_WINDOW = 3

_YEAR = r"((?:19|20)\d{2})"
# "from 2021 through 2025", "between 2021 and 2023", "2021-2025", "2021 to 2025".
_YEAR_RANGE = re.compile(
    rf"\b(?:from|between)\s+{_YEAR}\s*(?:to|through|thru|until|till|and|-|–|—)\s*{_YEAR}\b"
    rf"|\b{_YEAR}\s*(?:-|–|—|to|through|thru|until|till)\s*{_YEAR}\b",
    re.I,
)
# "since 2022", "after 2021", "2021 onwards" (an open start) and "before 2024", "until 2023" (an open end).
# "from 2024" alone means trials of 2024, as in "trials from 2024".
_YEAR_FROM = re.compile(rf"\b(since|after|starting(?:\s+in)?)\s+{_YEAR}\b|\b{_YEAR}\s+(onwards|and later|or later)\b", re.I)
_YEAR_TO = re.compile(rf"\b(before|until|till|up to)\s+{_YEAR}\b|\b{_YEAR}\s+(and earlier|or earlier)\b", re.I)

# Questions about supporting a claim get a summary of what the evidence shows.
_EVIDENCE = re.compile(r"\bevidence\b|\bsupport|\bprove|\bclaims?\b|\bback(?:s|ed)?\s+up\b|justif|substantiat", re.I)

# Where a crop name sits in a question: "sorghum yields", "sorghum trials", "on sorghum". A word there that the
# vocabulary doesn't know is reported as an unknown crop instead of being dropped (which would answer about
# every crop and look like evidence for this one).
_CROP_SLOT = re.compile(r"\b([a-z]+)\s+(?:yields?|trials?|crops?|harvests?|production)\b|\bon\s+([a-z]+)\b", re.I)
_NOT_CROPS = {
    "a", "an", "the", "all", "any", "some", "these", "those", "this", "that", "its", "their", "our", "my", "his", "her",
    "more", "less", "higher", "lower", "highest", "lowest", "best", "worst", "better", "worse", "good", "poor", "big",
    "average", "mean", "median", "total", "overall", "typical", "expected", "actual", "relative", "absolute",
    "treated", "treatment", "control", "untreated", "field", "fields", "trial", "crop", "crops", "plot", "plots",
    "new", "old", "previous", "past", "recent", "existing", "other", "similar", "earlier", "later", "latest",
    "many", "few", "several", "much", "which", "what", "whose", "each", "every", "both", "same", "different",
    "reported", "recorded", "measured", "improved", "improving", "increased", "increasing", "boosting", "raising",
    "single", "source", "sourced", "conflicting", "incomplete", "complete", "consistent", "manual", "missing",
    "verified", "commercial", "of", "in", "for", "and", "or", "with", "per", "by", "to", "from", "at", "as", "on",
    "than", "is", "are", "was", "were", "be", "it", "them", "they", "how", "why", "when", "who", "where", "do", "does",
    "did", "year", "years", "season", "seasons", "country", "countries", "product", "products", "data", "evidence",
    "result", "results", "value", "values", "yield", "yields", "uplift", "uplifts", "current", "last", "first",
    "such", "only", "also", "just", "between", "across", "among", "per", "average", "trials", "test", "tests",
    # Commands and pronouns that start questions ("List trials in France", "Show me trials").
    "list", "show", "find", "give", "display", "compare", "get", "count", "see", "view", "fetch", "return", "search",
    "summarize", "summarise", "rank", "sort", "filter", "me", "us", "you", "your", "ran", "run", "conflict",
    # Nationalities read as countries, not crops ("German trials").
    "french", "german", "spanish", "italian", "british", "english", "dutch", "polish", "american", "brazilian",
    "indian", "chinese", "european",
}

_STOPWORDS = {
    "what", "which", "how", "does", "do", "did", "is", "are", "was", "were", "the", "a", "an", "show", "list",
    "compare", "tell", "me", "about", "and", "or", "in", "on", "for", "of", "with", "trial", "trials", "yield",
    "yields", "uplift", "average", "mean", "data", "evidence", "i", "can", "could", "you", "please", "give",
    "find", "all", "any", "between", "versus", "vs", "best", "worst", "top", "where", "when", "why", "who",
    "summary", "summarize", "overall", "scientific", "demonstration", "commercial", "control", "treated",
}


@dataclass
class QueryAnalysis:
    intent: Intent
    filters: dict[str, Any] = field(default_factory=dict)
    trial_ids: list[str] = field(default_factory=list)
    group_by: str | None = None
    unknown_terms: list[str] = field(default_factory=list)
    multi_values: dict[str, list[str]] = field(default_factory=dict)
    # The question asks what supports a claim ("what evidence…", "does it prove…").
    evidence: bool = False


def _find_entities(question: str, store: EvidenceStore) -> dict[str, list[str]]:
    """Canonical crops, products, countries and trial types mentioned in the question."""
    words = re.findall(r"[A-Za-z0-9]+", question)
    found: dict[str, list[str]] = {name: [] for name in FILTER_FIELDS}
    claimed: set[str] = set()
    for name in FILTER_FIELDS:
        lookup = store.lookup_table(name)
        for key, original in ngram_keys(question):
            if not key or key in claimed:
                continue
            canonical = lookup.get(key)
            if canonical is None:
                for suffix in ("es", "s"):
                    if key.endswith(suffix) and key[: -len(suffix)] in lookup:
                        canonical = lookup[key[: -len(suffix)]]
                        break
            if canonical is None:
                continue
            # Short codes (FR, DE, ES) only count when written in capitals, so "de" or "es" in prose don't.
            if len(key) <= 3 and name == "country" and original != original.upper():
                continue
            if name == "trial_type" and not _followed_by_trial_noun(words, original):
                continue
            if canonical not in found[name]:
                found[name].append(canonical)
            claimed.add(key)
    return found


def _followed_by_trial_noun(words: list[str], original: str) -> bool:
    parts = original.split()
    for index in range(len(words) - len(parts) + 1):
        if [word.lower() for word in words[index : index + len(parts)]] == [part.lower() for part in parts]:
            end = index + len(parts)
            following = {word.lower() for word in words[end : end + _TRIAL_NOUN_WINDOW]}
            preceding = words[index - 1].lower() if index > 0 else ""
            if following & _TRIAL_NOUNS or preceding in {"type", "types"}:
                return True
    return False


def _unknown_crop(question: str, store: EvidenceStore, entities: dict[str, list[str]]) -> str | None:
    """A word in a crop position ('sorghum yields') that no vocabulary knows, or None."""
    known = {key for name in FILTER_FIELDS for key in store.lookup_table(name)}
    entity_words = {norm_key(word) for values in entities.values() for value in values for word in value.split()}
    for match in _CROP_SLOT.finditer(question):
        word = match.group(1) or match.group(2)
        key = norm_key(word)
        if word.lower() in _NOT_CROPS or key in known or key in entity_words:
            continue
        if any(key.endswith(suffix) and key[: -len(suffix)] in known for suffix in ("es", "s")):
            continue
        return word
    return None


def _year_filters(question: str) -> dict[str, Any]:
    """A year, a list of years, or a range (year_from / year_to) from the question's wording."""
    if match := _YEAR_RANGE.search(question):
        low, high = sorted(int(year) for year in match.groups() if year)
        return {"year_from": low, "year_to": high}
    bounds: dict[str, int] = {}
    if match := _YEAR_FROM.search(question):
        word, year = (match.group(1), match.group(2)) if match.group(2) else (match.group(4), match.group(3))
        bounds["year_from"] = int(year) + (1 if word.lower() == "after" else 0)
    if match := _YEAR_TO.search(question):
        word, year = (match.group(1), match.group(2)) if match.group(2) else (match.group(4), match.group(3))
        bounds["year_to"] = int(year) - (1 if word.lower() == "before" else 0)
    if bounds:
        return bounds
    years = sorted({int(year) for year in re.findall(r"\b(?:19|20)\d{2}\b", question)})
    if len(years) == 1:
        return {"year": years[0]}
    return {"year": years} if years else {}


def _unknown_terms(question: str, store: EvidenceStore, entities: dict[str, list[str]]) -> list[str]:
    """Capitalised words that look like names (e.g. a product) but match nothing in the data."""
    known = {key for name in FILTER_FIELDS for key in store.lookup_table(name)}
    known |= {norm_key(trial_id) for trial_id in store.trial_ids}
    entity_words = {norm_key(word) for values in entities.values() for value in values for word in value.split()}
    terms = []
    for index, token in enumerate(re.findall(r"[A-Za-z][A-Za-z0-9\-]+", question)):
        key = norm_key(token)
        is_name_like = token[0].isupper() and (index > 0 or any(char.isupper() for char in token[1:]))
        if is_name_like and token.lower() not in _STOPWORDS and key not in known and key not in entity_words:
            terms.append(token)
    return terms


def analyze(question: str, store: EvidenceStore, previous: QueryAnalysis | None = None) -> QueryAnalysis:
    trial_ids = store.mentioned_trial_ids(question)
    entities = _find_entities(question, store)

    filters: dict[str, Any] = {}
    multi_values: dict[str, list[str]] = {}
    for name, values in entities.items():
        if len(values) == 1:
            filters[name] = values[0]
        elif len(values) > 1:
            # "in France and Germany": keep only those, and compare them.
            filters[name] = values
            multi_values[name] = values
    filters.update(_year_filters(question))
    if isinstance(filters.get("year"), list):
        multi_values["year"] = [str(year) for year in filters["year"]]

    group_by = next((name for name, pattern in _GROUP_BY_RULES if pattern.search(question)), None)
    if group_by is None and multi_values:
        group_by = next(iter(multi_values))

    intent: Intent | None = None
    if _HELP.search(question) and not (filters or trial_ids):
        intent = "help"
    else:
        intent = next((name for name, pattern in _INTENT_RULES if pattern.search(question)), None)
    if len(trial_ids) == 1 and intent in (None, "stats", "list", "reference"):
        intent = "detail"
    elif len(trial_ids) > 1 and intent in (None, "stats", "list", "detail"):
        intent = "compare"

    unknown = _unknown_terms(question, store, entities)
    if "crop" not in filters and (crop_word := _unknown_crop(question, store, entities)):
        filters["crop"] = crop_word
        unknown = [term for term in unknown if norm_key(term) != norm_key(crop_word)]
    if unknown and "product" not in filters and intent in (None, "stats", "list", "compare"):
        # An unknown capitalised name is most likely a product the data doesn't have.
        filters["product"] = unknown[0]

    analysis = QueryAnalysis(
        intent=intent or ("stats" if filters or group_by else "overview"),
        filters=filters,
        trial_ids=trial_ids,
        group_by=group_by,
        unknown_terms=unknown,
        multi_values=multi_values,
        evidence=bool(_EVIDENCE.search(question)),
    )

    # "What about in Germany?" keeps the previous question's subject and adds the new filter.
    if previous is not None and _FOLLOW_UP.search(question):
        analysis = replace(
            analysis,
            intent=previous.intent if intent is None else analysis.intent,
            filters={**previous.filters, **filters},
            group_by=group_by or previous.group_by,
            trial_ids=trial_ids or previous.trial_ids,
        )
    return analysis
