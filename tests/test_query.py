"""How the offline planner reads a question: entities, years and ranges, and words it doesn't know."""

import pytest

from agritrial_llm.query import analyze
from agritrial_llm.store import EvidenceStore


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Which wheat trials ran from 2021 through 2025?", {"year_from": 2021, "year_to": 2025}),
        ("Trials between 2022 and 2024", {"year_from": 2022, "year_to": 2024}),
        ("Harvest Plus trials 2021-2023", {"year_from": 2021, "year_to": 2023}),
        ("Root Boost trials since 2024", {"year_from": 2024}),
        ("Wheat trials after 2022", {"year_from": 2023}),
        ("Potato trials before 2024", {"year_to": 2023}),
        ("Harvest Plus trials from 2024", {"year": 2024}),
        ("Trials in 2021 and 2023", {"year": [2021, 2023]}),
    ],
)
def test_year_wording(store: EvidenceStore, question: str, expected: dict):
    filters = analyze(question, store).filters
    assert {key: filters[key] for key in ("year", "year_from", "year_to") if key in filters} == expected


def test_trial_type_may_come_before_the_crop(store: EvidenceStore):
    assert analyze("Which scientific wheat trials ran in 2023?", store).filters["trial_type"] == "Scientific"
    assert "trial_type" not in analyze("Is there scientific evidence for Harvest Plus?", store).filters


def test_several_countries_are_kept_and_compared(store: EvidenceStore):
    analysis = analyze("Compare Harvest Plus wheat results in France and Germany", store)
    assert analysis.filters["country"] == ["France", "Germany"]
    assert analysis.group_by == "country"


def test_unknown_crop_words_are_kept(store: EvidenceStore):
    assert analyze("Is there evidence for Harvest Plus improving sorghum yields?", store).filters["crop"] == "sorghum"
    for question in (
        "Show single source trials",
        "How many trials are there?",
        "Which trials are on average better?",
        "List trials in France",
        "Show me trials from 2024",
        "German trials with Harvest Plus",
    ):
        assert "crop" not in analyze(question, store).filters, question


def test_evidence_questions_are_recognised(store: EvidenceStore):
    assert analyze("What evidence supports Root Boost improving potato yields?", store).evidence
    assert not analyze("Show all wheat trials involving Harvest Plus.", store).evidence
