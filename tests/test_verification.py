from agritrial_llm.store import EvidenceStore
from agritrial_llm.verification import verify_answer


def observe(store: EvidenceStore, **filters) -> list[dict]:
    """Tool observations as the agent would have them after calling trial_statistics and search_trials."""
    import json

    return [
        {"tool": "trial_statistics", "args": filters, "output": json.dumps(store.statistics(**filters))},
        {"tool": "search_trials", "args": filters, "output": json.dumps(store.search(**filters))},
    ]


QUESTION = "What is the average uplift of Harvest Plus on wheat?"


def test_grounded_answer_passes(store: EvidenceStore):
    answer = (
        "Harvest Plus on wheat shows a mean uplift of +6.3% across T01 (+5.0%), T02 (+4.0%) and T06 (+10.0%). "
        "T05 is excluded because its control yield is missing. Yields were 8.4 vs 8 t/ha in T01."
    )
    result = verify_answer(answer, QUESTION, observe(store, product="Harvest Plus", crop="Wheat"), store)
    assert result.passed, result.issues


def test_rounded_numbers_are_accepted(store: EvidenceStore):
    answer = "The mean uplift is about 6% (T01, T02, T06); T05 has no control yield."
    result = verify_answer(answer, QUESTION, observe(store, product="Harvest Plus", crop="Wheat"), store)
    assert result.passed, result.issues


def test_unit_converted_numbers_are_accepted(store: EvidenceStore):
    answer = "In T02 the treated plots yielded 7,800 kg/ha against 7,500 kg/ha (+4.0%)."
    result = verify_answer(answer, QUESTION, observe(store, product="Harvest Plus", crop="Wheat"), store)
    assert result.passed, result.issues


def test_invented_numbers_fail(store: EvidenceStore):
    answer = "Harvest Plus raises wheat yields by 12.7% on average (T01, T02, T06); T05 lacks a control."
    result = verify_answer(answer, QUESTION, observe(store, product="Harvest Plus", crop="Wheat"), store)
    assert not result.passed
    assert "12.7%" in result.issues[0]


def test_conflicts_must_be_disclosed(store: EvidenceStore):
    observations = observe(store, product="Root Boost")
    hidden = "Root Boost: T03 yielded 42 t/ha against 40 t/ha (+5.0%); T04 showed 0.0%."
    result = verify_answer(hidden, "How does Root Boost perform?", observations, store)
    assert not result.passed
    assert any("T03 has conflicting" in issue for issue in result.issues)

    disclosed = (
        "Root Boost: T03 yielded 42 t/ha against 40 t/ha (+5.0%), but the sources conflict (44 t/ha in "
        "T03_summary.txt); T04 showed 0.0%."
    )
    assert verify_answer(disclosed, "How does Root Boost perform?", observations, store).passed


def test_missing_control_must_be_mentioned(store: EvidenceStore):
    answer = "Harvest Plus averages +6.3% in T01, T02 and T06. T05 yielded 8.1 t/ha."
    result = verify_answer(answer, QUESTION, observe(store, product="Harvest Plus", crop="Wheat"), store)
    assert any("T05 has no control yield" in issue for issue in result.issues)


def test_unknown_trial_ids_fail(store: EvidenceStore):
    answer = "T01 and T09 both show +5.0%."
    result = verify_answer(answer, QUESTION, observe(store, product="Harvest Plus"), store)
    assert any("T09" in issue for issue in result.issues)


def test_data_questions_need_tool_results(store: EvidenceStore):
    result = verify_answer("Harvest Plus works well.", QUESTION, [], store)
    assert not result.passed
    assert "without looking at the data" in result.issues[0]


def test_small_talk_needs_no_tools(store: EvidenceStore):
    assert verify_answer("Hello! Ask me about the trials.", "hi there", [], store).passed


def test_empty_answer_fails(store: EvidenceStore):
    assert not verify_answer("  ", QUESTION, [], store).passed


def test_caveats_must_belong_to_the_trial(store: EvidenceStore):
    observations = observe(store, crop="Wheat", trial_type="Scientific")
    # A small model gave T01 another trial's caveats; T01 has none.
    invented = "Two scientific wheat trials:\n- **T01** (France, 2023)\n  - Caveats: Reported by a single source\n- T06 (Spain, 2021)"
    result = verify_answer(invented, "Which scientific wheat trials are there?", observations, store)
    assert any(issue.startswith("T01 does not have these caveats: Reported by a single source") for issue in result.issues)

    correct = "Two scientific wheat trials:\n- T01 (France, 2023): no caveats\n- T06 (Spain, 2021): Reported by a single source"
    assert verify_answer(correct, "Which scientific wheat trials are there?", observations, store).passed


def test_conflicts_must_belong_to_the_trial(store: EvidenceStore):
    observations = observe(store, crop="Wheat", product="Harvest Plus")
    claims = "- T01: conflicting values between sources\n- T02 has no conflicts."
    issues = verify_answer(claims, "Show wheat trials with Harvest Plus", observations, store).issues
    assert any(issue.startswith("T01 does not have these caveats: conflicting values") for issue in issues)
    assert not any(issue.startswith("T02") for issue in issues)


def test_blanket_claims_must_fit_the_trials_cited(store: EvidenceStore):
    observations = observe(store, crop="Potato", product="Root Boost")
    answer = (
        "T03 shows +5.0% but its sources conflict (42 vs 44 t/ha); T04 shows 0.0%.\n"
        "The trials are mixed, with some being single-source."
    )
    issues = verify_answer(answer, "What evidence supports Root Boost on potato?", observations, store).issues
    assert "None of the trials in the answer is single-source. Remove that claim." in issues
