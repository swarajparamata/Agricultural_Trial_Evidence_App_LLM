"""The example requests from the client brief, answered by the agent with the offline planner on the
dataset the backend sends for the sample data."""

import asyncio

from agritrial_llm.agent import run_agent
from agritrial_llm.schemas import AgentOptions, ChatRequest, Dataset


def ask(dataset: Dataset, settings, question: str):
    request = ChatRequest(question=question, dataset=dataset, options=AgentOptions(force_offline=True))
    response = asyncio.run(run_agent(request, settings))
    assert response.verification.passed, response.verification.issues
    return response


def tool_log(response) -> list[str]:
    return [event.message for event in response.trace if event.message.startswith("[Tool]")]


def test_show_all_wheat_trials_involving_harvest_plus(backend_dataset, settings):
    response = ask(backend_dataset, settings, "Show all wheat trials involving Harvest Plus.")
    assert response.answer.startswith("**4 trials of Harvest Plus on Wheat**")
    assert response.trial_ids == ["T01", "T02", "T05", "T06"]
    assert "control yield missing" in response.answer  # T05


def test_compare_harvest_plus_wheat_in_france_and_germany(backend_dataset, settings):
    response = ask(backend_dataset, settings, "Compare Harvest Plus wheat results in France and Germany.")
    assert response.answer.startswith("**France** has the highest mean yield uplift: **+5.0%**, ahead of Germany at +4.0%.")
    assert "1. France (T01): +5.0%" in response.answer
    assert "2. Germany (T02 and T05): +4.0%" in response.answer
    assert "Spain" not in response.answer  # only the two countries asked about
    assert response.trial_ids == ["T01", "T02", "T05"]


def test_evidence_for_root_boost_on_potato(backend_dataset, settings):
    response = ask(backend_dataset, settings, "What evidence supports Root Boost improving potato yields?")
    assert "**What the evidence shows:** 1 of 2 trials with a control shows a yield increase: T03" in response.answer
    assert "No difference in T04 (Scientific, 0.0%)" in response.answer
    assert "42 t/ha (trials_a.csv) vs 44 t/ha (T03_summary.txt)" in response.answer
    # The conflicting trial was read in full in a second round, chosen from the first round's results.
    assert "Neither source indicates which result is final" in response.answer
    assert any(message.startswith("[Tool] get_trial_details(trial_id='T03')") for message in tool_log(response))
    assert response.tool_calls == 3


def test_scientific_wheat_trials_from_2021_through_2025(backend_dataset, settings):
    response = ask(backend_dataset, settings, "Which scientific wheat trials ran from 2021 through 2025?")
    assert response.answer.startswith("**2 trials on Wheat from 2021 to 2025 (Scientific)**")
    assert response.trial_ids == ["T01", "T06"]  # T02 and T05 are demonstrations
    assert tool_log(response)[0].startswith(
        "[Tool] search_trials(crop='Wheat', trial_type='Scientific', year_from=2021, year_to=2025)"
    )


def test_no_evidence_for_harvest_plus_on_rice(backend_dataset, settings):
    response = ask(backend_dataset, settings, "Is there evidence for Harvest Plus improving rice yields?")
    assert response.answer.startswith("No trials in the data test **Harvest Plus on Rice**, so there is no evidence either way.")
    assert "Harvest Plus has only been trialled on Wheat (4 trials)." in response.answer
    assert "No trial in the data involves Rice." in response.answer
    # Nothing matched, so the agent searched for each part on its own before answering.
    calls = tool_log(response)
    assert any(message.startswith("[Tool] search_trials(product='Harvest Plus')") for message in calls)
    assert any(message.startswith("[Tool] search_trials(crop='Rice')") for message in calls)
    assert response.tool_calls == 4


def test_unknown_crop_is_not_ignored(backend_dataset, settings):
    response = ask(backend_dataset, settings, "Is there evidence for Harvest Plus improving sorghum yields?")
    assert response.answer.startswith("No trials in the data test **Harvest Plus on sorghum**")
    assert "There is no crop called 'sorghum' in the data." in response.answer
    assert response.trial_ids == []
