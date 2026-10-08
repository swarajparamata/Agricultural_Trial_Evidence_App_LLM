import asyncio

from langchain_core.runnables import RunnableLambda

from agritrial_llm.extraction import run_extraction, validate_extraction
from agritrial_llm.schemas import ExtractRequest, ReportExtraction

REPORT = """Trial T07 field report
Crop: Wheat. Product: HarvestPlus. Location: Italy. Year: 2025.
Trial type: Demonstration.
Results
Treated yield: 7,9 t/ha. Untreated yield: 7,4 t/ha.
"""

UNITS = ["t/ha", "kg/ha"]


def test_validation_accepts_values_copied_from_the_text():
    fields = {
        "trial_id": "T07", "crop": "Wheat", "product": "HarvestPlus", "country": "Italy", "year": 2025,
        "trial_type": "Demonstration", "treatment_yield": 7.9, "control_yield": 7.4, "yield_unit": "t/ha",
    }
    assert validate_extraction(fields, REPORT, "T07_report.txt", UNITS) == []


def test_validation_rejects_invented_or_missing_values():
    fields = {"trial_id": "T08", "crop": None, "product": "HarvestPlus", "year": 3025, "treatment_yield": 8.2,
              "control_yield": 7.4, "yield_unit": "bushels"}
    issues = validate_extraction(fields, REPORT, "T07_report.txt", UNITS)
    assert any(issue.startswith("crop is missing") for issue in issues)
    assert any("trial_id 'T08' does not appear" in issue for issue in issues)
    assert any("year 3025" in issue for issue in issues)
    assert any("treatment_yield 8.2 does not appear" in issue for issue in issues)
    assert any("yield_unit 'bushels'" in issue for issue in issues)


def test_extraction_loop_retries_with_feedback(settings):
    attempts = [
        ReportExtraction(trial_id="T07", crop="Wheat", product="HarvestPlus", year=2025, treatment_yield=8.9,
                         control_yield=7.4, yield_unit="t/ha"),
        ReportExtraction(trial_id="T07", crop="Wheat", product="HarvestPlus", country="Italy", year=2025,
                         trial_type="Demonstration", treatment_yield=7.9, control_yield=7.4, yield_unit="t/ha"),
    ]
    seen = []

    def fake_model(messages):
        seen.append(messages)
        return attempts[len(seen) - 1]

    request = ExtractRequest(filename="T07_report.txt", text=REPORT, units=UNITS)
    response = asyncio.run(run_extraction(request, settings, model=RunnableLambda(fake_model)))

    assert response.ok
    assert response.attempts == 2
    assert response.fields.treatment_yield == 7.9
    assert "treatment_yield 8.9 does not appear" in seen[1][-1].content
    assert [event.status for event in response.trace if event.node == "validate"] == ["error", "success"]


def test_extraction_reports_unavailable_llm(settings):
    request = ExtractRequest(filename="T07_report.txt", text=REPORT)
    response = asyncio.run(run_extraction(request, settings))
    assert not response.ok
    assert response.mode == "unavailable"
    assert "not reachable" in response.issues[0]
