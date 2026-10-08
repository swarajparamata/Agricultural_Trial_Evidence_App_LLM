"""Answer templates of the offline planner, for cases the sample data doesn't reach."""

from agritrial_llm.composer import _ranking


def group(name: str, mean_uplift_pct: float) -> dict:
    return {"group": name, "trial_count": 1, "trials_with_uplift": 1, "mean_uplift_pct": mean_uplift_pct}


def test_ranking_names_every_group_tied_for_the_top():
    groups = [group("Harvest Plus", 5.0), group("Root Boost", 5.0), group("Grow Max", 2.0)]
    text = _ranking({"group_by": "product"}, groups)
    assert text.startswith(
        "**Harvest Plus** and **Root Boost** share the highest mean yield uplift: **+5.0%**, ahead of Grow Max at +2.0%."
    )


def test_ranking_without_a_runner_up():
    text = _ranking({"group_by": "crop"}, [group("Wheat", 4.0), group("Potato", 4.0)])
    assert text.startswith("**Wheat** and **Potato** share the highest mean yield uplift: **+4.0%**.")
