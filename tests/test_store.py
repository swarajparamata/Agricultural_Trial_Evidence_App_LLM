from agritrial_llm.store import EvidenceStore, fmt_num, fmt_pct


def test_resolves_spelling_variants(store: EvidenceStore):
    assert store.resolve("product", "HarvestPlus") == "Harvest Plus"
    assert store.resolve("product", "ROOTBOOST") == "Root Boost"
    assert store.resolve("crop", "potatoes") == "Potato"
    assert store.resolve("country", "FR") == "France"
    assert store.resolve("country", "deutschland") == "Germany"
    assert store.resolve("product", "GrowMax") is None


def test_search_filters_with_aliases(store: EvidenceStore):
    result = store.search(product="HARVESTPLUS", crop="wheat")
    assert [trial["id"] for trial in result["trials"]] == ["T01", "T02", "T05", "T06"]
    assert result["filters"] == {"crop": "Wheat", "product": "Harvest Plus"}


def test_filters_take_several_values_and_year_ranges(store: EvidenceStore):
    both = store.search(product="Harvest Plus", country=["France", "Germany"])
    assert [t["id"] for t in both["trials"]] == ["T01", "T02", "T05"]
    assert both["filters"]["country"] == ["France", "Germany"]
    assert [t["id"] for t in store.search(country="FR, DE")["trials"]] == ["T01", "T02", "T04", "T05"]
    assert [t["id"] for t in store.search(crop="Wheat", year_from=2021, year_to=2023)["trials"]] == ["T01", "T06"]
    assert [t["id"] for t in store.search(year=[2021, 2025])["trials"]] == ["T05", "T06"]
    partly_known = store.search(country=["France", "Atlantis"])
    assert partly_known["unresolved_terms"] == {"country": "Atlantis"}
    assert [t["id"] for t in partly_known["trials"]] == ["T01", "T04"]


def test_statistics_split_the_evidence(store: EvidenceStore):
    stats = store.statistics(product="Root Boost")
    assert (stats["trials_with_increase"], stats["trials_without_difference"], stats["trials_with_decrease"]) == (1, 1, 0)


def test_search_by_status_accepts_synonyms(store: EvidenceStore):
    assert [t["id"] for t in store.search(status="conflicting")["trials"]] == ["T03"]
    assert [t["id"] for t in store.search(status="single source")["trials"]] == ["T06"]


def test_unknown_terms_return_known_values(store: EvidenceStore):
    result = store.search(product="GrowMax")
    assert result["count"] == 0
    assert result["unresolved_terms"] == {"product": "GrowMax"}
    assert result["known_values"]["product"] == ["Harvest Plus", "Root Boost"]


def test_statistics_exclude_trials_without_control(store: EvidenceStore):
    stats = store.statistics(product="Harvest Plus", crop="Wheat")
    assert stats["trial_count"] == 4
    assert stats["trials_with_uplift"] == 3
    assert stats["mean_uplift_pct"] == 6.3  # (5.0 + 4.0 + 10.0) / 3
    assert stats["max_uplift"] == {"trial_ids": ["T06"], "uplift_pct": 10.0}
    assert stats["min_uplift"] == {"trial_ids": ["T02"], "uplift_pct": 4.0}
    assert stats["excluded_from_uplift"] == [{"trial_id": "T05", "reason": "control yield missing"}]
    assert stats["mean_difference"] == 0.433


def test_statistics_list_every_trial_tied_at_an_extreme(store: EvidenceStore):
    stats = store.statistics(year=2023)  # T01 and T03 both show +5.0%
    assert stats["max_uplift"] == {"trial_ids": ["T01", "T03"], "uplift_pct": 5.0}
    assert stats["min_uplift"] == stats["max_uplift"]


def test_statistics_show_the_range_caused_by_conflicts(store: EvidenceStore):
    stats = store.statistics(group_by="product")
    groups = {group["group"]: group for group in stats["groups"]}
    assert [group["group"] for group in stats["groups"]] == ["Harvest Plus", "Root Boost"]
    assert groups["Root Boost"]["mean_uplift_pct"] == 2.5
    assert groups["Root Boost"]["mean_uplift_range_pct"] == [2.5, 5.0]
    assert "mean_difference" not in stats  # several crops: absolute differences aren't comparable
    assert "several crops" in stats["note"]


def test_compare_lists_differences_and_warnings(store: EvidenceStore):
    comparison = store.compare(["T01", "t04"])
    assert comparison["fields_that_differ"] == ["Product", "Crop", "Year"]
    assert comparison["shared_values"] == {"Country": "France", "Trial type": "Scientific"}
    assert comparison["ranking_by_uplift"][0] == {"trial_id": "T01", "uplift_pct": 5.0}
    assert any("Different crops" in warning for warning in comparison["warnings"])


def test_compare_reports_unknown_ids(store: EvidenceStore):
    comparison = store.compare(["T01", "T99"])
    assert comparison["not_found"] == ["T99"]
    assert "at least two" in comparison["note"]


def test_data_issues(store: EvidenceStore):
    issues = store.data_issues()
    assert [c["trial_id"] for c in issues["conflicts"]] == ["T03"]
    assert issues["conflicts"][0]["values"] == ["42 t/ha (trials_a.csv)", "44 t/ha (T03_summary.txt)"]
    assert issues["missing_values"] == [{"trial_id": "T05", "missing": ["control_yield"], "sources": ["trials_b.csv"]}]
    assert [item["trial_id"] for item in issues["single_source_trials"]] == ["T05", "T06"]
    assert issues["caveat_counts"]["Replication not documented"] == 3


def test_reference_summary_counts_values(store: EvidenceStore):
    reference = store.reference_summary()
    assert reference["trial_ids"] == ["T01", "T02", "T03", "T04", "T05", "T06"]
    products = {row["name"]: row["trials"] for row in reference["products"]}
    assert products == {"Harvest Plus": 4, "Root Boost": 2}
    assert reference["vocabulary_without_trials"]["crop"] == ["Maize"]


def test_trial_id_mentions(store: EvidenceStore):
    # Canonical IDs in order of first mention, whatever case the text uses.
    assert store.mentioned_trial_ids("Compare t04 with T01 and T99") == ["T04", "T01"]
    assert store.unknown_trial_ids("T04 and T99 but not Q3") == ["T99"]


def test_number_formatting():
    assert fmt_num(8.0) == "8"
    assert fmt_num(7800) == "7,800"
    assert fmt_num(0.433, 3) == "0.433"
    assert fmt_pct(5) == "+5.0%"
    assert fmt_pct(0) == "0.0%"
    assert fmt_pct(-2.26) == "-2.3%"


def test_empty_combinations_say_what_each_part_finds(store: EvidenceStore):
    output = store.search(product="Harvest Plus", crop="Potato")
    assert output["count"] == 0
    assert output["found_separately"]["product=Harvest Plus"]["crops"] == ["Wheat"]
    assert output["found_separately"]["crop=Potato"]["products"] == ["Root Boost"]
    assert "found_separately" not in store.search(product="Harvest Plus")
