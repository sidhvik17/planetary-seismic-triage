"""Guard scientific distinctions missed by the initial secondary summary."""
import pytest

from scripts.aggregate_grouped_secondary import monte_carlo_summary, uncertainty_ratios
from scripts.analyze_catalog_tolerance import classify_catalog_match


def test_zero_exceedances_is_not_zero_probability():
    result = monte_carlo_summary({"permutation_n": 10000, "permutation_p_value": 0.0})
    assert result["permutation_exceedances"] == 0
    assert result["permutation_p"] == pytest.approx(1 / 10001)
    assert result["stored_raw_p"] == 0.0


def test_new_plus_one_result_is_not_corrected_twice():
    result = monte_carlo_summary({"permutation_n": 100, "permutation_exceedances": 2,
                                 "permutation_p_value": 3 / 101,
                                 "permutation_estimator": "plus_one"})
    assert result["permutation_p"] == pytest.approx(3 / 101)


def test_all_false_positive_ratio_includes_catalog_matches():
    ratios = uncertainty_ratios({"sigma_separation_fp_over_tp": 0.5,
        "median_sigma": {"tp": 0.1, "fp_all_unmatched": 0.2,
                         "fp_excluding_nakamura": 0.05,
                         "real_events_tp_plus_nakamura": 0.125}})
    assert ratios["fp_over_tp"] == 2.0
    assert ratios["catalog_unmatched_fp_over_benchmark_tp"] == 0.5
    assert ratios["cleanfp_over_real"] == 0.4


def test_late_detection_of_existing_label_is_not_new_catalog_event():
    association = classify_catalog_match(1200, [1000, 9000], [1000])
    assert association["catalog_match"]
    assert association["detection_to_grade_a_sec"] > 120
    assert association["catalog_entry_already_grade_a"]
    assert association["catalog_entry_exact_grade_a_time"]


def test_additional_catalog_entry_is_distinguished_from_existing_label():
    association = classify_catalog_match(9200, [1000, 9000], [1000])
    assert association["catalog_match"]
    assert not association["catalog_entry_already_grade_a"]
    assert association["nearest_catalog_to_grade_a_sec"] == 8000


def test_catalog_label_identity_tolerance_does_not_use_detection_distance():
    association = classify_catalog_match(1300, [1000], [1040])
    assert association["catalog_entry_already_grade_a"]
    assert not association["catalog_entry_exact_grade_a_time"]
    assert association["detection_to_grade_a_sec"] == 260
