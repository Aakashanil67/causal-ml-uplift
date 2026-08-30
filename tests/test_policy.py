import numpy as np
import pandas as pd
import pytest

from src.config import ARM_COL, CONTROL_ARM, NOMINAL_PROPENSITIES
from src.policy import (
    _treatment_name_to_arm,
    bootstrap_policy_difference_ci,
    dr_policy_value,
    heuristic_recommendations,
    ipw_policy_value,
    policy_value_contributions,
    stratified_bootstrap_indices,
)


def test_treatment_name_to_arm_maps_none_to_control():
    assert _treatment_name_to_arm("None") == CONTROL_ARM
    assert _treatment_name_to_arm("T0_Mens E-Mail") == "Mens E-Mail"
    assert _treatment_name_to_arm("T0_Womens E-Mail") == "Womens E-Mail"


def test_heuristic_prioritises_womens_when_both_purchased():
    df = pd.DataFrame({"mens": [1, 0, 1, 0], "womens": [0, 1, 1, 0]})
    rec = heuristic_recommendations(df)
    assert rec.tolist() == ["Mens E-Mail", "Womens E-Mail", "Womens E-Mail", "No E-Mail"]


def test_ipw_policy_value_recovers_arm_mean_when_recommending_that_arm_for_everyone():
    # if the policy recommends the same arm for everyone, the IPW value should equal that arm's
    # observed mean outcome exactly (every unit whose actual arm matches gets counted, correctly
    # reweighted by its known assignment probability).
    df = pd.DataFrame(
        {
            ARM_COL: ["No E-Mail", "No E-Mail", "Mens E-Mail", "Mens E-Mail", "Womens E-Mail"],
            "visit": [0.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    propensities = {"No E-Mail": 0.4, "Mens E-Mail": 0.4, "Womens E-Mail": 0.2}
    rec = np.full(len(df), "No E-Mail")
    value = ipw_policy_value(df, rec, "visit", propensities)
    # only the two No E-Mail rows contribute: (0/0.4 + 1/0.4) / 5 = 0.5
    expected = (0 / 0.4 + 1 / 0.4) / 5
    assert value == pytest.approx(expected)


def test_ipw_policy_value_on_real_data_matches_control_arm_rate():
    from src.cate import split_train_eval
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    _train, eval_df = split_train_eval(df)
    propensities = df[ARM_COL].value_counts(normalize=True).to_dict()
    rec = np.full(len(eval_df), CONTROL_ARM)
    value = ipw_policy_value(eval_df, rec, "visit", propensities)
    # true control-arm visit rate is 0.106167 (reports/data_dictionary.md); the IPW estimator
    # should recover it closely on a real held-out split, since assignment truly was random.
    assert value == pytest.approx(0.106167, abs=0.01)


def test_policy_value_contributions_average_to_the_ipw_value():
    df = pd.DataFrame(
        {ARM_COL: ["No E-Mail", "Mens E-Mail", "Womens E-Mail"], "visit": [0.0, 1.0, 1.0]}
    )
    propensities = {"No E-Mail": 1 / 3, "Mens E-Mail": 1 / 3, "Womens E-Mail": 1 / 3}
    rec = np.full(len(df), "Mens E-Mail")
    contributions = policy_value_contributions(df, rec, "visit", propensities)
    assert contributions.tolist() == pytest.approx([0.0, 3.0, 0.0])
    assert contributions.mean() == pytest.approx(ipw_policy_value(df, rec, "visit", propensities))


def test_paired_policy_difference_ci_is_exactly_zero_for_identical_policies():
    df = pd.DataFrame(
        {ARM_COL: ["No E-Mail", "Mens E-Mail", "Womens E-Mail"], "visit": [0.0, 1.0, 1.0]}
    )
    propensities = {"No E-Mail": 1 / 3, "Mens E-Mail": 1 / 3, "Womens E-Mail": 1 / 3}
    rec = np.full(len(df), "Mens E-Mail")
    point, ci_low, ci_high = bootstrap_policy_difference_ci(
        df, rec, rec, "visit", propensities, n_boot=200, seed=0
    )
    assert (point, ci_low, ci_high) == pytest.approx((0.0, 0.0, 0.0))


def test_nominal_randomisation_probabilities_are_explicit():
    assert NOMINAL_PROPENSITIES == {
        "No E-Mail": pytest.approx(1 / 3),
        "Mens E-Mail": pytest.approx(1 / 3),
        "Womens E-Mail": pytest.approx(1 / 3),
    }


def test_stratified_bootstrap_preserves_arm_counts():
    df = pd.DataFrame({ARM_COL: ["No E-Mail"] * 2 + ["Mens E-Mail"] * 3 + ["Womens E-Mail"] * 4})

    draws = stratified_bootstrap_indices(df, n_boot=20, seed=7)

    expected = df[ARM_COL].value_counts().sort_index()
    for idx in draws:
        actual = df.iloc[idx][ARM_COL].value_counts().sort_index()
        pd.testing.assert_series_equal(actual, expected)


def test_dr_policy_value_uses_outcome_model_for_every_customer():
    df = pd.DataFrame(
        {
            ARM_COL: ["No E-Mail", "Mens E-Mail", "Womens E-Mail"],
            "visit": [0.0, 1.0, 0.0],
        }
    )
    predictions = pd.DataFrame(
        {
            "No E-Mail": [0.1, 0.1, 0.1],
            "Mens E-Mail": [0.8, 0.8, 0.8],
            "Womens E-Mail": [0.2, 0.2, 0.2],
        }
    )
    rec = np.full(len(df), "Mens E-Mail")

    value = dr_policy_value(df, rec, "visit", NOMINAL_PROPENSITIES, predictions)

    # Baseline model contribution is 0.8 for everyone. The one matched mens row adds
    # (1.0 - 0.8) / (1/3), so the average is (0.8 + 1.4 + 0.8) / 3 = 1.0.
    assert value == pytest.approx(1.0)


def test_real_policy_evidence_supports_only_the_no_email_comparison():
    from src.cate import split_train_eval
    from src.data_loader import load_hillstrom
    from src.evaluation import crossfit_arm_outcomes, evaluate_policies
    from src.policy import fit_policy_forest, policy_forest_recommendations

    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    pf = fit_policy_forest(train)
    learned_rec = policy_forest_recommendations(pf, eval_df)
    heuristic_rec = heuristic_recommendations(eval_df)
    none_rec = np.full(len(eval_df), CONTROL_ARM)
    mens_rec = np.full(len(eval_df), "Mens E-Mail")
    outcome_predictions = crossfit_arm_outcomes(eval_df, n_estimators=20)
    _values, comparisons = evaluate_policies(
        eval_df,
        {
            "learned (DRPolicyForest)": learned_rec,
            "email everyone (mens creative)": mens_rec,
            "purchase-history heuristic": heuristic_rec,
            "email nobody": none_rec,
        },
        outcome_predictions,
        n_boot=300,
    )

    assert comparisons.loc["learned (DRPolicyForest) - email nobody", "ci_low"] > 0
    for baseline in ("email everyone (mens creative)", "purchase-history heuristic"):
        row = comparisons.loc[f"learned (DRPolicyForest) - {baseline}"]
        assert row["ci_low"] <= 0 <= row["ci_high"]
