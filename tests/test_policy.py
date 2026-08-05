import numpy as np
import pandas as pd
import pytest

from src.config import ARM_COL, CONTROL_ARM
from src.policy import (
    _cis_overlap,
    _treatment_name_to_arm,
    heuristic_recommendations,
    ipw_policy_value,
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


def test_cis_overlap():
    a = pd.Series({"ci_low": 0.1, "ci_high": 0.3})
    b = pd.Series({"ci_low": 0.2, "ci_high": 0.4})
    c = pd.Series({"ci_low": 0.5, "ci_high": 0.6})
    assert _cis_overlap(a, b)
    assert not _cis_overlap(a, c)


def test_learned_policy_beats_no_email_and_heuristic_on_real_data():
    from src.cate import split_train_eval
    from src.data_loader import load_hillstrom
    from src.policy import fit_policy_forest, policy_forest_recommendations

    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    propensities = df[ARM_COL].value_counts(normalize=True).to_dict()

    pf = fit_policy_forest(train)
    learned_rec = policy_forest_recommendations(pf, eval_df)
    heuristic_rec = heuristic_recommendations(eval_df)
    none_rec = np.full(len(eval_df), CONTROL_ARM)

    learned_value = ipw_policy_value(eval_df, learned_rec, "visit", propensities)
    heuristic_value = ipw_policy_value(eval_df, heuristic_rec, "visit", propensities)
    none_value = ipw_policy_value(eval_df, none_rec, "visit", propensities)

    assert learned_value > none_value
    assert learned_value > heuristic_value
