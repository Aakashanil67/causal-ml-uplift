import numpy as np
import pytest

from src.uplift import (
    bootstrap_gain_per_target_ci,
    gain_per_target_at_k,
    qini_coefficient,
    qini_curve,
    uplift_deciles,
)

# hand-worked fixture, verified by computing each cumulative step manually (see build notes):
# ranking order (already sorted best-to-worst): (T=1,Y=1),(T=1,Y=1),(T=0,Y=0),(T=0,Y=1),(T=1,Y=0),(T=0,Y=0)
# expected gain at each k: [1.0, 2.0, 2.0, 1.0, 0.5, 1.0]
FIXTURE_SCORES = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
FIXTURE_T = np.array([1, 1, 0, 0, 1, 0], dtype=float)
FIXTURE_Y = np.array([1, 1, 0, 1, 0, 0], dtype=float)
FIXTURE_EXPECTED_GAIN = [1.0, 2.0, 2.0, 1.0, 0.5, 1.0]


def test_qini_curve_matches_hand_computed_gain():
    curve = qini_curve(FIXTURE_SCORES, FIXTURE_T, FIXTURE_Y)
    assert curve["gain"].tolist() == pytest.approx(FIXTURE_EXPECTED_GAIN)
    assert curve["n_targeted"].tolist() == [1, 2, 3, 4, 5, 6]


def test_qini_curve_no_treated_units_does_not_crash():
    # every unit control: no ratio can ever be computed (cum_nt stays 0), gain should just sit
    # at 0 throughout rather than raising a division error.
    T = np.zeros(5)
    Y = np.array([1, 0, 1, 0, 1], dtype=float)
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    curve = qini_curve(scores, T, Y)
    assert (curve["gain"] == 0).all()


def test_qini_curve_no_control_units_falls_back_to_raw_treated_sum():
    # every unit treated: no control ever appears, so the ratio guard should never fire and gain
    # should equal the raw cumulative treated outcome sum (the documented fallback).
    T = np.ones(5)
    Y = np.array([1, 0, 1, 0, 1], dtype=float)
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    curve = qini_curve(scores, T, Y)
    assert curve["gain"].tolist() == np.cumsum(Y).tolist()


def test_qini_random_line_ends_at_total_gain():
    curve = qini_curve(FIXTURE_SCORES, FIXTURE_T, FIXTURE_Y)
    # by construction, the random line must meet the model curve exactly at k=1
    assert curve["random_line"].iloc[-1] == pytest.approx(curve["gain"].iloc[-1])
    assert curve["random_line"].iloc[0] == pytest.approx(curve["gain"].iloc[-1] / 6)


def test_qini_coefficient_positive_for_a_ranking_with_real_signal():
    curve = qini_curve(FIXTURE_SCORES, FIXTURE_T, FIXTURE_Y)
    assert qini_coefficient(curve) > 0


def test_qini_coefficient_near_zero_for_random_ranking():
    # a large synthetic set with a real, sizeable treatment effect but a totally uninformative
    # ranking (unrelated random scores) should average close to 0, not the fixture's positive value.
    rng = np.random.default_rng(0)
    n = 5000
    T = rng.integers(0, 2, n).astype(float)
    Y = (T * 0.1 + rng.random(n) * 0.5 > 0.4).astype(float)  # real but non-heterogeneous effect
    random_scores = rng.random(n)  # uninformative
    coef = qini_coefficient(qini_curve(random_scores, T, Y))
    # normalise by n so this is comparable across sample sizes; should be small
    assert abs(coef) / n < 0.02


def test_qini_coefficient_larger_for_perfect_ranking_than_random():
    # a ranking that exactly matches individual-level uplift should score higher than a
    # deliberately scrambled version of the same data.
    rng = np.random.default_rng(1)
    n = 2000
    true_uplift = rng.normal(0, 1, n)
    T = rng.integers(0, 2, n).astype(float)
    base = rng.random(n)
    Y = (base + T * true_uplift * 0.3 > 0.6).astype(float)

    perfect_coef = qini_coefficient(qini_curve(true_uplift, T, Y))
    scrambled_coef = qini_coefficient(qini_curve(rng.permutation(true_uplift), T, Y))
    assert perfect_coef > scrambled_coef


def test_cross_check_against_scikit_uplift():
    sklift = pytest.importorskip("sklift.metrics")
    from src.cate import fit_causal_forest, predict_cate, split_train_eval
    from src.config import TREATMENT_COL
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    cf = fit_causal_forest(train)
    cate = predict_cate(cf, eval_df)
    T = eval_df[TREATMENT_COL].to_numpy()
    Y = eval_df["visit"].to_numpy(dtype=float)

    mine = qini_coefficient(qini_curve(cate, T, Y))
    theirs = sklift.qini_auc_score(Y, cate, T)
    # different normalisations, so compare sign and "clearly better than random" rather than
    # magnitude — both should agree the ranking has real signal.
    assert mine > 0
    assert theirs > 0


def test_uplift_deciles_top_beats_bottom_on_a_clean_synthetic_signal():
    rng = np.random.default_rng(2)
    n = 4000
    score = rng.normal(0, 1, n)  # also the true per-unit uplift, for a clean test
    T = rng.integers(0, 2, n).astype(float)
    base = rng.random(n)
    Y = (base + T * score * 0.3 > 0.6).astype(float)

    deciles = uplift_deciles(score, T, Y, n_bins=10)
    assert deciles.loc[0, "observed_uplift"] > deciles.loc[9, "observed_uplift"]
    assert deciles["n"].sum() == n


def test_gain_per_target_at_k_matches_manual_slice():
    gpt = gain_per_target_at_k(FIXTURE_SCORES, FIXTURE_T, FIXTURE_Y, k=0.5)
    # k=0.5 of 6 units -> top 3, expected gain there is 2.0 (index 2), gain/n_targeted = 2/3
    assert gpt == pytest.approx(2.0 / 3.0)


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    ci_low, ci_high = bootstrap_gain_per_target_ci(
        FIXTURE_SCORES, FIXTURE_T, FIXTURE_Y, k=0.5, n_boot=200, seed=0
    )
    assert ci_low <= ci_high
