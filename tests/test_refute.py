import numpy as np
import pytest

from src.refute import (
    _ci_contains_zero,
    _close_to,
    data_subset_refuter,
    placebo_treatment_refuter,
    random_common_cause_refuter,
)


def test_ci_contains_zero():
    assert _ci_contains_zero({"ci_low": -0.01, "ci_high": 0.02})
    assert not _ci_contains_zero({"ci_low": 0.01, "ci_high": 0.02})
    assert not _ci_contains_zero({"ci_low": -0.02, "ci_high": -0.01})


def test_close_to():
    assert _close_to(0.06, 0.061, tol=0.01)
    assert not _close_to(0.06, 0.10, tol=0.01)


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


def test_placebo_treatment_kills_the_effect_on_real_data(df):
    # shuffling treatment should destroy the real, known-positive effect on visit — the estimated
    # placebo ATE should be close to zero, nowhere near the true +0.0601 pooled ATE.
    result = placebo_treatment_refuter(df, "visit")
    assert abs(result["ate"]) < 0.02
    assert _ci_contains_zero(result)


def test_random_common_cause_barely_moves_the_estimate(df):
    from src.config import TREATMENT_COL
    from src.dml_ate import dml_ate

    original = dml_ate(df, TREATMENT_COL, "visit")
    result = random_common_cause_refuter(df, "visit")
    assert _close_to(result["ate"], original["ate"], tol=0.01)


def test_data_subset_refuter_is_stable_on_real_data(df):
    ates = data_subset_refuter(df, "visit", n_runs=3)
    assert len(ates) == 3
    assert np.std(ates) < 0.01


def test_drop_cols_changes_the_refuted_model():
    # this is the exact bug caught during development: without drop_cols, the refuter silently
    # includes a column that was supposed to be withheld, refuting a different, better-specified
    # model than the one actually under test. Confirm the two calls give visibly different answers.
    from src.config import RANDOM_SEED
    from src.confounded import make_confounded_sample
    from src.data_loader import load_hillstrom

    real_df = load_hillstrom()
    confounded = make_confounded_sample(
        real_df, ["newbie"], {"newbie": -1}, strength=1.5, seed=RANDOM_SEED
    )
    with_newbie = random_common_cause_refuter(confounded, "visit", drop_cols=None)
    without_newbie = random_common_cause_refuter(confounded, "visit", drop_cols=["newbie"])
    assert abs(with_newbie["ate"] - without_newbie["ate"]) > 0.01
