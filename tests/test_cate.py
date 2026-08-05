import numpy as np
import pytest

from src.cate import (
    fit_causal_forest,
    heterogeneity_by_purchase_history,
    predict_cate,
    split_train_eval,
)
from src.config import RANDOM_SEED, TREATMENT_COL


@pytest.fixture(scope="module")
def df():
    from src.data_loader import load_hillstrom

    return load_hillstrom()


@pytest.fixture(scope="module")
def split(df):
    return split_train_eval(df)


def test_split_preserves_arm_and_outcome_rates(df, split):
    train, eval_df = split
    assert len(train) + len(eval_df) == len(df)
    assert train[TREATMENT_COL].mean() == pytest.approx(df[TREATMENT_COL].mean(), abs=0.01)
    assert eval_df[TREATMENT_COL].mean() == pytest.approx(df[TREATMENT_COL].mean(), abs=0.01)
    assert train["visit"].mean() == pytest.approx(df["visit"].mean(), abs=0.01)
    assert eval_df["visit"].mean() == pytest.approx(df["visit"].mean(), abs=0.01)


def test_split_is_deterministic(df):
    train1, eval1 = split_train_eval(df, seed=RANDOM_SEED)
    train2, eval2 = split_train_eval(df, seed=RANDOM_SEED)
    assert train1.index.equals(train2.index) or (train1 == train2).all().all()
    assert len(eval1) == len(eval2)


@pytest.fixture(scope="module")
def cate_eval(split):
    train, eval_df = split
    cf = fit_causal_forest(train)
    cate = predict_cate(cf, eval_df)
    return eval_df, cate


def test_cate_shape_matches_eval_set(cate_eval):
    eval_df, cate = cate_eval
    assert cate.shape == (len(eval_df),)
    assert not np.isnan(cate).any()


def test_mean_cate_close_to_pooled_ate_benchmark(cate_eval):
    _eval_df, cate = cate_eval
    # pooled DML ATE on visit is +0.0601 (reports/05_dml_ate.md); the CATE mean on a held-out
    # split should land near it, since it's estimating the same population quantity a different way.
    assert 0.04 < cate.mean() < 0.08


def test_purchase_history_heterogeneity_is_real_not_noise(cate_eval):
    eval_df, cate = cate_eval
    table = heterogeneity_by_purchase_history(eval_df, cate)
    # confirmed with an OLS interaction test (T x womens p<0.001, T x mens p=0.006) before
    # trusting this — womens-only buyers should show a clearly higher CATE than mens-only buyers.
    womens_only = table.loc[(0, 1), "mean"]
    mens_only = table.loc[(1, 0), "mean"]
    assert womens_only > mens_only
    assert womens_only - mens_only > 0.02
