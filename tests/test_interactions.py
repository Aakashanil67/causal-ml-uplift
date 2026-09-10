import numpy as np
import pandas as pd
import pytest

from src.cate import split_train_eval
from src.data_loader import load_hillstrom
from src.interactions import interaction_test, purchase_segment


@pytest.fixture(scope="module")
def result():
    _train, eval_df = split_train_eval(load_hillstrom())
    return interaction_test(eval_df)


def test_interaction_test_reports_observed_segment_effects(result):
    effects, contrasts, joint_p = result

    assert effects.index.tolist() == ["mens only", "womens only", "both"]
    assert effects.index.name == "segment"
    assert effects.columns.tolist() == ["estimate", "std_error", "ci_low", "ci_high"]
    assert np.isfinite(effects.to_numpy()).all()
    assert effects.loc["mens only", "estimate"] == pytest.approx(0.04, abs=0.02)
    assert effects.loc["womens only", "estimate"] == pytest.approx(0.062, abs=0.02)
    assert effects.loc["both", "estimate"] == pytest.approx(0.158, abs=0.03)
    assert contrasts.index.tolist() == [
        "womens only - mens only",
        "both - mens only",
        "recency",
        "history",
    ]
    assert contrasts.columns.tolist() == [
        "estimate",
        "std_error",
        "ci_low",
        "ci_high",
        "p_value",
        "p_holm",
    ]
    assert contrasts.loc["womens only - mens only", "p_value"] < 0.10
    assert contrasts.loc["recency", "p_holm"] > 0.10
    assert 0 <= joint_p <= 1


def test_holm_adjustment_never_reduces_a_p_value(result):
    _effects, contrasts, _joint_p = result

    assert (contrasts["p_holm"] >= contrasts["p_value"]).all()
    assert (contrasts["p_holm"] <= 1).all()


def test_purchase_segment_rejects_unobserved_profiles():
    with pytest.raises(ValueError, match="neither purchase category"):
        purchase_segment(pd.DataFrame({"mens": [0], "womens": [0]}))
