import pandas as pd
import pytest

from src.config import ARMS, TREATMENT_COL
from src.data_loader import _validate_schema, build_covariate_matrix, load_hillstrom


@pytest.fixture(scope="module")
def df():
    return load_hillstrom()


def test_row_and_arm_counts(df):
    assert len(df) == 64000
    counts = df["segment"].value_counts()
    assert counts["No E-Mail"] == 21306
    assert counts["Mens E-Mail"] == 21307
    assert counts["Womens E-Mail"] == 21387


def test_no_nulls(df):
    assert not df.isnull().any().any()


def test_treatment_column_matches_arm(df):
    assert set(df[TREATMENT_COL].unique()) == {0, 1}
    assert (df.loc[df["segment"] == "No E-Mail", TREATMENT_COL] == 0).all()
    assert (df.loc[df["segment"] != "No E-Mail", TREATMENT_COL] == 1).all()


def test_conversion_implies_visit(df):
    assert ((df["conversion"] == 1) & (df["visit"] == 0)).sum() == 0


def test_spend_censored_at_499(df):
    # documents a real quirk of the published data (see reports/data_dictionary.md) — this test
    # exists to catch it silently changing if the source file is ever re-downloaded/updated.
    assert df["spend"].max() == 499.0
    assert (df["spend"] == 499.0).sum() == 12
    assert (df["spend"] > 499.0).sum() == 0


def test_spend_nonzero_matches_conversion(df):
    assert ((df["spend"] > 0) != (df["conversion"] == 1)).sum() == 0


def test_reject_unexpected_arm_labels():
    bad = pd.DataFrame(
        {
            "recency": [1],
            "history_segment": ["1) $0 - $100"],
            "history": [30.0],
            "mens": [1],
            "womens": [0],
            "zip_code": ["Urban"],
            "newbie": [0],
            "channel": ["Web"],
            "segment": ["Not A Real Arm"],
            "visit": [0],
            "conversion": [0],
            "spend": [0.0],
        }
    )
    with pytest.raises(ValueError, match="Unexpected arm labels"):
        _validate_schema(bad)


def test_reject_conversion_without_visit():
    bad = pd.DataFrame(
        {
            "recency": [1],
            "history_segment": ["1) $0 - $100"],
            "history": [30.0],
            "mens": [1],
            "womens": [0],
            "zip_code": ["Urban"],
            "newbie": [0],
            "channel": ["Web"],
            "segment": [ARMS[0]],
            "visit": [0],
            "conversion": [1],
            "spend": [50.0],
        }
    )
    with pytest.raises(ValueError, match="conversion=1 with visit=0"):
        _validate_schema(bad)


def test_covariate_matrix_shape_and_dtype(df):
    X = build_covariate_matrix(df)
    assert len(X) == len(df)
    assert X.isnull().sum().sum() == 0
    assert X.select_dtypes(include="float").shape[1] == X.shape[1]
    # one-hot dummies drop the first level: 3 zip codes -> 2 columns, 3 channels -> 2 columns
    assert "zip_code_Surburban" in X.columns
    assert "zip_code_Rural" not in X.columns  # dropped as the reference level


def test_covariate_matrix_columns_stable_on_a_single_row(df):
    # regression test for a real bug: a single-row input only has one zip_code/channel value, and
    # without a fixed category list get_dummies() silently produces fewer columns than a full-batch
    # fit does, misaligning every downstream model at serving time (src/persist.py, app/simulator.py).
    full = build_covariate_matrix(df)
    single = build_covariate_matrix(df.iloc[[0]])
    assert list(single.columns) == list(full.columns)


def test_covariate_matrix_rejects_unknown_category_instead_of_encoding_reference(df):
    invalid = df.iloc[[0]].copy()
    invalid["zip_code"] = "Typo City"

    with pytest.raises(ValueError, match="zip_code.*unsupported category"):
        build_covariate_matrix(invalid)
