from types import SimpleNamespace

import numpy as np
import pandas as pd

import src.build_artifacts as build_artifacts
from src.build_artifacts import policy_conclusion, records_for_json


def test_records_for_json_preserves_named_index_and_native_scalars():
    frame = pd.DataFrame(
        {"value": np.array([0.1838], dtype=np.float64)},
        index=pd.Index(["learned"], name="policy"),
    )

    records = records_for_json(frame)

    assert records == [{"policy": "learned", "value": 0.1838}]
    assert isinstance(records[0]["value"], float)


def test_policy_conclusion_follows_paired_interval_not_point_ordering():
    comparisons = pd.DataFrame(
        {"difference": [0.002], "ci_low": [-0.003], "ci_high": [0.006]},
        index=pd.Index(
            ["learned (DRPolicyForest) - email everyone (mens creative)"],
            name="comparison",
        ),
    )

    conclusion = policy_conclusion(comparisons)

    assert "does not establish" in conclusion
    assert "blanket mens" in conclusion.lower()


def test_identification_rows_use_the_public_backdoor_adjustment_set(monkeypatch):
    estimand = SimpleNamespace(
        estimand_type="nonparametric-ate",
        backdoor_variables=["backdoor1"],
        estimands={"backdoor": "ATE"},
        get_backdoor_variables=lambda: [],
    )
    monkeypatch.setattr(build_artifacts, "identify", lambda _df, _outcome: (None, estimand))

    rows = build_artifacts._identification_records(pd.DataFrame())

    assert len(rows) == 3
    assert all(row["backdoor_variables"] == [] for row in rows)


def test_policy_conclusion_follows_entirely_positive_interval():
    comparisons = pd.DataFrame(
        {"difference": [0.01], "ci_low": [0.002], "ci_high": [0.018]},
        index=pd.Index(
            ["learned (DRPolicyForest) - email everyone (mens creative)"],
            name="comparison",
        ),
    )

    conclusion = policy_conclusion(comparisons)

    assert "higher expected visit value" in conclusion
    assert "does not establish" not in conclusion


def test_policy_manifest_names_all_static_actions():
    policies = {
        "learned (DRPolicyForest)": [],
        "email everyone (mens creative)": [],
        "email everyone (womens creative)": [],
        "purchase-history heuristic": [],
        "email nobody": [],
    }

    assert "email everyone (mens creative)" in policies
    assert "email everyone (womens creative)" in policies
    assert "email nobody" in policies


def test_action_shares_include_zero_for_missing_arms():
    recommendations = pd.Series(["Mens E-Mail", "Mens E-Mail", "Womens E-Mail"])
    shares = recommendations.value_counts(normalize=True).reindex(
        ["Mens E-Mail", "Womens E-Mail", "No E-Mail"], fill_value=0.0
    )

    assert shares.sum() == 1.0
    assert shares["No E-Mail"] == 0.0
