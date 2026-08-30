import numpy as np
import pandas as pd

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
    assert "blanket mens" in conclusion
