"""Post-hoc held-out contrasts for interpretable treatment-effect heterogeneity."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from src.config import TREATMENT_COL

SEGMENTS = ("mens only", "womens only", "both")


def purchase_segment(df: pd.DataFrame) -> pd.Series:
    """Return the observed purchase-category segment for every row.

    Hillstrom contains customers with at least one prior category purchase.  Treating a
    neither-category row as mens-only would silently extrapolate the interaction model.
    """
    mens = df["mens"].astype(bool)
    womens = df["womens"].astype(bool)
    labels = np.select(
        [mens & ~womens, ~mens & womens, mens & womens],
        [SEGMENTS[0], SEGMENTS[1], SEGMENTS[2]],
        default="neither",
    )
    result = pd.Series(labels, index=df.index, name="segment")
    if (result == "neither").any():
        raise ValueError("purchase profile contains a customer with neither purchase category")
    return result


def _contrast_row(model, vector: np.ndarray) -> dict[str, float]:
    test = model.t_test(vector)
    estimate = float(np.asarray(test.effect).squeeze())
    std_error = float(np.asarray(test.sd).squeeze())
    ci = np.asarray(test.conf_int(alpha=0.05)).reshape(-1)
    return {
        "estimate": estimate,
        "std_error": std_error,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(np.asarray(test.pvalue).squeeze()),
    }


def interaction_test(
    df: pd.DataFrame,
    outcome: str = "visit",
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Estimate observed segment effects and post-hoc effect-modification contrasts.

    The model is fit only on the supplied held-out evaluation data.  Mens-only is the real
    reference segment; recency and history are centred so the segment effects are evaluated at
    the evaluation-sample means.  The returned interaction p-values use Holm correction across
    the four reported effect-modification contrasts.
    """
    work = df.copy()
    work["purchase_segment"] = purchase_segment(work)
    work["segment_womens"] = (work["purchase_segment"] == "womens only").astype(int)
    work["segment_both"] = (work["purchase_segment"] == "both").astype(int)
    work["recency_c"] = work["recency"] - work["recency"].mean()
    work["history_c"] = work["history"] - work["history"].mean()
    formula = (
        f"{outcome} ~ {TREATMENT_COL} * "
        "(segment_womens + segment_both + recency_c + history_c) "
        "+ newbie + C(zip_code) + C(channel)"
    )
    model = smf.ols(formula, data=work).fit(cov_type="HC1")
    names = list(model.params.index)
    position = {name: index for index, name in enumerate(names)}

    def vector(*terms: tuple[str, float]) -> np.ndarray:
        result = np.zeros(len(names))
        for term, coefficient in terms:
            result[position[term]] = coefficient
        return result

    treatment = TREATMENT_COL
    womens_interaction = f"{TREATMENT_COL}:segment_womens"
    both_interaction = f"{TREATMENT_COL}:segment_both"
    recency_interaction = f"{TREATMENT_COL}:recency_c"
    history_interaction = f"{TREATMENT_COL}:history_c"

    effect_vectors = {
        SEGMENTS[0]: vector((treatment, 1.0)),
        SEGMENTS[1]: vector((treatment, 1.0), (womens_interaction, 1.0)),
        SEGMENTS[2]: vector((treatment, 1.0), (both_interaction, 1.0)),
    }
    effects = pd.DataFrame(
        {segment: _contrast_row(model, contrast) for segment, contrast in effect_vectors.items()}
    ).T[["estimate", "std_error", "ci_low", "ci_high"]]
    effects.index.name = "segment"

    contrast_vectors = {
        "womens only - mens only": vector((womens_interaction, 1.0)),
        "both - mens only": vector((both_interaction, 1.0)),
        "recency": vector((recency_interaction, 1.0)),
        "history": vector((history_interaction, 1.0)),
    }
    contrast_rows = {
        label: _contrast_row(model, contrast) for label, contrast in contrast_vectors.items()
    }
    raw_p = np.array([row["p_value"] for row in contrast_rows.values()])
    adjusted = multipletests(raw_p, method="holm")[1]
    for label, p_holm in zip(contrast_rows, adjusted, strict=True):
        contrast_rows[label]["p_holm"] = float(p_holm)
    contrasts = pd.DataFrame(contrast_rows).T[
        ["estimate", "std_error", "ci_low", "ci_high", "p_value", "p_holm"]
    ]
    contrasts.index.name = "contrast"

    joint = np.zeros((4, len(names)))
    for row, contrast in enumerate(contrast_vectors.values()):
        joint[row] = contrast
    joint_p = float(np.asarray(model.wald_test(joint, scalar=True).pvalue).squeeze())
    return effects, contrasts, joint_p


if __name__ == "__main__":
    from src.cate import split_train_eval
    from src.data_loader import load_hillstrom

    _train, evaluation = split_train_eval(load_hillstrom())
    effects_table, contrast_table, joint_pvalue = interaction_test(evaluation)
    print(effects_table)
    print(contrast_table)
    print(f"joint Wald p-value: {joint_pvalue:.6g}")
