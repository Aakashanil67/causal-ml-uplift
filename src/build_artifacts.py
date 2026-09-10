"""Regenerate the numerical manifest and compact artifacts used by Streamlit."""

import numpy as np
import pandas as pd

from src.cate import (
    fit_causal_forest,
    heterogeneity_by_purchase_history,
    predict_cate,
    split_train_eval,
)
from src.config import (
    ARM_COL,
    ARMS,
    CONTROL_ARM,
    NOMINAL_PROPENSITIES,
    RANDOM_SEED,
    TREATMENT_COL,
)
from src.dml_ate import per_arm_ate_table, pooled_ate_table
from src.evaluation import crossfit_arm_outcomes, evaluate_policies
from src.interactions import interaction_test
from src.persist import fit_and_save
from src.policy import (
    break_even_margin,
    fit_policy_forest,
    heuristic_recommendations,
    incremental_net_value,
    policy_forest_recommendations,
)
from src.results import build_provenance, save_evaluation_artifacts, save_results
from src.uplift import (
    bootstrap_normalized_qini_ci,
    bootstrap_uplift_per_email_ci,
    normalized_qini_score,
    qini_coefficient,
    qini_curve,
    summarize_repeated_qini,
    uplift_deciles,
    uplift_per_email_at_k,
)

REPEAT_SEEDS = (RANDOM_SEED, 7, 19, 73, 101)


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def records_for_json(frame: pd.DataFrame) -> list[dict]:
    return [
        {key: _native(value) for key, value in row.items()}
        for row in frame.reset_index().to_dict(orient="records")
    ]


def policy_conclusion(comparisons: pd.DataFrame) -> str:
    label = "learned (DRPolicyForest) - email everyone (mens creative)"
    row = comparisons.loc[label]
    if row["ci_low"] <= 0 <= row["ci_high"]:
        return (
            "Held-out evidence does not establish that the learned policy beats blanket mens "
            "emailing. Blanket mens is the simpler evidence-supported action; personalised "
            "policy deployment needs a new experiment or stronger cross-campaign evidence."
        )
    if row["ci_low"] > 0:
        return "The learned policy beats blanket mens emailing on the held-out evaluation set."
    return "Blanket mens emailing beats the learned policy on the held-out evaluation set."


def _top_k_table(cate, treatment, outcome, n_boot):
    rows = []
    for k in (0.1, 0.2, 0.3, 0.5, 1.0):
        point = uplift_per_email_at_k(cate, treatment, outcome, k)
        low, high = bootstrap_uplift_per_email_ci(
            cate, treatment, outcome, k, n_boot=n_boot, seed=0
        )
        rows.append({"k": k, "value": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows).set_index("k")


def build_all(n_boot: int = 1000, repeat_seeds: tuple[int, ...] = REPEAT_SEEDS) -> dict:
    from src.data_loader import load_hillstrom

    provenance = build_provenance()
    df = load_hillstrom()
    pooled = pooled_ate_table(df)
    per_arm = per_arm_ate_table(df)
    repeated_rows = []
    primary = None
    for seed in repeat_seeds:
        train, eval_df = split_train_eval(df, seed=seed)
        forest = fit_causal_forest(train, seed=seed)
        cate = predict_cate(forest, eval_df)
        treatment = eval_df[TREATMENT_COL].to_numpy()
        outcome = eval_df["visit"].to_numpy(dtype=float)
        repeated_rows.append(
            {
                "seed": seed,
                "normalized_qini": normalized_qini_score(cate, treatment, outcome),
            }
        )
        if seed == RANDOM_SEED:
            primary = (train, eval_df, cate)
    if primary is None:
        raise ValueError(f"repeat_seeds must include the primary seed {RANDOM_SEED}.")

    train, eval_df, cate = primary
    segment_effects, contrasts, joint_p = interaction_test(eval_df)
    treatment = eval_df[TREATMENT_COL].to_numpy()
    visit = eval_df["visit"].to_numpy(dtype=float)
    spend = eval_df["spend"].to_numpy(dtype=float)
    curve = qini_curve(cate, treatment, visit)
    normalized, qini_low, qini_high = bootstrap_normalized_qini_ci(
        cate, treatment, visit, n_boot=n_boot, seed=0
    )
    deciles = uplift_deciles(cate, treatment, visit)
    purchase_history = heterogeneity_by_purchase_history(eval_df, cate)
    top_k = _top_k_table(cate, treatment, visit, n_boot)
    spend_point = uplift_per_email_at_k(cate, treatment, spend, 0.3)
    spend_low, spend_high = bootstrap_uplift_per_email_ci(
        cate, treatment, spend, 0.3, n_boot=n_boot, seed=1
    )

    policy_forest = fit_policy_forest(train)
    learned = policy_forest_recommendations(policy_forest, eval_df)
    policies = {
        "learned (DRPolicyForest)": learned,
        "email everyone (mens creative)": np.full(len(eval_df), "Mens E-Mail"),
        "email everyone (womens creative)": np.full(len(eval_df), "Womens E-Mail"),
        "purchase-history heuristic": heuristic_recommendations(eval_df),
        "email nobody": np.full(len(eval_df), CONTROL_ARM),
    }
    outcome_predictions = crossfit_arm_outcomes(eval_df)
    policy_values, comparisons = evaluate_policies(
        eval_df,
        policies,
        outcome_predictions,
        propensities=NOMINAL_PROPENSITIES,
        n_boot=n_boot,
        seed=2,
    )
    empirical_propensities = df[ARM_COL].value_counts(normalize=True).to_dict()
    sensitivity_values, _ = evaluate_policies(
        eval_df,
        policies,
        outcome_predictions,
        propensities=empirical_propensities,
        n_boot=max(200, n_boot // 5),
        seed=3,
    )
    spend_predictions = crossfit_arm_outcomes(eval_df, outcome_col="spend")
    spend_values, spend_comparisons = evaluate_policies(
        eval_df,
        policies,
        spend_predictions,
        propensities=NOMINAL_PROPENSITIES,
        outcome_col="spend",
        n_boot=n_boot,
        seed=4,
    )
    learned_spend = float(spend_values.loc["learned (DRPolicyForest)", "value"])
    no_email_spend = float(spend_values.loc["email nobody", "value"])
    contact_rate = float(np.mean(learned != CONTROL_ARM))
    margin_sensitivity = []
    for margin in (0.25, 0.50, 1.00):
        margin_sensitivity.append(
            {
                "gross_margin": margin,
                "incremental_net_value": incremental_net_value(
                    learned_spend,
                    no_email_spend,
                    contact_rate,
                    margin,
                    0.10,
                ),
            }
        )

    repeated = pd.DataFrame(repeated_rows)
    results = {
        "schema_version": 2,
        "metadata": provenance,
        "headline": {
            "pooled_ate": records_for_json(pooled),
            "per_arm_ate": records_for_json(per_arm),
        },
        "interactions": {
            "analysis": "post-hoc held-out interaction audit",
            "n": int(len(eval_df)),
            "joint_p_value": joint_p,
            "segment_effects": records_for_json(segment_effects),
            "contrasts": records_for_json(contrasts),
        },
        "ranking": {
            "cate_summary": {
                "mean": float(cate.mean()),
                "std": float(cate.std()),
                "min": float(cate.min()),
                "max": float(cate.max()),
            },
            "purchase_history": records_for_json(purchase_history),
            "raw_qini": qini_coefficient(curve),
            "normalized_qini": {
                "value": normalized,
                "ci_low": qini_low,
                "ci_high": qini_high,
            },
            "repeated_splits": records_for_json(repeated.set_index("seed")),
            "repeated_summary": summarize_repeated_qini(repeated),
            "deciles": records_for_json(deciles),
            "top_k": records_for_json(top_k),
            "gross_spend_top_30": {
                "value": spend_point,
                "ci_low": spend_low,
                "ci_high": spend_high,
            },
        },
        "policy": {
            "values": records_for_json(policy_values),
            "comparisons": records_for_json(comparisons),
            "empirical_propensity_values": records_for_json(sensitivity_values),
            "nominal_propensities": NOMINAL_PROPENSITIES,
            "empirical_propensities": empirical_propensities,
            "recommendation_counts": {
                arm: int(count) for arm, count in pd.Series(learned).value_counts().items()
            },
            "recommendation_shares": {
                arm: float(
                    pd.Series(learned)
                    .value_counts(normalize=True)
                    .reindex(ARMS, fill_value=0.0)[arm]
                )
                for arm in ARMS
            },
            "conclusion": policy_conclusion(comparisons),
        },
        "reported_spend_sensitivity": {
            "analysis": "evaluation of a visit-optimised policy; reported spend is not profit",
            "values": records_for_json(spend_values),
            "comparisons": records_for_json(spend_comparisons),
            "contact_rate": contact_rate,
            "email_cost_usd": 0.10,
            "break_even_gross_margin": break_even_margin(
                learned_spend, no_email_spend, contact_rate, 0.10
            ),
            "margin_sensitivity": margin_sensitivity,
        },
    }
    serving_payload = {
        "eval_df": eval_df[[TREATMENT_COL, "visit", "spend"]].reset_index(drop=True),
        "cate": cate,
        "policy_values": policy_values,
        "policy_comparisons": comparisons,
    }
    save_results(results)
    save_evaluation_artifacts(serving_payload, metadata=provenance)
    fit_and_save(df, metadata=provenance)
    return results


def main() -> None:
    results = build_all()
    ranking = results["ranking"]["normalized_qini"]
    print(
        "normalized Qini: "
        f"{ranking['value']:.4f} [{ranking['ci_low']:.4f}, {ranking['ci_high']:.4f}]"
    )
    print(results["policy"]["conclusion"])


if __name__ == "__main__":
    main()
