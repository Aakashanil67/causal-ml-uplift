"""Regenerate the numerical manifest, figures, reports, and compact serving artifacts."""

import hashlib

import numpy as np
import pandas as pd

from src.cate import (
    fit_causal_forest,
    heterogeneity_by_bin,
    heterogeneity_by_purchase_history,
    plot_cate_distribution,
    plot_heterogeneity_by_covariate,
    plot_heterogeneity_by_purchase_history,
    predict_cate,
    split_train_eval,
)
from src.config import (
    ARM_COL,
    ARMS,
    CONTROL_ARM,
    COVARIATE_COLS,
    FIGURES_DIR,
    NOMINAL_PROPENSITIES,
    RANDOM_SEED,
    TREATMENT_COL,
)
from src.dag import build_dag, draw_dag
from src.dml_ate import per_arm_ate_table, pooled_ate_table
from src.evaluation import crossfit_arm_outcomes, evaluate_policies
from src.identify import identify
from src.interactions import interaction_test
from src.naive import covariate_balance, naive_estimates
from src.persist import fit_and_save
from src.policy import (
    break_even_margin,
    fit_policy_forest,
    heuristic_recommendations,
    incremental_net_value,
    policy_comparison_conclusion,
    policy_forest_recommendations,
)
from src.regression_baseline import average_marginal_effects, fit_logit, fit_ols
from src.results import build_provenance, save_evaluation_artifacts, save_results
from src.simulation import run_extended_monte_carlo, run_monte_carlo
from src.uplift import (
    bootstrap_normalized_qini_ci,
    bootstrap_uplift_per_email_ci,
    normalized_qini_score,
    plot_qini_curve,
    qini_coefficient,
    qini_curve,
    summarize_repeated_qini,
    uplift_deciles,
    uplift_per_email_at_k,
)

REPEAT_SEEDS = (RANDOM_SEED, 7, 19, 73, 101)
FOREST_SENSITIVITY = (
    {"label": "smaller leaves", "min_samples_leaf": 25, "max_depth": None},
    {"label": "primary", "min_samples_leaf": 50, "max_depth": None},
    {"label": "larger leaves", "min_samples_leaf": 100, "max_depth": None},
    {"label": "depth capped", "min_samples_leaf": 50, "max_depth": 5},
)


def _native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records_for_json(frame: pd.DataFrame, nullable_fields: tuple[str, ...] = ()) -> list[dict]:
    """Convert a table to JSON-native records, allowing null only for named unavailable fields."""
    records = []
    for row in frame.reset_index().to_dict(orient="records"):
        converted = {}
        for key, value in row.items():
            if key in nullable_fields:
                native = _native(value)
                if pd.isna(value) or (isinstance(native, (int, float)) and not np.isfinite(native)):
                    converted[key] = None
                    continue
            converted[key] = _native(value)
        records.append(converted)
    return records


def build_classical_sections(df: pd.DataFrame, pooled: pd.DataFrame) -> dict:
    """Compute manifest-owned naive, regression, identification, and selection evidence."""
    naive = naive_estimates(df, TREATMENT_COL)
    balance = covariate_balance(df, TREATMENT_COL)

    regression_estimates = []
    logit_contrasts = {}
    for outcome in ("visit", "conversion"):
        contrasts = average_marginal_effects(fit_logit(df, outcome), df)
        logit_contrasts[outcome] = records_for_json(contrasts)
        treatment = contrasts.loc[TREATMENT_COL]
        regression_estimates.append(
            {
                "outcome": outcome,
                "contrast": TREATMENT_COL,
                "effect": float(treatment["effect"]),
                "ci_low": float(treatment["ci_low"]),
                "ci_high": float(treatment["ci_high"]),
                "kind": "average 0-to-1 change",
            }
        )
    spend_model = fit_ols(df, "spend")
    spend_ci = spend_model.conf_int().loc[TREATMENT_COL]
    regression_estimates.append(
        {
            "outcome": "spend",
            "contrast": TREATMENT_COL,
            "effect": float(spend_model.params[TREATMENT_COL]),
            "ci_low": float(spend_ci.iloc[0]),
            "ci_high": float(spend_ci.iloc[1]),
            "kind": "OLS coefficient",
        }
    )

    identification_rows = _identification_records(df)

    from src.confounded import run_variant

    visit_reference = pooled.loc["visit"]
    variants = {}
    selected_samples = {}
    for name, columns, directions, strength, withheld in (
        ("observable", ["recency", "history"], {"recency": -1, "history": 1}, 1.2, None),
        ("unmeasured", ["newbie"], {"newbie": -1}, 1.5, ["newbie"]),
    ):
        run = run_variant(df, "visit", columns, directions, strength, withhold=withheld)
        selected_samples[name] = run["sample"]
        variants[name] = {key: value for key, value in run.items() if key != "sample"}
    variants["unmeasured"]["newbie_effect"] = next(
        row["effect"] for row in logit_contrasts["visit"] if row["contrast"] == "newbie"
    )

    from src.refute import (
        data_subset_refuter,
        placebo_treatment_refuter,
        random_common_cause_refuter,
    )

    selected_sample = selected_samples["unmeasured"]
    selected_estimate = variants["unmeasured"]["dml"]["ate"]
    refutations = {}
    for name, sample, dropped, original in (
        ("rct", df, None, float(visit_reference["ate"])),
        ("selected_sample", selected_sample, ["newbie"], selected_estimate),
    ):
        refutations[name] = {
            "reference_estimate": original,
            "placebo": placebo_treatment_refuter(sample, "visit", drop_cols=dropped),
            "random_cause": random_common_cause_refuter(sample, "visit", drop_cols=dropped),
            "subset": data_subset_refuter(sample, "visit", drop_cols=dropped),
        }

    return {
        "naive": {
            "estimates": records_for_json(naive),
            "balance": records_for_json(balance, nullable_fields=("standardised_diff",)),
        },
        "regression": {"estimates": regression_estimates, "logit_contrasts": logit_contrasts},
        "identification": {"outcomes": identification_rows},
        "confounding": {
            "experimental_reference": {
                "ate": float(visit_reference["ate"]),
                "ci_low": float(visit_reference["ci_low"]),
                "ci_high": float(visit_reference["ci_high"]),
            },
            "variants": variants,
        },
        "refutations": refutations,
    }


def _identification_records(df: pd.DataFrame) -> list[dict]:
    rows = []
    for outcome in ("visit", "conversion", "spend"):
        _model, estimand = identify(df, outcome)
        rows.append(
            {
                "outcome": outcome,
                "estimand_type": str(estimand.estimand_type),
                "backdoor_variables": list(estimand.get_backdoor_variables()),
                "estimand": str(estimand.estimands.get("backdoor", "")),
            }
        )
    return rows


def policy_conclusion(comparisons: pd.DataFrame) -> str:
    label = "learned (DRPolicyForest) - email everyone (mens creative)"
    row = comparisons.loc[label]
    return policy_comparison_conclusion(float(row["ci_low"]), float(row["ci_high"]))


def _top_k_table(cate, treatment, outcome, n_boot):
    rows = []
    for k in (0.1, 0.2, 0.3, 0.5, 1.0):
        point = uplift_per_email_at_k(cate, treatment, outcome, k)
        low, high = bootstrap_uplift_per_email_ci(
            cate, treatment, outcome, k, n_boot=n_boot, seed=0
        )
        rows.append({"k": k, "value": point, "ci_low": low, "ci_high": high})
    return pd.DataFrame(rows).set_index("k")


def build_all(
    n_boot: int = 1000,
    repeat_seeds: tuple[int, ...] = REPEAT_SEEDS,
    full_simulation_runs: int = 500,
) -> dict:
    from src.data_loader import load_hillstrom

    provenance = build_provenance()
    df = load_hillstrom()
    pooled = pooled_ate_table(df)
    per_arm = per_arm_ate_table(df)
    classical = build_classical_sections(df, pooled)
    repeated_rows = []
    primary = None
    for seed in repeat_seeds:
        train, eval_df = split_train_eval(df, seed=seed)
        forest = fit_causal_forest(train, seed=seed)
        cate = predict_cate(forest, eval_df)
        treatment = eval_df[TREATMENT_COL].to_numpy()
        outcome = eval_df["visit"].to_numpy(dtype=float)
        repeated_qini = bootstrap_normalized_qini_ci(
            cate, treatment, outcome, n_boot=300, seed=seed
        )
        repeated_rows.append(
            {
                "seed": seed,
                "normalized_qini": normalized_qini_score(cate, treatment, outcome),
                "qini_ci_low": repeated_qini[1],
                "qini_ci_high": repeated_qini[2],
                "n_eval": int(len(eval_df)),
            }
        )
        if seed == RANDOM_SEED:
            primary = (train, eval_df, cate)
    if primary is None:
        raise ValueError(f"repeat_seeds must include the primary seed {RANDOM_SEED}.")

    train, eval_df, cate = primary
    segment_effects, contrasts, joint_p = interaction_test(eval_df)

    def pooled_visit_difference(frame: pd.DataFrame) -> float:
        return float(
            frame.loc[frame[TREATMENT_COL] == 1, "visit"].mean()
            - frame.loc[frame[TREATMENT_COL] == 0, "visit"].mean()
        )

    deduplicated = df.drop_duplicates().reset_index(drop=True)
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

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    draw_dag(build_dag(), COVARIATE_COLS)
    plot_cate_distribution(cate, FIGURES_DIR / "cate_distribution.png")
    plot_heterogeneity_by_purchase_history(
        purchase_history, FIGURES_DIR / "cate_by_purchase_history.png"
    )
    plot_heterogeneity_by_covariate(
        heterogeneity_by_bin(eval_df, cate, "recency"),
        "recency",
        FIGURES_DIR / "cate_by_recency.png",
    )
    plot_heterogeneity_by_covariate(
        heterogeneity_by_bin(eval_df, cate, "history"),
        "history",
        FIGURES_DIR / "cate_by_history.png",
    )
    plot_qini_curve(curve, qini_coefficient(curve), FIGURES_DIR / "qini_curve.png")
    figure_files = [
        {"path": path.name, "sha256": _sha256(path)} for path in sorted(FIGURES_DIR.glob("*.png"))
    ]

    forest_sensitivity = []
    for config in FOREST_SENSITIVITY:
        sensitivity_forest = fit_causal_forest(
            train,
            seed=RANDOM_SEED,
            n_estimators=500,
            min_samples_leaf=config["min_samples_leaf"],
            max_depth=config["max_depth"],
        )
        sensitivity_cate = predict_cate(sensitivity_forest, eval_df)
        forest_sensitivity.append(
            {
                "label": config["label"],
                "min_samples_leaf": config["min_samples_leaf"],
                "max_depth": config["max_depth"],
                "normalized_qini": normalized_qini_score(sensitivity_cate, treatment, visit),
            }
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

    policy_split_sensitivity = []
    for seed in repeat_seeds:
        split_train, split_eval = split_train_eval(df, seed=seed)
        split_policy_forest = fit_policy_forest(split_train, seed=seed)
        split_learned = policy_forest_recommendations(split_policy_forest, split_eval)
        split_predictions = crossfit_arm_outcomes(split_eval, seed=seed)
        split_policies = {
            "learned (DRPolicyForest)": split_learned,
            "email everyone (mens creative)": np.full(len(split_eval), "Mens E-Mail"),
        }
        split_values, split_comparisons = evaluate_policies(
            split_eval,
            split_policies,
            split_predictions,
            propensities=NOMINAL_PROPENSITIES,
            n_boot=1,
            seed=seed,
        )
        comparison = split_comparisons.loc[
            "learned (DRPolicyForest) - email everyone (mens creative)"
        ]
        shares = pd.Series(split_learned).value_counts(normalize=True).reindex(ARMS, fill_value=0.0)
        policy_split_sensitivity.append(
            {
                "seed": seed,
                "learned_value": float(split_values.loc["learned (DRPolicyForest)", "value"]),
                "blanket_mens_value": float(
                    split_values.loc["email everyone (mens creative)", "value"]
                ),
                "learned_minus_blanket_mens": float(comparison["difference"]),
                "recommendation_shares": {arm: float(shares[arm]) for arm in ARMS},
            }
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
    contact_rates = {
        name: float(np.mean(recommendation != CONTROL_ARM))
        for name, recommendation in policies.items()
    }
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
        "schema_version": 3,
        "metadata": provenance,
        **classical,
        "figures": {"files": figure_files},
        "headline": {
            "pooled_ate": records_for_json(pooled),
            "per_arm_ate": records_for_json(per_arm),
            "deduplicated_visit_sensitivity": {
                "full_rows": int(len(df)),
                "deduplicated_rows": int(len(deduplicated)),
                "full_visit_difference": pooled_visit_difference(df),
                "deduplicated_visit_difference": pooled_visit_difference(deduplicated),
            },
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
            "repeated_split_note": (
                "repeated sample splits are sensitivity evidence, not independent replications"
            ),
            "forest_sensitivity": forest_sensitivity,
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
            "split_sensitivity": policy_split_sensitivity,
        },
        "reported_spend_sensitivity": {
            "analysis": "evaluation of a visit-optimised policy; reported spend is not profit",
            "values": records_for_json(spend_values),
            "comparisons": records_for_json(spend_comparisons),
            "contact_rate": contact_rate,
            "contact_rates": contact_rates,
            "email_cost_usd": 0.10,
            "break_even_gross_margin": break_even_margin(
                learned_spend, no_email_spend, contact_rate, 0.10
            ),
            "margin_sensitivity": margin_sensitivity,
        },
        "simulation": {
            "analysis": (
                "fully synthetic Monte Carlo; target is the sample average of known p1 - p0"
            ),
            "rows": records_for_json(run_monte_carlo()),
            "extended_rows": records_for_json(
                run_extended_monte_carlo(n_runs=full_simulation_runs)
            ),
            "extended_repetitions": full_simulation_runs,
        },
    }
    serving_payload = {
        "eval_df": eval_df[[TREATMENT_COL, "visit", "spend"]].reset_index(drop=True),
        "cate": cate,
        "policy_values": policy_values,
        "policy_comparisons": comparisons,
        "spend_policy_values": spend_values,
        "spend_policy_comparisons": spend_comparisons,
        "learned_contact_rate": contact_rate,
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
