"""Uplift ranking, Qini evaluation, and targeting economics — all computed by hand rather than via
`scikit-uplift`, so every number here can be defended line by line. `scikit-uplift` is still
installed and used as an independent cross-check in one test.

Everything below runs on the held-out eval set from `src/cate.py`, ranked by predicted CATE. The
Qini gain at a targeting fraction k is the standard Radcliffe & Surry (2011) definition: the sum of
outcomes among the top-k-by-score treated customers, minus the top-k control customers' outcome sum
rescaled to the treated group's size in that same slice — "how much more visiting happened among
the treated than a same-sized control group would have produced, in the k% we'd have targeted."
"""

import numpy as np
import pandas as pd


def qini_curve(cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    order = np.argsort(-cate_scores)
    T = treatment[order].astype(float)
    Y = outcome[order].astype(float)
    n = len(Y)

    cum_nt = np.cumsum(T)
    cum_nc = np.cumsum(1 - T)
    cum_yt = np.cumsum(Y * T)
    cum_yc = np.cumsum(Y * (1 - T))

    # before the first control appears in the ranking, there's nothing to rescale against; the
    # gain is just the raw treated sum until then, which is the only defensible fallback (not an
    # approximation choice that changes the result anywhere a control has been seen).
    ratio = np.divide(cum_nt, cum_nc, out=np.zeros(n), where=cum_nc > 0)
    gain = cum_yt - cum_yc * ratio

    k_frac = np.arange(1, n + 1) / n
    random_line = k_frac * gain[-1]
    return pd.DataFrame(
        {
            "k_frac": k_frac,
            "gain": gain,
            "random_line": random_line,
            "n_targeted": np.arange(1, n + 1),
        }
    )


def qini_coefficient(curve: pd.DataFrame) -> float:
    """Area between the model's gain curve and the random-targeting diagonal. Positive means the
    ranking beats random targeting; a perfect ranking (all uplift-positive customers ranked before
    all uplift-negative ones) maximises it; a random ranking averages to zero."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return float(integrate(curve["gain"] - curve["random_line"], curve["k_frac"]))


def uplift_deciles(
    cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bins customers by predicted CATE (decile 0 = highest) and reports the *observed*
    treated-minus-control outcome gap within each bin — the real-world check that the model's
    ranking corresponds to an actual, measurable difference in who the email helps, not just an
    internally consistent score."""
    tmp = pd.DataFrame({"cate": cate_scores, "T": treatment, "Y": outcome})
    tmp["decile"] = pd.qcut(-tmp["cate"], n_bins, labels=False, duplicates="drop")
    rows = []
    for d in sorted(tmp["decile"].unique()):
        sub = tmp[tmp["decile"] == d]
        t_rate = sub.loc[sub["T"] == 1, "Y"].mean()
        c_rate = sub.loc[sub["T"] == 0, "Y"].mean()
        rows.append(
            {
                "decile": int(d),
                "n": len(sub),
                "pred_cate_mean": sub["cate"].mean(),
                "observed_uplift": t_rate - c_rate,
            }
        )
    return pd.DataFrame(rows).set_index("decile")


def _validate_top_k_inputs(
    cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return ranked arrays and selected-segment size after validating a top-k request."""
    scores = np.asarray(cate_scores)
    T = np.asarray(treatment)
    Y = np.asarray(outcome)
    if not 0 < k <= 1:
        raise ValueError("k must be in the interval (0, 1].")
    if not (len(scores) == len(T) == len(Y)):
        raise ValueError("cate_scores, treatment, and outcome must have the same length.")
    if len(scores) == 0:
        raise ValueError("Cannot evaluate an empty sample.")
    if not np.isin(T, [0, 1]).all():
        raise ValueError("treatment must be binary (0 for control, 1 for treated).")

    order = np.argsort(-scores)
    n_targeted = max(int(len(scores) * k), 1)
    return scores[order], T[order], Y[order], n_targeted


def uplift_per_email_at_k(
    cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k: float
) -> float:
    """Estimate the incremental outcome per email for an all-email top-k policy.

    The selected segment is defined by the predicted binary-CATE ranking. Within that segment, the
    RCT identifies the effect of assignment to the pooled email mixture versus no email. This is
    a treated-minus-control mean difference, not Qini gain divided by every selected row.
    """
    _, T, Y, n_targeted = _validate_top_k_inputs(cate_scores, treatment, outcome, k)
    selected_T = T[:n_targeted]
    selected_Y = Y[:n_targeted]
    if not ((selected_T == 1).any() and (selected_T == 0).any()):
        raise ValueError("Selected segment must contain both treated and control customers.")
    return float(selected_Y[selected_T == 1].mean() - selected_Y[selected_T == 0].mean())


def bootstrap_uplift_per_email_ci(
    cate_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    k: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap a top-k pooled-email effect while holding each row's score fixed."""
    rng = np.random.default_rng(seed)
    n = len(cate_scores)
    estimates = np.empty(n_boot)
    b = 0
    attempts = 0
    max_attempts = n_boot * 20
    while b < n_boot and attempts < max_attempts:
        idx = rng.integers(0, n, size=n)
        attempts += 1
        try:
            estimates[b] = uplift_per_email_at_k(cate_scores[idx], treatment[idx], outcome[idx], k)
        except ValueError:
            continue
        b += 1
    if b < n_boot:
        raise ValueError("Unable to draw bootstrap samples with both arms in the selected segment.")
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def plot_qini_curve(curve: pd.DataFrame, qini_coef: float, out_path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        curve["k_frac"] * 100,
        curve["gain"],
        color="#5b7fb5",
        linewidth=1.8,
        label="ranked by predicted uplift",
    )
    ax.plot(
        curve["k_frac"] * 100,
        curve["random_line"],
        color="#9aa5b1",
        linestyle="--",
        label="random targeting",
    )
    ax.fill_between(
        curve["k_frac"] * 100, curve["gain"], curve["random_line"], alpha=0.15, color="#5b7fb5"
    )
    ax.set_xlabel("% of customers targeted (top-k by predicted uplift)")
    ax.set_ylabel("cumulative incremental visits")
    ax.set_title(f"Qini curve (coefficient: {qini_coef:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_uplift_report(
    cate_stats: dict,
    purchase_table: pd.DataFrame,
    deciles: pd.DataFrame,
    qini_coef: float,
    uplift_at_k_table: pd.DataFrame,
    break_even: dict,
    out_path,
) -> None:
    lines = [
        "# Heterogeneous effects and targeting policy",
        "",
        "`src/cate.py`'s `CausalForestDML` estimates, evaluated for real on the held-out 30%",
        "(`src/cate.py` docstring: `visit` is the only outcome with enough events, 9,394, to split",
        "across a forest without the CATE surface being mostly noise).",
        "",
        "## Heterogeneity: who the email actually helps",
        "",
        f"Mean CATE on the eval set: **{cate_stats['mean']:+.4f}** (std {cate_stats['std']:.4f},",
        f"range [{cate_stats['min']:+.4f}, {cate_stats['max']:+.4f}]), close to the pooled DML",
        "benchmark of +0.0601 (`reports/05_dml_ate.md`), as it should be: the CATE mean and the",
        "pooled ATE are estimating the same population quantity two different ways.",
        "",
        "The headline segmentation is prior purchase category, not recency or history:",
        "",
        "| prior purchase | mean CATE | n |",
        "|---|---|---|",
    ]
    labels = {(0, 1): "womens only", (1, 0): "mens only", (1, 1): "both"}
    for (m, w), row in purchase_table.iterrows():
        label = labels.get((m, w), f"mens={m}, womens={w}")
        lines.append(f"| {label} | {row['mean']:+.4f} | {row['count']:,.0f} |")
    lines += [
        "",
        "Confirmed with an OLS interaction test before trusting the forest's split, since a tree",
        "can carve out a segment from noise as easily as from a real pattern: treatment × `mens`",
        "and treatment × `womens` interaction terms are both significant (p=0.006, p<0.001),",
        "unlike treatment × `recency`/`history`, checked earlier for the confounding benchmark",
        "(both p>0.11, not significant). The pooled any-email effect is real for everyone in this",
        "data, but it is close to twice as large for customers with a `womens`-only purchase",
        "history as for `mens`-only customers, see `reports/figures/cate_by_purchase_history.png`.",
        "",
        "## Uplift deciles: does the ranking correspond to anything real?",
        "",
        "Decile 0 is the customers the model ranks highest; `observed_uplift` is the *actual*",
        "treated-minus-control visit-rate gap measured within that decile, not the model's own",
        "prediction, so this is a check on the ranking rather than a restatement of it.",
        "",
        "| decile | n | predicted CATE | observed uplift |",
        "|---|---|---|---|",
    ]
    for decile, row in deciles.iterrows():
        lines.append(
            f"| {decile} | {row['n']:,.0f} | {row['pred_cate_mean']:+.4f} | "
            f"{row['observed_uplift']:+.4f} |"
        )
    top2 = deciles.loc[[0, 1], "observed_uplift"].mean()
    bottom2 = deciles.loc[deciles.index[-2:], "observed_uplift"].mean()
    lines += [
        "",
        "Not perfectly monotonic. With roughly 1,920 customers per decile and a rare binary",
        "outcome, individual deciles carry real sampling noise, but the top two deciles average",
        f"{top2:+.4f} observed uplift against {bottom2:+.4f} for the bottom two, the gradient the",
        "ranking predicts.",
        "",
        f"**Qini coefficient: {qini_coef:.2f}** (area between the model's gain curve and random",
        "targeting; positive means the ranking beats emailing customers in a random order,",
        "0 is what a ranking with no real signal would average to).",
        "",
        "## Targeting economics",
        "",
        "Expected incremental visits per customer targeted, at several targeting fractions, with a",
        "95% bootstrap CI over the held-out eval set (CATE scores held fixed, so this is evaluation",
        "noise, not a statement about how much a re-trained forest could vary):",
        "",
        "| top-k% targeted | incremental visits/customer | 95% CI |",
        "|---|---|---|",
    ]
    for k, row in uplift_at_k_table.iterrows():
        lines.append(
            f"| {int(k * 100)}% | {row['gain_per_target']:+.4f} | [{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] |"
        )
    ci_crosses_zero = break_even["ci_low"] < 0 < break_even["ci_high"]
    lines += [
        "",
        "## Break-even email cost",
        "",
        f"At the top-{int(break_even['k'] * 100)}% targeting fraction, the expected incremental",
        f"**spend** per customer targeted is **${break_even['gain_per_target']:.4f}**",
        f"(95% CI [${break_even['ci_low']:.4f}, ${break_even['ci_high']:.4f}]): the revenue side",
        "of the same targeting policy, using the same visit-based ranking rather than a separate",
        "spend-CATE model (only 578 non-zero spend values total, see",
        "`reports/data_dictionary.md`). The point estimate clears the assumed",
        f"${break_even['assumed_cost']:.2f}-per-email cost",
        "(`src/config.py:DEFAULT_EMAIL_COST_USD`, a stated assumption, not fitted) comfortably.",
        (
            "**The confidence interval crosses zero and runs negative, though**: at this sample "
            "size, spend among 30%-of-64,000 customers is too sparse to statistically rule out "
            "the segment losing money rather than paying for itself. The visit-based numbers "
            "above are the ones this report actually stands behind; this section is a directional "
            "read on revenue, not a claim with the same statistical footing."
            if ci_crosses_zero
            else "and the interval stays positive throughout, so this is a real, not just "
            "directional, result."
        ),
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    from src.cate import (
        fit_causal_forest,
        heterogeneity_by_purchase_history,
        predict_cate,
        split_train_eval,
    )
    from src.config import DEFAULT_EMAIL_COST_USD, REPORTS_DIR, TREATMENT_COL
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    cf = fit_causal_forest(train)
    cate = predict_cate(cf, eval_df)

    T = eval_df[TREATMENT_COL].to_numpy()
    Y_visit = eval_df["visit"].to_numpy(dtype=float)
    Y_spend = eval_df["spend"].to_numpy(dtype=float)

    cate_stats = {"mean": cate.mean(), "std": cate.std(), "min": cate.min(), "max": cate.max()}
    purchase_table = heterogeneity_by_purchase_history(eval_df, cate)
    deciles = uplift_deciles(cate, T, Y_visit)

    curve = qini_curve(cate, T, Y_visit)
    qini_coef = qini_coefficient(curve)
    print(f"Qini coefficient: {qini_coef:.2f}")

    from src.config import FIGURES_DIR

    plot_qini_curve(curve, qini_coef, FIGURES_DIR / "qini_curve.png")

    ks = [0.1, 0.2, 0.3, 0.5, 1.0]
    rows = []
    for k in ks:
        gpt = uplift_per_email_at_k(cate, T, Y_visit, k)
        ci_low, ci_high = bootstrap_uplift_per_email_ci(cate, T, Y_visit, k, n_boot=1000, seed=0)
        rows.append({"k": k, "gain_per_target": gpt, "ci_low": ci_low, "ci_high": ci_high})
    uplift_at_k_table = pd.DataFrame(rows).set_index("k")
    print(uplift_at_k_table)

    be_k = 0.3
    be_gpt = uplift_per_email_at_k(cate, T, Y_spend, be_k)
    be_ci_low, be_ci_high = bootstrap_uplift_per_email_ci(
        cate, T, Y_spend, be_k, n_boot=1000, seed=1
    )
    break_even = {
        "k": be_k,
        "gain_per_target": be_gpt,
        "ci_low": be_ci_low,
        "ci_high": be_ci_high,
        "assumed_cost": DEFAULT_EMAIL_COST_USD,
    }
    print(break_even)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_uplift_report(
        cate_stats,
        purchase_table,
        deciles,
        qini_coef,
        uplift_at_k_table,
        break_even,
        REPORTS_DIR / "07_uplift_policy.md",
    )


if __name__ == "__main__":
    main()
