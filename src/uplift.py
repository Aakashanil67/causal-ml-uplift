"""Uplift ranking, Qini evaluation, and targeting economics.

The cumulative gain curve and raw Qini area are implemented locally. Normalized Qini uses
`scikit-uplift` as the reference implementation and is covered by fixed-fixture tests.

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


def normalized_qini_score(
    cate_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray
) -> float:
    """Normalized Qini area, comparable across evaluation-sample sizes."""
    from sklift.metrics import qini_auc_score

    return float(qini_auc_score(outcome, cate_scores, treatment))


def bootstrap_normalized_qini_ci(
    cate_scores: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Fixed-model conditional interval for normalized Qini via arm-stratified bootstrap."""
    scores = np.asarray(cate_scores)
    T = np.asarray(treatment)
    Y = np.asarray(outcome)
    if not (len(scores) == len(T) == len(Y)):
        raise ValueError("cate_scores, treatment, and outcome must have the same length.")
    arms = [np.flatnonzero(T == arm) for arm in np.unique(T)]
    if len(arms) != 2:
        raise ValueError("Normalized Qini requires both treatment arms.")

    point = normalized_qini_score(scores, T, Y)
    rng = np.random.default_rng(seed)
    estimates = []
    attempts = 0
    while len(estimates) < n_boot and attempts < n_boot * 20:
        idx = np.concatenate([rng.choice(arm, len(arm), replace=True) for arm in arms])
        attempts += 1
        try:
            estimate = normalized_qini_score(scores[idx], T[idx], Y[idx])
        except ValueError:
            continue
        if np.isfinite(estimate):
            estimates.append(estimate)
    if len(estimates) < n_boot:
        raise ValueError("Unable to draw enough finite normalized-Qini bootstrap samples.")
    low, high = np.percentile(estimates, [2.5, 97.5])
    return point, float(low), float(high)


def summarize_repeated_qini(runs: pd.DataFrame) -> dict[str, float]:
    """Summarize split-level normalized Qini scores without hiding their range."""
    values = runs["normalized_qini"]
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


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
    """Fixed-ranking conditional interval for a top-k pooled-email effect."""
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
    ax.set_ylabel("raw Qini gain (treated-count-rescaled visits)")
    ax.set_title(f"Qini curve (coefficient: {qini_coef:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    raise SystemExit("Run `python -m src.pipeline` to rebuild the published uplift outputs.")


if __name__ == "__main__":
    main()
