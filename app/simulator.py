"""Artifact-only Streamlit simulator for profile CATEs and held-out policy evidence."""

import sys
from pathlib import Path

# `streamlit run app/simulator.py` puts app/ on sys.path, not the project root, so `import src`
# fails unless it's added explicitly here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

# a server-side app never wants an interactive GUI backend, and the default backend on some
# machines (TkAgg) isn't thread-safe — it crashed under streamlit's AppTest, which runs the
# script in a worker thread (tests/test_simulator.py). Agg is the standard headless choice.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (
    CATEGORICAL_LEVELS,
    DEFAULT_EMAIL_COST_USD,
    TREATMENT_COL,
)
from src.persist import load_model, predict_cate_for_profile
from src.policy import incremental_net_value
from src.results import load_evaluation_artifacts, load_results
from src.uplift import qini_curve, uplift_per_email_at_k

st.set_page_config(page_title="Causal ML Uplift Simulator", layout="wide")


@st.cache_resource(show_spinner="Loading the production CATE model...")
def get_production_model():
    return load_model()


@st.cache_resource(show_spinner="Loading held-out evaluation artifacts...")
def get_eval_artifacts():
    artifact = load_evaluation_artifacts()
    return artifact["eval_df"], artifact["cate"]


@st.cache_resource(show_spinner="Loading policy evidence...")
def get_public_artifacts():
    return load_evaluation_artifacts(), load_results()


def profile_form() -> dict:
    col1, col2 = st.columns(2)
    with col1:
        recency = st.slider("Months since last purchase (recency)", 1, 12, 6)
        history = st.number_input(
            "Dollars spent in the past 12 months (history)", 29.99, 3345.93, 200.0
        )
        newbie = st.checkbox("Opened their account in the last 12 months (newbie)", value=False)
    with col2:
        mens = st.checkbox("Bought a mens product before", value=True)
        womens = st.checkbox("Bought a womens product before", value=False)
        zip_code = st.selectbox("Zip code type", CATEGORICAL_LEVELS["zip_code"], index=2)
        channel = st.selectbox("Purchase channel", CATEGORICAL_LEVELS["channel"], index=2)
    return {
        "recency": recency,
        "history": history,
        "mens": int(mens),
        "womens": int(womens),
        "newbie": int(newbie),
        "zip_code": zip_code,
        "channel": channel,
    }


def plot_cate_gauge(customer_cate: float, eval_cate: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(eval_cate, bins=40, color="#c7d5ea", edgecolor="white")
    ax.axvline(
        customer_cate, color="#b5842b", linewidth=2.5, label=f"this customer: {customer_cate:+.4f}"
    )
    ax.axvline(
        eval_cate.mean(),
        color="#5b7fb5",
        linewidth=1.2,
        linestyle="--",
        label=f"population mean: {eval_cate.mean():+.4f}",
    )
    ax.set_xlabel("estimated CATE on visit")
    ax.set_yticks([])
    ax.set_title("Where this customer sits in the held-out CATE distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def render_profile_tab():
    st.subheader("Estimated conditional average effect for a customer profile")
    st.caption(
        "Estimates the average effect for customers with this profile of assignment to the pooled "
        "email mixture versus no email; it is not an individual-level causal guarantee."
    )
    profile = profile_form()
    if not (profile["mens"] or profile["womens"]):
        st.warning(
            "This profile has no recorded product category. It is outside the observed purchase-history "
            "segments used in the main interpretation, so treat its estimate as extrapolative."
        )
    model = get_production_model()
    result = predict_cate_for_profile(model, profile)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated effect on visit probability", f"{result['cate']:+.2%}")
    m2.metric("95% CI low", f"{result['ci_low']:+.2%}")
    m3.metric("95% CI high", f"{result['ci_high']:+.2%}")

    if result["ci_low"] > 0:
        st.success(
            "This profile's conditional-average interval is entirely positive in this model."
        )
    elif result["ci_high"] < 0:
        st.warning(
            "This profile's conditional-average interval is entirely negative in this model."
        )
    else:
        st.info(
            "This profile's conditional-average interval includes zero: not distinguishable from no effect."
        )

    _eval_df, eval_cate = get_eval_artifacts()
    st.pyplot(plot_cate_gauge(result["cate"], eval_cate))
    st.caption(
        "The distribution is the held-out 30% eval set's CATE spread "
        "(`reports/07_uplift_policy.md`), not this customer's own uncertainty — the CI above "
        "already covers that."
    )


def render_policy_tab():
    st.subheader("Pooled-email mixture targeting diagnostic")
    st.caption(
        "This ranking evaluates assignment to the historical mens/womens email mixture versus no "
        "email. Use the three-action policy comparison below for a creative-specific action."
    )
    artifact, results = get_public_artifacts()
    qini = results["ranking"]["normalized_qini"]
    st.warning(
        f"Normalized Qini is {qini['value']:.4f} "
        f"[{qini['ci_low']:.4f}, {qini['ci_high']:.4f}]. The interval includes zero, so this "
        "ranking is diagnostic rather than deployment-ready."
    )
    eval_df, eval_cate = get_eval_artifacts()
    T = eval_df[TREATMENT_COL].to_numpy()
    Y = eval_df["visit"].to_numpy(dtype=float)
    n = len(eval_df)

    pct = st.slider("Target the top X% of customers by predicted uplift", 1, 100, 30)
    k = pct / 100
    n_targeted = int(n * k)
    uplift_per_email = uplift_per_email_at_k(eval_cate, T, Y, k)
    total_incremental_visits = uplift_per_email * n_targeted

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers targeted", f"{n_targeted:,} / {n:,}")
    m2.metric("Incremental visits per customer emailed", f"{uplift_per_email:+.4f}")
    m3.metric("Total incremental visits (this eval set)", f"{total_incremental_visits:+.1f}")

    curve = qini_curve(eval_cate, T, Y)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(
        curve["k_frac"] * 100, curve["gain"], color="#5b7fb5", label="ranked by predicted uplift"
    )
    ax.plot(
        curve["k_frac"] * 100,
        curve["random_line"],
        color="#9aa5b1",
        linestyle="--",
        label="random targeting",
    )
    ax.axvline(pct, color="#b5842b", linewidth=1.2)
    ax.set_xlabel("% of customers targeted (top-k by predicted uplift)")
    ax.set_ylabel("cumulative incremental visits")
    ax.set_title("Qini curve")
    ax.legend(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)

    est_cost = n_targeted * DEFAULT_EMAIL_COST_USD
    # a lone literal "$" is safe in Streamlit markdown, but two "$" in one string get parsed as
    # a LaTeX math span between them and silently swallow everything in between — escape both.
    st.caption(
        f"At the assumed \\${DEFAULT_EMAIL_COST_USD:.2f}/email "
        "(`src/config.py:DEFAULT_EMAIL_COST_USD`), "
        f"emailing this segment costs an estimated \\${est_cost:,.2f}."
    )

    st.divider()
    st.subheader("Three-action policy: no email, mens email, or womens email")
    st.caption(
        "Cross-fitted doubly robust values on the held-out eval set. The paired differences, not "
        "the ordering of point estimates, determine whether one policy beats another."
    )
    values = artifact["policy_values"].copy()
    values["95% CI"] = values.apply(
        lambda row: f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]", axis=1
    )
    st.dataframe(values[["value", "95% CI"]], width="stretch")

    shares = pd.Series(results["policy"]["recommendation_shares"], name="share")
    st.caption("Learned-policy action shares on the held-out evaluation set")
    st.dataframe(shares.to_frame().style.format("{:.1%}"), width="stretch")

    comparisons = artifact["policy_comparisons"].copy()
    comparisons["95% CI"] = comparisons.apply(
        lambda row: f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]", axis=1
    )
    st.dataframe(comparisons[["difference", "95% CI"]], width="stretch")
    st.info(results["policy"]["conclusion"])

    st.divider()
    st.subheader("Reported-spend sensitivity")
    st.warning(
        "Spend is reported and capped in the source data. The learned policy was trained to "
        "maximise visits, not profit; this calculation is a margin sensitivity, not a profit "
        "estimate."
    )
    margin = st.slider("Assumed gross margin", 0, 100, 30) / 100
    email_cost = st.number_input(
        "Cost per email", min_value=0.0, value=DEFAULT_EMAIL_COST_USD, step=0.01
    )
    spend_values = results["reported_spend_sensitivity"]["values"]
    spend_by_policy = {row["policy"]: row["value"] for row in spend_values}
    learned_spend = spend_by_policy["learned (DRPolicyForest)"]
    no_email_spend = spend_by_policy["email nobody"]
    contact_rate = results["reported_spend_sensitivity"]["contact_rate"]
    contribution = incremental_net_value(
        learned_spend, no_email_spend, contact_rate, margin, email_cost
    )
    st.metric("Estimated incremental contribution per customer", f"${contribution:+.4f}")
    st.caption(
        f"Uses stored reported-spend values of ${learned_spend:.4f} for the learned policy and "
        f"${no_email_spend:.4f} for email nobody, with a {contact_rate:.1%} learned contact rate."
    )


def main():
    st.title("Causal ML Uplift Simulator")
    st.caption(
        "Hillstrom e-mail experiment, DoWhy + EconML. "
        "[Full report and methodology](https://github.com/Aakashanil67/causal-ml-uplift)."
    )
    tab1, tab2 = st.tabs(["Customer profile", "Targeting policy"])
    with tab1:
        render_profile_tab()
    with tab2:
        render_policy_tab()


if __name__ == "__main__":
    main()
