"""What-if simulator: a customer-profile tab (per-customer CATE with CI, plotted against the
held-out CATE distribution) and a targeting-policy tab (email top X% by uplift, expected
incremental visits, and the three-arm policy comparison from `reports/07_uplift_policy.md`).

The heavy fits (`CausalForestDML`, `DRPolicyForest`) run once per server process via
`st.cache_resource`, not on every slider move — the first load after a cold start takes ~30s, every
interaction after that is instant. Tab 1 predicts from the production artifact
(`models/causal_forest.joblib`, fit on all 64,000 rows); tab 2 evaluates on the held-out 30% split,
matching the honest numbers already in the reports rather than a different, easier-to-compute one.
"""

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

from src.cate import fit_causal_forest, predict_cate, split_train_eval
from src.config import (
    ARM_COL,
    CATEGORICAL_LEVELS,
    CONTROL_ARM,
    DEFAULT_EMAIL_COST_USD,
    TREATMENT_COL,
)
from src.data_loader import load_hillstrom
from src.persist import load_model, predict_cate_for_profile
from src.policy import (
    fit_policy_forest,
    heuristic_recommendations,
    ipw_policy_value,
    policy_forest_recommendations,
)
from src.uplift import gain_per_target_at_k, qini_curve

st.set_page_config(page_title="Causal ML Uplift Simulator", layout="wide")


@st.cache_resource(show_spinner="Loading the production CATE model...")
def get_production_model():
    return load_model()


@st.cache_resource(show_spinner="Fitting the evaluation split (~30s on a cold start)...")
def get_eval_artifacts():
    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    cf = fit_causal_forest(train)
    cate = predict_cate(cf, eval_df)
    return eval_df.reset_index(drop=True), cate


@st.cache_resource(show_spinner="Fitting the three-arm policy forest...")
def get_policy_artifacts():
    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    pf = fit_policy_forest(train)
    propensities = df[ARM_COL].value_counts(normalize=True).to_dict()
    return eval_df.reset_index(drop=True), pf, propensities


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
    st.subheader("Estimated treatment effect for one customer")
    st.caption(
        "Predicts the effect of sending this customer any email (mens or womens) on their "
        "probability of visiting the site, from the production `CausalForestDML` model."
    )
    profile = profile_form()
    model = get_production_model()
    result = predict_cate_for_profile(model, profile)

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated effect on visit probability", f"{result['cate']:+.2%}")
    m2.metric("95% CI low", f"{result['ci_low']:+.2%}")
    m3.metric("95% CI high", f"{result['ci_high']:+.2%}")

    if result["ci_low"] > 0:
        st.success(
            "This customer's confidence interval is entirely positive: the email likely helps."
        )
    elif result["ci_high"] < 0:
        st.warning("This customer's confidence interval is entirely negative: the email may hurt.")
    else:
        st.info(
            "This customer's confidence interval includes zero: not distinguishable from no effect."
        )

    _eval_df, eval_cate = get_eval_artifacts()
    st.pyplot(plot_cate_gauge(result["cate"], eval_cate))
    st.caption(
        "The distribution is the held-out 30% eval set's CATE spread "
        "(`reports/07_uplift_policy.md`), not this customer's own uncertainty — the CI above "
        "already covers that."
    )


def render_policy_tab():
    st.subheader("Targeting policy: email the top X% by predicted uplift")
    eval_df, eval_cate = get_eval_artifacts()
    T = eval_df[TREATMENT_COL].to_numpy()
    Y = eval_df["visit"].to_numpy(dtype=float)
    n = len(eval_df)

    pct = st.slider("Target the top X% of customers by predicted uplift", 1, 100, 30)
    k = pct / 100
    n_targeted = int(n * k)
    gain_per_target = gain_per_target_at_k(eval_cate, T, Y, k)
    total_incremental_visits = gain_per_target * n_targeted

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers targeted", f"{n_targeted:,} / {n:,}")
    m2.metric("Incremental visits per customer emailed", f"{gain_per_target:+.4f}")
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
        "The learned per-customer policy against three baselines, all scored the same way on the "
        "held-out eval set (`reports/07_uplift_policy.md`)."
    )
    policy_eval_df, pf, propensities = get_policy_artifacts()
    learned_rec = policy_forest_recommendations(pf, policy_eval_df)
    heuristic_rec = heuristic_recommendations(policy_eval_df)
    mens_rec = np.full(len(policy_eval_df), "Mens E-Mail")
    none_rec = np.full(len(policy_eval_df), CONTROL_ARM)

    rows = []
    for name, rec in [
        ("learned (DRPolicyForest)", learned_rec),
        ("purchase-history heuristic", heuristic_rec),
        ("email everyone (mens creative)", mens_rec),
        ("email nobody", none_rec),
    ]:
        value = ipw_policy_value(policy_eval_df, rec, "visit", propensities)
        rows.append({"policy": name, "expected visit rate": value})
    st.dataframe(pd.DataFrame(rows).set_index("policy"), width="stretch")


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
