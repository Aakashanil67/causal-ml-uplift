"""Profile-level conditional average treatment effects via CausalForestDML, on `visit` only.

`reports/data_dictionary.md` is why: `conversion`/`spend` have 578 events total across 64,000
rows, nowhere near enough to split across a forest's leaves without the CATE surface being mostly
noise. `visit` has 9,394. Fit on a 70% train split, every number reported comes from the held-out
30% the forest never saw — the same discipline `src/uplift.py` and `src/policy.py` reuse for their
own held-out evaluation.
"""

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split

from src.config import ARM_COL, EVAL_FRACTION, RANDOM_SEED, TREATMENT_COL
from src.data_loader import build_covariate_matrix


def split_train_eval(
    df: pd.DataFrame, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified on arm and visit jointly, not just one or the other, so both the treatment mix
    and the (rare) outcome rate are preserved in both halves."""
    strat_key = df[ARM_COL].astype(str) + "_" + df["visit"].astype(str)
    train_idx, eval_idx = train_test_split(
        df.index, test_size=EVAL_FRACTION, stratify=strat_key, random_state=seed
    )
    return df.loc[train_idx].reset_index(drop=True), df.loc[eval_idx].reset_index(drop=True)


def fit_causal_forest(train: pd.DataFrame, seed: int = RANDOM_SEED) -> CausalForestDML:
    X = build_covariate_matrix(train).to_numpy()
    T = train[TREATMENT_COL].to_numpy()
    Y = train["visit"].to_numpy(dtype=float)
    cf = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=100, verbose=-1, random_state=seed),
        model_t=LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed),
        discrete_treatment=True,
        cv=3,
        n_estimators=500,
        min_samples_leaf=50,
        random_state=seed,
    )
    cf.fit(Y, T, X=X)
    return cf


def predict_cate(cf: CausalForestDML, df: pd.DataFrame) -> np.ndarray:
    X = build_covariate_matrix(df).to_numpy()
    return cf.effect(X)


def heterogeneity_by_bin(eval_df: pd.DataFrame, cate: np.ndarray, covariate: str, bins: int = 4):
    tmp = eval_df.copy()
    tmp["cate"] = cate
    tmp["_bin"] = pd.qcut(tmp[covariate], bins, duplicates="drop", precision=0)
    return tmp.groupby("_bin", observed=True)["cate"].agg(["mean", "count"])


def heterogeneity_by_purchase_history(eval_df: pd.DataFrame, cate: np.ndarray) -> pd.DataFrame:
    """The headline heterogeneity finding: `mens`/`womens` purchase-history interacts with
    treatment far more strongly than `recency`/`history` do — confirmed with an OLS interaction
    test before trusting the forest's split, not just read off the CATE distribution (T×mens
    p=0.006, T×womens p<0.001; see reports/07_uplift_policy.md)."""
    tmp = eval_df.copy()
    tmp["cate"] = cate
    return tmp.groupby(["mens", "womens"])["cate"].agg(["mean", "count"])


def plot_cate_distribution(cate: np.ndarray, out_path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(cate, bins=40, color="#5b7fb5", edgecolor="white")
    ax.axvline(cate.mean(), color="#b5842b", linewidth=2, label=f"mean {cate.mean():.4f}")
    ax.set_xlabel("estimated CATE on visit")
    ax.set_ylabel("customers (held-out eval set)")
    ax.set_title("Distribution of profile-level conditional average treatment effects")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heterogeneity_by_purchase_history(purchase_table: pd.DataFrame, out_path) -> None:
    import matplotlib.pyplot as plt

    def _label(m, w):
        if m and w:
            return "both"
        if m:
            return "mens only"
        if w:
            return "womens only"
        return "neither"  # doesn't occur in this data — every customer bought mens or womens

    labels = [_label(m, w) for m, w in purchase_table.index]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, purchase_table["mean"], color="#4a8c4a")
    ax.set_ylabel("mean CATE on visit")
    ax.set_title("CATE by prior purchase category")
    for i, row in enumerate(purchase_table.itertuples()):
        ax.text(
            i, row.mean, f"{row.mean:.3f}\n(n={row.count})", ha="center", va="bottom", fontsize=8
        )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heterogeneity_by_covariate(table: pd.DataFrame, covariate: str, out_path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [str(b) for b in table.index]
    ax.bar(labels, table["mean"], color="#5b7fb5")
    ax.set_ylabel("mean CATE on visit")
    ax.set_title(f"CATE by {covariate} quartile")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    from src.config import FIGURES_DIR
    from src.data_loader import load_hillstrom

    df = load_hillstrom()
    train, eval_df = split_train_eval(df)
    print(f"train: {len(train)}, eval: {len(eval_df)}")

    cf = fit_causal_forest(train)
    cate = predict_cate(cf, eval_df)
    print(
        f"CATE: mean={cate.mean():.4f} std={cate.std():.4f} min={cate.min():.4f} max={cate.max():.4f}"
    )

    purchase_table = heterogeneity_by_purchase_history(eval_df, cate)
    print(purchase_table)
    recency_table = heterogeneity_by_bin(eval_df, cate, "recency")
    history_table = heterogeneity_by_bin(eval_df, cate, "history")

    plot_cate_distribution(cate, FIGURES_DIR / "cate_distribution.png")
    plot_heterogeneity_by_purchase_history(
        purchase_table, FIGURES_DIR / "cate_by_purchase_history.png"
    )
    plot_heterogeneity_by_covariate(recency_table, "recency", FIGURES_DIR / "cate_by_recency.png")
    plot_heterogeneity_by_covariate(history_table, "history", FIGURES_DIR / "cate_by_history.png")
    print(f"figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
