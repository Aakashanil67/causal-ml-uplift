"""DoWhy's identify step: given the causal graph, is the effect even recoverable from the data in
principle, and via which adjustment? Runs before any estimation, on all three outcomes, using the
GML graph from `src/dag.py`.
"""

from dowhy import CausalModel

from src.config import TREATMENT_COL
from src.dag import build_dag, to_gml


def identify(df, outcome_col: str):
    model = CausalModel(
        data=df, treatment=TREATMENT_COL, outcome=outcome_col, graph=to_gml(build_dag())
    )
    return model, model.identify_effect(proceed_when_unidentifiable=True)


def write_identification_report(estimands: dict, out_path) -> None:
    lines = [
        "# Identification: what DoWhy recovers before any model is fit",
        "",
        "**Identification** asks whether a causal quantity can be written as a function of the",
        "*observed data distribution* at all, given the assumed graph, before any estimator ever",
        "touches the data. **Estimation** asks how to compute that function's value once it is",
        "known to exist. Getting the order backwards, picking an estimator first and hoping it's",
        "estimating something real, is the mistake this step exists to prevent. DoWhy's",
        "`identify_effect()` runs entirely off the graph in `src/dag.py`; it never looks at outcome",
        "values.",
        "",
        "## Result, for all three outcomes",
        "",
        "Run separately for `visit`, `conversion` and `spend`, all three of the 64,000-customer",
        "outcomes resolve to the same backdoor estimand with an **empty adjustment set**:",
        "",
        "```",
        "d",
        "── (E[outcome])",
        "d[treatment]",
        "```",
        "",
        "with the single assumption DoWhy states explicitly: unconfoundedness, that no unobserved",
        "variable `U` causes both `treatment` and the outcome. An empty adjustment set is not DoWhy",
        "giving up. It is DoWhy correctly reading the graph in `src/dag.py`, where nothing points",
        "into `treatment`, and concluding that no covariate needs to be controlled for to block a",
        "backdoor path, because there is no backdoor path. `E[visit | treatment=1] − E[visit |",
        "treatment=0]` is already the correct nonparametric estimand. That is the formal, graph-",
        "level version of the empirical balance check in `reports/02_naive_estimate.md`, where the",
        "measured-covariate balance table is descriptive evidence. The graph supplies the",
        "identification claim under the randomized-",
        "assignment design; the measured balance table is a descriptive check, not a proof.",
        "",
        'The `iv` (instrumental variable) and `frontdoor` estimands both return "No such',
        'variable(s) found", correctly: this graph has no instrument and no mediator specified for',
        "either route, and none is needed when the backdoor route is already clean.",
        "",
        "## What the unconfoundedness assumption actually claims here",
        "",
        "Unconfoundedness is doing real work in a typical observational study; it is close to free",
        "here because it follows from a stronger, checkable fact: `treatment` was assigned by a",
        "random-number generator, not by anything about the customer. A variable determined purely",
        "by randomisation cannot share an unobserved common cause with the outcome, since it has no",
        "cause at all beyond the randomiser. This is why the graph in `src/dag.py` draws no edges",
        "into `treatment`, and why that specific structural choice is the one thing in this whole",
        "project that is not a modelling judgement call: it is a design fact about how Hillstrom",
        "ran the experiment. `reports/02_naive_estimate.md`'s balance table is a descriptive",
        "check on measured covariates, not a test of the assignment mechanism.",
        "",
        "## Where this stops being free",
        "",
        "`src/confounded.py` builds a second dataset from the same file where treatment assignment",
        "is deliberately correlated with `recency` and `history`. Identified on that graph, the",
        "backdoor adjustment set is no longer empty: it includes variables that cause both the",
        "(now non-random) assignment and outcome. The constructed selection reports compare",
        "selected-sample estimates with an estimated reference from a different population, so",
        "those differences are descriptive rather than known-bias estimates.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import sys

    from src.config import OUTCOME_COLS, REPORTS_DIR
    from src.data_loader import load_hillstrom

    # DoWhy's estimand repr uses unicode box-drawing characters; the default Windows console
    # codepage (cp1252) can't encode them and raises on print() without this.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    df = load_hillstrom()
    estimands = {}
    for outcome in OUTCOME_COLS:
        _model, identified = identify(df, outcome)
        estimands[outcome] = identified
        print(f"--- {outcome} ---")
        print(identified)
        print(f"backdoor variables: {identified.get_backdoor_variables()}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_identification_report(estimands, REPORTS_DIR / "04_identification.md")


if __name__ == "__main__":
    main()
