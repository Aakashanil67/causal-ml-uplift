# Identification: what DoWhy recovers before any model is fit

Identification asks whether the causal estimand follows from the assumed causal graph and observed distribution before choosing an estimator. DoWhy evaluates the graph in `src/dag.py`; it does not inspect the outcome values.

## Result

| outcome | estimand type | backdoor variables |
|---|---|---|
| visit | EstimandType.NONPARAMETRIC_ATE | empty set |
| conversion | EstimandType.NONPARAMETRIC_ATE | empty set |
| spend | EstimandType.NONPARAMETRIC_ATE | empty set |

Under the graph's randomized-assignment assumption, the backdoor set is empty and the estimand is the assignment-group difference. The graph encodes the experiment's design; the measured balance table is a descriptive check and cannot verify the absence of unmeasured causes. Identification and estimation remain separate: this step gives the target, while regression and DML compute estimates.
