# Identification: what DoWhy recovers before any model is fit

**Identification** asks whether a causal quantity can be written as a function of the
*observed data distribution* at all, given the assumed graph, before any estimator ever
touches the data. **Estimation** asks how to compute that function's value once it is
known to exist. Getting the order backwards, picking an estimator first and hoping it's
estimating something real, is the mistake this step exists to prevent. DoWhy's
`identify_effect()` runs entirely off the graph in `src/dag.py`; it never looks at outcome
values.

## Result, for all three outcomes

Run separately for `visit`, `conversion` and `spend`, all three of the 64,000-customer
outcomes resolve to the same backdoor estimand with an **empty adjustment set**:

```
d
── (E[outcome])
d[treatment]
```

with the single assumption DoWhy states explicitly: unconfoundedness, that no unobserved
variable `U` causes both `treatment` and the outcome. An empty adjustment set is not DoWhy
giving up. It is DoWhy correctly reading the graph in `src/dag.py`, where nothing points
into `treatment`, and concluding that no covariate needs to be controlled for to block a
backdoor path, because there is no backdoor path. `E[visit | treatment=1] − E[visit |
treatment=0]` is already the correct nonparametric estimand. That is the formal, graph-
level version of the empirical balance check in `reports/02_naive_estimate.md`, where the
largest of 11 covariate standardised differences was 0.0088, an order of magnitude under
the 0.1 threshold. Both are saying the naive comparison is unbiased, one from the assumed
graph, one from the observed covariates.

The `iv` (instrumental variable) and `frontdoor` estimands both return "No such
variable(s) found", correctly: this graph has no instrument and no mediator specified for
either route, and none is needed when the backdoor route is already clean.

## What the unconfoundedness assumption actually claims here

Unconfoundedness is doing real work in a typical observational study; it is close to free
here because it follows from a stronger, checkable fact: `treatment` was assigned by a
random-number generator, not by anything about the customer. A variable determined purely
by randomisation cannot share an unobserved common cause with the outcome, since it has no
cause at all beyond the randomiser. This is why the graph in `src/dag.py` draws no edges
into `treatment`, and why that specific structural choice is the one thing in this whole
project that is not a modelling judgement call: it is a design fact about how Hillstrom
ran the experiment, and `reports/02_naive_estimate.md`'s balance table is the empirical
test of whether it held.

## Where this stops being free

`src/confounded.py` builds a second dataset from the same file where treatment assignment
is deliberately correlated with `recency` and `history`. Identified on that graph, the
backdoor adjustment set is no longer empty: it has to include every variable that causes
both the (now non-random) assignment and the outcome, and estimating the effect without
adjusting for them is exactly the naive-estimate bias this project is built to show, with a
number attached rather than only argued in prose.
