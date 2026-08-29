# Constructed confounding stress test

Every method so far agreed because Hillstrom is randomised and there was nothing to
disagree about. This section manufactures real confounding from the same data and checks
whether each estimator can see through it, against a benchmark we actually know: the
pooled DML ATE on `visit` from the full, unconfounded RCT, **+6.01pp**
(`reports/05_dml_ate.md`).

**What this validates and what it doesn't.** Both variants below confound on named,
real covariates from this dataset. Variant 1 gives every estimator access to the
confounder; variant 2 deliberately withholds it. Passing variant 1 shows DML can correct
for confounding *it can see*. Failing variant 2 shows, honestly, that it cannot correct
for confounding it cannot see, which is a limitation of every method here, DML included,
not a defect specific to it. Neither variant says anything about whether unobserved
confounding exists in the real, unconfounded Hillstrom data; the whole point of using an
RCT for the rest of this project is that it doesn't need to.

## Variant 1: selection on observables (`recency`, `history`)

Retained 31,592 of 64,000 customers. Treated customers were kept preferentially
when recent and high-spending, and control customers preferentially when lapsed and
low-spending: a plausible stand-in for a marketer targeting the best customers. Both
confounders stay in every estimator's covariate matrix.

| estimator | ATE on visit | vs benchmark |
|---|---|---|
| naive diff-in-means | +10.03pp | off by +4.02pp |
| OLS, `recency`+`history` as controls | +5.93pp | off by -0.09pp |
| LinearDML | +5.99pp [+4.79pp, +7.19pp] | benchmark inside the CI |

The naive estimate overstates the true effect by roughly two-thirds of its own size:
confounded customers were always more likely to visit, email or not, and the naive
comparison credits all of that to the email. Both OLS and DML, given the same two
confounders as controls, land back close to the benchmark. DoWhy's identification step on
the equivalent confounded graph (`src/confounded.py:identify_confounded`) confirms this is
not a coincidence: adding `recency → treatment` and `history → treatment` edges to the
graph in `src/dag.py` changes the backdoor adjustment set from empty to exactly
`{recency, history}`, which is precisely what both estimators condition on here.

## Variant 2: selection on an unobservable (`newbie`, withheld)

Retained 31,980 customers. This time treated customers were kept preferentially
when `newbie=0` (an established customer) and control customers preferentially when
`newbie=1` (a new account) — `newbie` has a real, sizeable effect on `visit` on its own
(`reports/03_regression_baseline.md`: −6.45pp). Every estimator in this variant is fit
**without `newbie` in its covariate matrix**, standing in for a confounder nobody
measured.

| estimator | ATE on visit | vs benchmark |
|---|---|---|
| naive diff-in-means | +9.56pp | off by +3.55pp |
| OLS, `newbie` withheld | +10.09pp | off by +4.08pp |
| LinearDML, `newbie` withheld | +9.49pp [+8.72pp, +10.26pp] | benchmark outside the CI |

Neither OLS nor DML recovers the benchmark here, and DML's confidence interval does not
contain it. This is the expected result, not a bug: no amount of model flexibility can
adjust for a variable it never receives. The reason this is worth showing rather than just
stating is that it forecloses the easy version of the question an interviewer would ask,
"doesn't DML fix confounding?": it fixes confounding on the variables it's given, and no
method here, or anywhere, fixes what it was never told about.
