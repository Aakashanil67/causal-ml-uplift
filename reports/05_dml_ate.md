# Average treatment effects via Double Machine Learning

`LinearDML` (EconML) with `LightGBM` nuisance models for both the outcome and treatment
regressions, 3-fold cross-fitting. See `src/dml_ate.py`'s docstring for what partialling
out and cross-fitting actually do; the short version is that on a randomised experiment
there's nothing for DML to correct for, so the value of running it here is establishing
that the machinery agrees with simpler methods before `src/cate.py` reuses the same
machinery on a question OLS cannot answer at all: how the effect varies by customer.

## Pooled treatment (any email vs none), vs the regression baseline

`visit`/`conversion` in percentage points, `spend` in dollars, matching
`reports/03_regression_baseline.md`'s units so the two reports read side by side.

| outcome | DML ATE | DML 95% CI | regression AME/coef |
|---|---|---|---|
| visit | +6.01pp | [5.48pp, 6.55pp] | +6.49pp |
| conversion | +0.50pp | [0.36pp, 0.64pp] | +0.56pp |
| spend | +$0.6131 | [0.3891, 0.8371] | +$0.5958 |

DML and the covariate-adjusted regression from `reports/03_regression_baseline.md` land
within each other's confidence intervals on every outcome — the expected result on an RCT,
and a real check rather than a formality: if a flexible nuisance model had found nonlinear
structure in the covariates that a linear control missed, these numbers could have
diverged.

## By arm: mens email and womens email are not the same treatment

| arm | outcome | DML ATE vs control | 95% CI |
|---|---|---|---|
| Mens E-Mail | visit | +7.47pp | [6.82pp, 8.12pp] |
| Mens E-Mail | conversion | +0.67pp | [0.49pp, 0.85pp] |
| Mens E-Mail | spend | +$0.7829 | [0.4872, 1.0787] |
| Womens E-Mail | visit | +4.49pp | [3.87pp, 5.12pp] |
| Womens E-Mail | conversion | +0.33pp | [0.17pp, 0.50pp] |
| Womens E-Mail | spend | +$0.4468 | [0.1885, 0.7051] |

On `visit`, the mens email lifts the visit rate by 7.47 percentage
points and the womens email by 4.49, a real gap between two
creatives that the pooled any-email number above averages away. `src/policy.py` treats
this as what it is: a three-action decision (no email, mens email, womens email) rather
than a binary send/don't-send call, and asks whether targeting each customer with the
better-matched creative beats the obvious `mens`/`womens` purchase-history heuristic.
