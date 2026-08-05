# Regression baseline: logit and OLS with robust standard errors

Logit for `visit` and `conversion` (binary), OLS for `spend`, all with heteroscedasticity-
robust (HC1) standard errors and the full covariate set as controls. Reported as average
marginal effects for the logit models rather than raw log-odds coefficients. That is how an
economist would actually read these out: a coefficient of 0.5366 on treatment in the
`visit` logit says nothing directly interpretable; a marginal effect of +6.49 percentage
points does.

## Treatment effect, three ways to read it

| outcome | regression AME / coefficient | 95% CI | naive diff-in-means |
|---|---|---|---|
| visit | +6.49pp | [5.87pp, 7.10pp] | +6.09pp |
| conversion | +0.56pp | [0.38pp, 0.75pp] | +0.50pp |
| spend | +$0.5958 | [0.3753, 0.8163] | +$0.5968 |

The regression estimate and the naive diff-in-means agree closely on all three outcomes,
which is exactly what should happen on a randomised experiment: adding covariate controls
should barely move the estimate, because treatment is uncorrelated with those covariates by
design (`reports/02_naive_estimate.md`). Controls earn their place here for precision (a
tighter CI on `spend`, where outcome variance is high), not for removing confounding that
was never there.

## Full marginal-effects table, `visit`

| covariate | dy/dx | 95% CI |
|---|---|---|
| recency | -0.64pp | [-0.72pp, -0.56pp] |
| history | +0.01pp | [0.00pp, 0.01pp] |
| mens | +6.91pp | [6.04pp, 7.78pp] |
| womens | +8.78pp | [7.90pp, 9.66pp] |
| newbie | -6.73pp | [-7.30pp, -6.16pp] |
| zip_code_Surburban | -4.32pp | [-5.08pp, -3.56pp] |
| zip_code_Urban | -4.50pp | [-5.27pp, -3.73pp] |
| channel_Phone | -1.21pp | [-2.11pp, -0.30pp] |
| channel_Web | +2.01pp | [1.12pp, 2.89pp] |
| treatment | +6.49pp | [5.87pp, 7.10pp] |

Reading a control coefficient the way an economist would: `newbie` has a marginal effect
of -6.73pp on the probability of visiting. A customer who opened their
account in the last twelve months is about 6.7 percentage points less
likely to visit after the campaign than an otherwise-identical existing customer, other
covariates held fixed. That is a real difference between customer types, not a treatment
effect: `newbie` predicts engagement, it does not predict which arm anyone was assigned to
(see the covariate balance table in `reports/02_naive_estimate.md`).
