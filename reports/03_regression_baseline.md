# Regression baseline: logit and OLS with robust standard errors

Logit models for `visit` and `conversion` use HC1-robust uncertainty and average counterfactual prediction differences for binary regressors, valid profile contrasts for categorical covariates, and average derivatives for continuous covariates. The treatment contrast sets assignment to 0 and 1 for every observed profile. OLS estimates the spend coefficient with HC1 uncertainty.

## Treatment effects

| outcome | adjusted contrast | 95% CI | naive difference |
|---|---:|---:|---:|
| visit | +6.07pp | [+5.54pp, +6.61pp] | +6.09pp |
| conversion | +0.49pp | [+0.35pp, +0.64pp] | +0.50pp |
| spend | $+0.5958 | [$0.3753, $0.8163] | $+0.5968 |

## Average logit contrasts: `visit`

Numeric covariates show average derivatives. Binary covariates use 0-to-1 counterfactual changes; categorical contrasts compare valid profiles to the reference level. These are associations/adjustment contrasts for covariates, not treatment effects.

| contrast | calculation | estimate | 95% CI |
|---|---|---:|---:|
| recency | average derivative | -0.64pp | [-0.72pp, -0.56pp] |
| history | average derivative | +0.01pp | [+0.00pp, +0.01pp] |
| treatment | average 0-to-1 change | +6.07pp | [+5.54pp, +6.61pp] |
| mens | average 0-to-1 change | +6.86pp | [+6.00pp, +7.73pp] |
| womens | average 0-to-1 change | +8.65pp | [+7.78pp, +9.52pp] |
| newbie | average 0-to-1 change | -6.70pp | [-7.27pp, -6.14pp] |
| zip_code=Surburban vs Rural | category profile contrast | -4.70pp | [-5.57pp, -3.83pp] |
| zip_code=Urban vs Rural | category profile contrast | -4.87pp | [-5.75pp, -4.00pp] |
| channel=Phone vs Multichannel | category profile contrast | -1.14pp | [-2.01pp, -0.27pp] |
| channel=Web vs Multichannel | category profile contrast | +2.08pp | [+1.19pp, +2.96pp] |

## Average logit contrasts: `conversion`

Numeric covariates show average derivatives. Binary covariates use 0-to-1 counterfactual changes; categorical contrasts compare valid profiles to the reference level. These are associations/adjustment contrasts for covariates, not treatment effects.

| contrast | calculation | estimate | 95% CI |
|---|---|---:|---:|
| recency | average derivative | -0.05pp | [-0.08pp, -0.03pp] |
| history | average derivative | +0.00pp | [+0.00pp, +0.00pp] |
| treatment | average 0-to-1 change | +0.49pp | [+0.35pp, +0.64pp] |
| mens | average 0-to-1 change | +0.32pp | [+0.10pp, +0.54pp] |
| womens | average 0-to-1 change | +0.43pp | [+0.21pp, +0.65pp] |
| newbie | average 0-to-1 change | -0.39pp | [-0.55pp, -0.23pp] |
| zip_code=Surburban vs Rural | category profile contrast | -0.28pp | [-0.52pp, -0.05pp] |
| zip_code=Urban vs Rural | category profile contrast | -0.22pp | [-0.46pp, +0.02pp] |
| channel=Phone vs Multichannel | category profile contrast | -0.15pp | [-0.39pp, +0.08pp] |
| channel=Web vs Multichannel | category profile contrast | +0.01pp | [-0.23pp, +0.25pp] |

The treatment contrast is an average discrete change in predicted probability. It is not the derivative of the logit probability with respect to treatment.
