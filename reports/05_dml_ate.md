# Average treatment effects via Double Machine Learning

LinearDML uses LightGBM nuisance models and three-fold cross-fitting. Cross-fitting keeps each row's nuisance predictions out of that row's nuisance-model fit. On a randomized experiment, adjustment is not needed for identification, but flexible nuisance models can still overfit. LinearDML uses a linear final effect stage; flexible nuisance models alone do not guarantee recovery under arbitrary treatment-effect heterogeneity.

## Pooled treatment: either email versus no email

| outcome | DML effect | 95% CI | regression discrete change / coefficient |
|---|---:|---:|---:|
| visit | +6.01pp | [+5.48pp, +6.55pp] | +6.07pp |
| conversion | +0.50pp | [+0.36pp, +0.64pp] | +0.49pp |
| spend | $+0.6131 | [$0.3891, $0.8371] | $+0.5958 |

## By creative versus no email

| arm | outcome | DML effect | 95% CI |
|---|---|---:|---:|
| Mens E-Mail | visit | +7.47pp | [+6.82pp, +8.12pp] |
| Mens E-Mail | conversion | +0.67pp | [+0.49pp, +0.85pp] |
| Mens E-Mail | spend | $+0.7829 | [$0.4872, $1.0787] |
| Womens E-Mail | visit | +4.49pp | [+3.87pp, +5.12pp] |
| Womens E-Mail | conversion | +0.33pp | [+0.17pp, +0.50pp] |
| Womens E-Mail | spend | $+0.4468 | [$0.1885, $0.7051] |

The pooled estimate averages the historical mixture of the two creatives. The separate arm estimates show why a decision analysis should preserve the three available actions. Differences in arm estimates are descriptive and do not by themselves establish customer-level personalization value.
