# Constructed selection diagnostics

The full randomized sample's DML estimate for `visit` is +6.01pp [+5.48pp, +6.55pp]. This is an estimated experimental reference, not known truth for either selected sample. Selection changes the covariate distribution and can change the target-population effect. Differences below are descriptive selection diagnostics, not numerical bias estimates.

Bias and coverage against known targets are evaluated in the fully synthetic Monte Carlo benchmark. LinearDML has flexible nuisance models but a linear final effect stage; neither property guarantees recovery under arbitrary effect heterogeneity.

## Selection on observed `recency` and `history`

Retained 31,592 rows. Both variables remain available to the estimators.

| estimator | selected-sample estimate | difference from full-RCT estimate |
|---|---:|---:|
| Naive difference in means | +10.03pp | +4.02pp |
| OLS with selected covariates | +5.93pp | -0.09pp |
| LinearDML | +5.99pp | -0.02pp |

These contrasts describe estimator movement under this constructed selection mechanism. They do not determine bias for the selected-population causal target.

## Selection on `newbie`, withheld from estimators

Retained 31,980 rows. The fitted estimators omit `newbie` by construction.

| estimator | selected-sample estimate | difference from full-RCT estimate |
|---|---:|---:|
| Naive difference in means | +9.56pp | +3.55pp |
| OLS with selected covariates | +10.09pp | +4.08pp |
| LinearDML | +9.49pp | +3.48pp |

These contrasts describe estimator movement under this constructed selection mechanism. They do not determine bias for the selected-population causal target.

In the visit logit, the `newbie` profile contrast is -6.70pp. This is an adjusted customer-profile contrast, not an effect of treatment.
