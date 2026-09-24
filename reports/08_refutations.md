# Refutation checks: what these results do and do not establish

Placebo treatment, random common cause, and data-subset checks challenge particular fitted estimates. Their outcomes do not certify identification or general estimator behavior.

## Randomized Hillstrom sample

Reference estimate: +0.0601.

| check | result |
|---|---:|
| placebo treatment | +0.0003 [-0.0054, +0.0061] |
| random common cause | +0.0605 [+0.0551, +0.0658] |
| data subset (5 refits) | mean +0.0610, SD 0.0015 |

These diagnostics show how this fitted estimate responds to the stated perturbations. For the selected sample, the full-RCT estimate (+0.0601) is a reference for a different population; the difference is descriptive, not a known-bias test.

## Constructed selected sample

Reference estimate: +0.0949.

| check | result |
|---|---:|
| placebo treatment | +0.0022 [-0.0061, +0.0105] |
| random common cause | +0.0952 [+0.0875, +0.1030] |
| data subset (5 refits) | mean +0.0950, SD 0.0015 |

These diagnostics show how this fitted estimate responds to the stated perturbations. For the selected sample, the full-RCT estimate (+0.0601) is a reference for a different population; the difference is descriptive, not a known-bias test.

The fully synthetic simulation is where the generating probabilities, sample-average target, bias, and coverage are known. Refutation checks are diagnostics, not correctness certificates.
