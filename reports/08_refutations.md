# Refutation tests: what passing does and does not prove

Placebo treatment, random common cause, and data-subset refutation, run against this
project's own `LinearDML` pipeline (`src/dml_ate.py`) rather than through DoWhy's
`refute_estimate()` wrapper, which fails on this project's categorical covariates before
any refutation logic runs (see `src/refute.py`'s docstring). Run twice: once on the real,
unconfounded RCT, and once on `src/confounded.py`'s already-known-biased estimate, to show
directly what these tests can and cannot catch, rather than asserting it.

## On the real RCT

Original pooled DML ATE on `visit`: **+0.0601** (`reports/05_dml_ate.md`).

| refuter | result | read |
|---|---|---|
| placebo treatment | +0.0003 [-0.0054, 0.0061] | PASS, CI contains 0 |
| random common cause | +0.0605 [0.0551, 0.0658] | PASS, close to the original |
| data subset (5 runs) | mean +0.0610, std 0.0015 | PASS, stable across subsamples |

All three pass, and here is exactly what that does and does not mean. The placebo
test shows the estimator does not manufacture an effect out of nothing when there is
genuinely nothing there. The random-common-cause test shows adding an irrelevant
column does not move the number, which it shouldn't. The subset test shows the
estimate does not depend on which 80% of customers happened to be sampled.
None of these three tests the one assumption this project's identification claim
actually rests on: that treatment assignment had no unobserved cause in common with the
outcome. That assumption is not refuted here; it holds by design, because Hillstrom is
randomised (`reports/01_causal_question.md`, `reports/04_identification.md`), and no
refutation test run on the data after the fact can substitute for that design fact.

## On a known-biased estimate, to show what these tests miss

`src/confounded.py`'s variant 2 (`newbie` withheld from the estimator) already showed a
DML estimate of **+0.0949**, confidence interval excluding the
true benchmark of **+0.0601** (`reports/06_confounding_benchmark.md`).
This is a real, demonstrated bias. Running the same three refuters against it, with the
same `newbie` column withheld so this is testing the actual biased model and not a
different, better-specified one:

| refuter | result | read |
|---|---|---|
| placebo treatment | +0.0022 [-0.0061, 0.0105] | "PASSES", CI contains 0 |
| random common cause | +0.0952 [0.0875, 0.1030] | "PASSES", close to the original |
| data subset (5 runs) | mean +0.0950, std 0.0015 | "PASSES", stable across subsamples |

The quotation marks are deliberate. These refuters test properties an estimator can
hold regardless of whether it is right, so a confounded, biased estimate sails
through all three exactly as cleanly as a correct one does, which is what the table
above shows happening.

The only thing in this project that actually tested the identification assumption
itself was `reports/06_confounding_benchmark.md`'s comparison against a known
experimental benchmark, because that is the one place a ground truth existed to check
against. Most real observational studies do not have that luxury, which is the honest
limitation worth sitting with: these three refutation tests are a standard part of the
DoWhy workflow and worth running, but passing them is not evidence against omitted-
variable bias, and no one should present it as such.
