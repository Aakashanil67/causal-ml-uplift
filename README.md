# causal-ml-uplift

Who does a marketing email actually persuade, on a real randomised experiment (Hillstrom, 64,000
customers, three arms), estimated with Double Machine Learning and causal forests, validated
against a constructed confounding benchmark, and shipped as a Streamlit what-if simulator.

Method ladder: naive difference-in-means → OLS/logit → DoWhy identification → LinearDML ATE →
CausalForestDML CATE → Qini-based targeting policy → DoWhy refutations.

Status: in progress.
