# Model artifacts

`causal_forest.joblib` is a serialized `EconML CausalForestDML` fitted on all 64,000 Hillstrom
rows for interactive profile scoring. It is a serving artifact, not the source of the published
evaluation claim: the heterogeneity and ranking claims are evaluated on the held-out 30% split.

The artifact is accepted only when its embedded data checksum, source fingerprint, feature schema
and dependency versions match the current checkout. Run `scripts/verify.ps1` after installing the
pinned requirements before loading it.

The model estimates a profile-level conditional average effect on the `visit` outcome for email
assignment versus no email. It does not estimate a person-level causal effect, delivery, opening,
reading or clicking effect. Regenerate it with `python -m src.pipeline`; do not hand-edit or rename
the artifact without rebuilding the provenance metadata.

Current artifact checksums:

- `causal_forest.joblib`: `966c39309c4cb79a25b5b122d31da766f4f17c8ddb3eb19805cb730c03da8b48`
- `evaluation_artifacts.joblib`: `28dc116e0a6a5d564d4d2f8632619821753705a30dc8d48a52e4549dc5ff2284`
