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

- `causal_forest.joblib`: `e5ba9562bde901c2181386e3a0093c90dcfb8f23e1fd8801873471942b8bce9a`
- `evaluation_artifacts.joblib`: `6c2bbecaae4aa29d32d2538f5150caae668d53918e9cfc895070f6fd1936ad9c`
