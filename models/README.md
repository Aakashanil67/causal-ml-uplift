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

- `causal_forest.joblib`: `d7cd8f2c72eafcc0daaf67ca2a564860e91b15a29ef1dc09d0aaf6e92cae0e28`
- `evaluation_artifacts.joblib`: `fc33eb5680ef9200ee629ed2bafba6328f4b045c3adce20e3efe9df361cf7e8c`
