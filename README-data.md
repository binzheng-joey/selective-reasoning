# Evaluation data and parameters

Extract this archive and coling2027-software.zip into the same directory,
preserving their directory structure. No existing files are overwritten by the
other archive. Each archive uses a distinct top-level README and MANIFEST.

The coling2027-paper/reproducibility/data directory contains the per-question
measurements, anonymous material-group metadata, statistical settings and frozen
comparison targets for CPU recomputation. The software attachment provides the
script and installation instructions. Measurements and expected results retain
their original bytes.

The output/ directory contains gate and trigger parameters, train/development/
evaluation manifests, collection protocols and aggregate results. Execution
series are kept separate; EXECUTION-IDENTITY.json lists their roles. The repeated
timing series average three executions within each of 48 questions.

Records contain correctness, timing and work measurements. They omit benchmark
text, answer categories, generated reasoning and token traces. Statistical
replication uses only the included measurements; reconstructing model inference
requires the external resources specified in Appendix A. Source-binding digests
refer to the original experiment files, while this archive's MANIFEST records
the digests of the supplied files and release transformations.
