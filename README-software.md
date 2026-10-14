# Software

Extract coling2027-software.zip and coling2027-data.zip into the same directory,
preserving their directory structure. The two attachments contain the code,
parameters, protocols and measurements supporting the paper.

## Recompute the statistics

Python 3.8 or later and NumPy are sufficient. No GPU or model download is needed.

```sh
cd coling2027-paper/reproducibility
python -m pip install -r requirements.txt
python recompute.py --check
```

The script recomputes nine evaluation series from 29,808 measurements and checks
16,888 frozen numerical results. It writes results/means.csv and
results/summary.json. The README in that directory describes each series,
bootstrap settings and optional selection of individual series.

## Method implementation

The output/ directory contains evaluation and analysis sources and the trained
4-by-2560 FP32 answer reader. The data attachment contains the fitted gate and
trigger coefficients, protocols, split manifests and aggregate results at the
paths used by those sources. Appendix A specifies the file layout.

Full model inference additionally requires the specified Qwen model, official
benchmark releases and feature caches. Historical scripts preserve the original
execution bindings and require local dependency and path configuration. The
CPU statistics command is self-contained after both archives are extracted;
it does not regenerate model predictions or verify raw token-prefix relations.
The serial/concurrent execution comparison and saved-state position diagnostics
are supplied as aggregates and inspection sources; their trace/cache inputs are
not included in the statistical entry point.

External models and datasets retain their original licenses and access terms.
No benchmark question text, answer keys or backbone weights are redistributed.
Each MANIFEST file records the included files and their SHA-256 digests.
