# CPU statistical replication

This directory provides anonymized per-question measurements from completed,
previously scored evaluation collections and a CPU-only script that recomputes
their means, accuracy counts, paired differences, ratios and pointwise 95%
bootstrap intervals. It requires Python 3.8 or later and NumPy. No model,
GPU, benchmark download or credentials are needed.

```sh
python -m pip install -r requirements.txt
python recompute.py --check
```

Run these commands inside this directory after extracting the attachment. The
script writes `results/means.csv` and `results/summary.json`. To replay only the
two central evaluation series:

```sh
python recompute.py --check --series primary13 matched-six
```

The summary JSON contains full-precision values and confidence intervals. The
CSV contains unrounded means; multiply accuracy/rate columns by 100 for
percentages. `expected.json` holds frozen comparison targets from the existing
analyses. The computation uses `records.csv` and `metadata.json`; expected
numbers are used only for checking. `--check` fails on a missing/duplicate
measurement, changed file hash, invalid accounting, or numeric mismatch.

| Series | What it recomputes |
|---|---|
| `primary13` | Primary 768-question, 13-configuration evaluation underlying the main comparison, including the predefined simple controller |
| `matched-six` | Separately executed 768-question, six-policy component comparison |
| `primary13-timing48` | Three executions averaged within each of 48 fixed questions for the primary configurations |
| `matched-six-timing48` | Three executions averaged within each of 48 fixed questions for the matched controls |
| `bridge48-first` | First executions on the fixed 48-question bridge cohort |
| `calibrated-primary` | Nine development-selected configurations on the already observed 768 questions, including the calibrated simple policies |
| `calibrated-timing48` | Within-question three-run timing for the calibrated configurations |
| `factorial-primary` | Four factorial controllers on the already observed 768 questions |
| `factorial-timing48` | Within-question three-run timing for the factorial controllers |

Each series retains its own executions. The script never combines primary
accuracy with costs from another series. Repeated timing remains a 48-question
analysis: repetitions are averaged within questions and do not increase the
accuracy sample size. Repeated timing outputs contain timing summaries only.
Some original analyses suppress particular paired timing contrasts; the replay
preserves that availability in the manifest rather than creating new contrasts.

The bootstrap uses 10,000 paired draws with seed `20261010`. Resampling is
stratified by dataset and ProofWriter subtype, with the original metadata order.
This released cohort contains one question per material group. The script
validates that property before resampling; its intervals condition on these
executions and the frozen trained components. They are pointwise intervals.

`records.csv` contains anonymous question IDs, method IDs, execution indices and
numeric measurements. `accuracy` is a correctness indicator, not an answer
class. `input_tokens` counts tokens submitted across requests; cached input
tokens are reported separately. `physical_output_tokens` includes generated
tokens and candidate-read output tokens. Reader requests and trigger calls are
separate counters. These measures are not FLOPs, energy or concurrent throughput.

The release excludes original question IDs, question text, gold answer classes,
predicted answer classes, reasoning text, generated token sequences, raw device
identities and personal filesystem paths. It recomputes statistics from the
exported measurements. It does not rerun inference or independently reproduce
the exact-prefix generation-trace audit. Collection/model source files elsewhere
in the attachment are provided for implementation inspection.
