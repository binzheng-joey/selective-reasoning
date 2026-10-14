# Selective Reasoning for Typed Decision Models
## When to Start, Read, and Stop

Bin Zheng and Zheng Chen contributed equally.
University of Electronic Science and Technology of China.
Contact: zchen@uestc.edu.cn

Research code and evaluation measurements for a shared candidate-answer reader
with policies for starting reasoning, acquiring answers, and stopping generation.

## Downloads

The Releases section provides:
- `coling2027-software.zip`: implementation sources, reader weights and statistical replication code.
- `coling2027-data.zip`: evaluation measurements, parameters, protocols and aggregate results.

## Statistical replication

Download both archives and extract them into the same directory, preserving
the directory structure. Python 3.8 or later and NumPy are required.

```bash
cd coling2027-paper/reproducibility
python -m pip install -r requirements.txt
python recompute.py --check
```

The CPU script recomputes nine evaluation series from 29,808 measurements
and checks 16,888 reference values.

Model inference requires external model weights, benchmark releases and feature
caches, as described in the archive documentation.
