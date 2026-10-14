"""Isolated side experiment: existing development data, frozen native reasoner."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
PR = PROJECT / 'output/kev-adaptive-reasoning/prefix-readout'
SPARSE = PROJECT / 'output/kev-adaptive-reasoning/sentence-async-readers-v1/sparse-checks'
OLD = PROJECT / 'output/kev-adaptive-reasoning/gate-utility-expanded1024-riskcost-20261005-v1'
MODEL = PROJECT / 'experiments/kev-adaptive-reasoning/models/Qwen3.5-4B'
GPUS = {0: 'device-C',
        1: 'device-B'}
GRID = [0, 128, 256, 512, 1024]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(PROJECT))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    with tmp.open('w') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)
