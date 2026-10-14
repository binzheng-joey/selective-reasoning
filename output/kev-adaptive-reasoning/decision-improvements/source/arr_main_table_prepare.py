"""CPU-only preparation for the proposed three-task main table.

Audits metadata/bindings and estimates execution counts. Does not load models,
read confirmation labels, select a winning method, or launch/modify GPU jobs.
ARC input is a local official JSONL or Hugging Face Parquet split. Only metadata
is exported: questions/options and answerKey are never written to the report.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
OUT = ROOT / 'arr-sprint/main-table'
SEED = 20261010


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def normalized(text):
    return ' '.join(text.split()).casefold()


def arc_metadata(row):
    """Accept official nested JSONL or flattened HF schema without reading gold."""
    question = row['question']
    if isinstance(question, dict):
        text, choices = question['stem'], question['choices']
    else:
        text, choices = question, row['choices']
    if isinstance(choices, dict):
        labels, options = choices['label'], choices['text']
    else:
        labels = [c['label'] for c in choices]
        options = [c['text'] for c in choices]
    labels = [str(x) for x in labels]
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty or invalid ARC question')
    if len(labels) != len(options) or len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('Invalid ARC candidate labels')
    if any(not isinstance(o, str) or not o.strip() for o in options):
        raise ValueError('Empty or invalid ARC candidate')
    sid = str(row['id'])
    if not sid:
        raise ValueError('Missing ARC ID')
    return dict(sample_id='arc:' + sid, dataset='arc_challenge',
                candidate_count=len(labels), source_labels=labels,
                question_sha256=digest(normalized(text)),
                options_sha256=digest(json.dumps([normalized(o) for o in options], ensure_ascii=False)),
                selection_rank=digest(str(SEED) + ':arc:' + sid))


def arc_rows(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        # Deliberately exclude answerKey from the projected Parquet columns.
        yield from pq.read_table(path, columns=['id', 'question', 'choices']).to_pylist()
    else:
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def audit_arc(path, split):
    records = [arc_metadata(row) for row in arc_rows(path)]
    if len({r['sample_id'] for r in records}) != len(records):
        raise ValueError('Duplicate ARC IDs; source needs review')
    # Collapse exact normalized question groups deterministically before sampling.
    groups = {}
    for r in sorted(records, key=lambda x: x['selection_rank']):
        groups.setdefault(r['question_sha256'], r)
    unique = list(groups.values())
    chosen = unique[:256] if split == 'test' else unique
    prior_groups = set()
    overlaps = {}
    for other in ('train', 'validation', 'test'):
        p = OUT / 'arc' / (other + '-metadata.json')
        if other != split and p.exists():
            old = read(p)
            keys = {r['question_sha256'] for r in old['records']}
            overlaps[other] = len(keys.intersection(groups))
            prior_groups.update(keys)
    unsupported = [r['sample_id'] for r in chosen if r['candidate_count'] not in (3, 4)]
    # Keep selected IDs visible even when blocked. Never resample to hide incompatibility.
    return dict(schema=1, source_sha256=sha(path), source_filename=path.name,
                split=split, seed=SEED, input_rows=len(records), unique_question_groups=len(unique),
                candidate_histogram=dict(Counter(r['candidate_count'] for r in records)),
                cross_split_overlaps=overlaps, supported_by_current_head=not unsupported,
                unsupported_selected_ids=unsupported,
                target_confirmation_count=256 if split == 'test' else None,
                target_count_met=split != 'test' or len(chosen) == 256,
                labels_read_for_decisions=False, question_text_exported=False,
                ready_for_inference=False,
                note='Metadata only; requires complete split audit, frozen prompt/trigger and worker support.',
                records=unique, selected_ids=[r['sample_id'] for r in chosen])


def readiness():
    draft = ROOT / 'arr-sprint/integrated/protocol-draft.json'
    p = read(draft)
    mismatches = []
    for path, expected in p['bindings'].items():
        q = Path(path)
        if not q.exists() or sha(q) != expected:
            mismatches.append(q.name)
    manifest = ROOT / 'arr-sprint/data-audit/confirmation-manifest.jsonl'
    records = [json.loads(s) for s in manifest.read_text().splitlines() if s.strip()]
    counts = dict(Counter(r['dataset'] for r in records))
    duplicates = len(records) - len({r['component_id'] for r in records})
    meta_ok = counts == {'proofwriter': 256, 'logiqa2': 256} and duplicates == 0
    recovery = ROOT / 'sentence-runtime/recovery'
    accepted = len(list(recovery.glob('gpu*/[ce]*/completion.json')))
    times = {}
    for file in recovery.glob('gpu*/compiled-off/rows/*.json'):
        r = read(file)
        if (file.parent.parent / 'completion.json').exists():
            times.setdefault(r['policy'], []).append(r['pipeline_seconds'])
    arc_audit = OUT / 'arc/split-audit.json'
    arc_test = OUT / 'arc/test-metadata.json'
    arc_state = read(arc_test) if arc_test.exists() else None
    task_agnostic = OUT / 'task-agnostic-trigger/completion.json'
    native_adapter = ROOT / 'source/arr_native_final.py'
    known_blockers = [
        ('ARC metadata audit exists; no-gold prompt projection and five-option head support remain pending'
         if arc_audit.exists() else 'ARC data/split audit and no-gold prompt projection are incomplete'),
        ('Task-agnostic trigger fitted separately; online/ARC validation and selection remain pending'
         if task_agnostic.exists() else 'Trigger contains a binary LogiQA feature; third-task behavior is not frozen'),
        ('Natural generated-final adapter CPU-tested; live GPU validation remains pending'
         if native_adapter.exists() else 'Natural generated-final baseline must be separated from forced-terminal'),
        'Attention-probe closest-prior training/online adaptation is incomplete',
        'Final method, curve points and three-task confirmation worker are not frozen',
    ]
    if not (recovery / 'completion.json').exists():
        known_blockers.insert(0, 'Runtime comparison and configuration selection are pending')
    return dict(UTC=datetime.now(timezone.utc).isoformat(),
                scope='CPU preparation snapshot; not an experiment or GPU launch',
                preparation_checks_passed=not mismatches and meta_ok,
                binding_mismatches=mismatches, confirmation_metadata_counts=counts,
                duplicate_material_groups=duplicates, confirmation_manifest_sha256=sha(manifest),
                arc_metadata_audited=arc_audit.exists(),
                arc_confirmation_reserved=len(arc_state['selected_ids']) if arc_state else 0,
                arc_unsupported_selected_ids=arc_state['unsupported_selected_ids'] if arc_state else None,
                separate_task_agnostic_candidate_fitted=task_agnostic.exists(),
                development_protocol_frozen=(ROOT / 'arr-sprint/integrated/protocol.json').exists(),
                development_queue_status=(read(ROOT / 'arr-sprint/integrated/queue/status.json')
                                          if (ROOT / 'arr-sprint/integrated/queue/status.json').exists() else None),
                accepted_runtime_phases=accepted, runtime_phases_total=8,
                compiled_off_engineering_seconds={k: statistics.mean(v) for k, v in times.items()},
                timing_scope='Eight observed questions when both shards are complete; external baselines unmeasured',
                integrated_pilot_cases=len(p['sample_ids']) * len(p['core_policies']) * p['repeats'],
                prior_compatibility_cases=8 * len(p['prior_policies']),
                independent_main_table_ready=False, auto_launch=False, blockers=known_blockers)


def estimate(configurations, mean_seconds, questions=768, repeat_questions=48,
             extra_repeats=2, overhead_hours=2):
    if configurations < 1 or questions < 1 or not math.isfinite(mean_seconds) or mean_seconds <= 0:
        raise ValueError('Positive count/time required')
    main = configurations * questions
    repeats = configurations * repeat_questions * extra_repeats
    return dict(assumed_mean_seconds=mean_seconds, measured_prediction=False,
                configurations=configurations, main_runs=main, additional_repeat_runs=repeats,
                main_hours=main * mean_seconds / 3600,
                repeat_hours=repeats * mean_seconds / 3600,
                overhead_hours=overhead_hours,
                total_hours=(main + repeats) * mean_seconds / 3600 + overhead_hours,
                excludes='Implementation, training, calibration, GPU waiting, longer-budget anchors')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Save readiness/budget metadata only')
    parser.add_argument('--configurations', type=int, default=10)
    parser.add_argument('--seconds', type=float, nargs='+', default=[8., 12.])
    parser.add_argument('--arc-source', type=Path)
    parser.add_argument('--arc-split', choices=['train', 'validation', 'test'])
    args = parser.parse_args()
    if bool(args.arc_source) != bool(args.arc_split):
        parser.error('--arc-source and --arc-split must be supplied together')
    report = readiness()
    report['budget_scenarios'] = [estimate(args.configurations, s) for s in args.seconds]
    if args.arc_source:
        audit = audit_arc(args.arc_source, args.arc_split)
        if args.write:
            dest = OUT / 'arc' / (args.arc_split + '-metadata.json')
            if dest.exists() and read(dest)['source_sha256'] != audit['source_sha256']:
                raise ValueError('Existing ARC source differs; preserve selection and review protocol')
            save(dest, audit)
        report['arc_audit'] = {k: v for k, v in audit.items() if k not in ('records', 'selected_ids')}
    if args.write:
        save(OUT / 'readiness.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    if not report['preparation_checks_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
