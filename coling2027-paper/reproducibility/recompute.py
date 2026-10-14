#!/usr/bin/env python3
"""Recompute measurement summaries and paired bootstrap intervals on a CPU."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METRICS = ('accuracy', 'pipeline_seconds', 'physical_output_tokens', 'generation_tokens',
           'full_reads', 'trigger_calls', 'input_tokens', 'cached_input_tokens',
           'reasoning_rate', 'early_stop_rate', 'invalid_rate', 'budget_hit_rate')
RATIO_METRICS = ('pipeline_seconds', 'physical_output_tokens', 'generation_tokens',
                 'full_reads', 'trigger_calls', 'input_tokens')
TASKS = ('proofwriter', 'logiqa2', 'arc_challenge')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_indices(metadata, draws, seed):
    """Exact original draw order: paired strata in metadata insertion order."""
    groups = collections.defaultdict(list)
    for i, row in enumerate(metadata):
        groups[(row['dataset'], row['stratum'])].append(i)
    require(len({r['component_id'] for r in metadata}) == len(metadata),
            'This released cohort requires exactly one question per material group')
    rng = np.random.default_rng(seed)
    return np.concatenate([np.asarray(indices)[rng.integers(0, len(indices), size=(draws, len(indices)))]
                           for indices in groups.values()], axis=1)


def bootstrap_means(values, indices, batch_size=250):
    # Batching bounds peak memory; the draw sequence and arithmetic match the
    # original indexing/mean estimator, without GPU/model dependencies.
    out = np.empty((len(indices), values.shape[1]), dtype=float)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        for col in range(values.shape[1]):
            out[start:stop, col] = values[indices[start:stop], col].mean(axis=1)
    return out


def compare_subset(actual, expected, path='', tolerance=1e-9):
    """Expected numbers are comparison targets only, never computation inputs."""
    count = 0
    if isinstance(expected, dict):
        require(isinstance(actual, dict), 'Type mismatch: ' + path)
        for key, value in expected.items():
            require(key in actual, 'Missing result: ' + path + '/' + key)
            count += compare_subset(actual[key], value, path + '/' + key, tolerance)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), 'List mismatch: ' + path)
        for i, value in enumerate(expected):
            count += compare_subset(actual[i], value, path + '/' + str(i), tolerance)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and
                math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
                'Numeric mismatch %s: recomputed=%r expected=%r' % (path, actual, expected))
        count += 1
    else:
        require(actual == expected, 'Value mismatch: ' + path)
    return count


def load(data):
    manifest = read(data / 'manifest.json')
    require(manifest['schema_version'] == 1, 'Unsupported schema')
    for name, digest in manifest['files'].items():
        require(sha(data / name) == digest, 'Changed release input: ' + name)
    metadata = read(data / 'metadata.json')
    rows = collections.defaultdict(list)
    fields = ['series', 'question_id', 'method', 'repeat'] + list(METRICS)
    with (data / 'records.csv').open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames == fields, 'Unexpected CSV schema')
        for row in reader:
            row['repeat'] = int(row['repeat'])
            for metric in METRICS:
                row[metric] = float(row[metric])
                require(math.isfinite(row[metric]) and row[metric] >= 0, 'Invalid exported metric')
            require(row['accuracy'] in (0., 1.) and row['cached_input_tokens'] <= row['input_tokens'],
                    'Invalid correctness/cache accounting')
            rows[row['series']].append(row)
    expected = read(data / 'expected.json')
    require(set(rows) == set(metadata) == set(expected) == {s['id'] for s in manifest['series']}, 'Series inventory mismatch')
    return manifest, metadata, rows, expected


def analyze(config, metadata, rows, draws, seed):
    series = config['id']
    require(len(metadata) == config['questions'], 'Unexpected question count: ' + series)
    require([m['order'] for m in metadata] == list(range(len(metadata))), 'Changed metadata order')
    ids = [m['question_id'] for m in metadata]
    require(len(set(ids)) == len(ids), 'Duplicate question metadata')
    key = {(r['question_id'], r['method'], r['repeat']): r for r in rows}
    target = {(sid, method, repeat) for sid in ids for method in config['methods'] for repeat in range(config['repetitions'])}
    require(len(rows) == len(key) and set(key) == target, 'Missing, duplicate or extra measurement: ' + series)
    arrays = {}
    for method in config['methods']:
        values = []
        for sid in ids:
            initial = key[sid, method, 0]
            row = [initial[metric] for metric in METRICS]
            if config['repetitions'] > 1:
                for metric in config['repeat_average']:
                    row[METRICS.index(metric)] = float(np.mean([key[sid, method, rep][metric] for rep in range(config['repetitions'])]))
            values.append(row)
        arrays[method] = np.asarray(values, dtype=float)
    summaries, pairs = [], []
    for task in ('all',) + TASKS:
        chosen = [i for i, m in enumerate(metadata) if task == 'all' or m['dataset'] == task]
        if not chosen:
            continue
        selected = [metadata[i] for i in chosen]
        if task == 'all':
            require(len(set(collections.Counter(m['dataset'] for m in selected).values())) == 1,
                    'Task-equal aggregation needs balanced task counts')
        ix = bootstrap_indices(selected, draws, seed)
        local = {method: values[chosen] for method, values in arrays.items()}
        samples = {method: bootstrap_means(values, ix) for method, values in local.items()}
        for method in config['methods']:
            values, sampled = local[method], samples[method]
            row = dict(dataset=task, method=method, n=len(chosen))
            if not config['repeated_timing_only']:
                row['correct'] = int(values[:, 0].sum())
            row.update({metric: float(values[:, METRICS.index(metric)].mean()) for metric in config['summary_metrics']})
            row['ci95'] = {metric: np.quantile(sampled[:, METRICS.index(metric)], [.025, .975]).tolist()
                           for metric in config['summary_metrics']}
            row['p95_pipeline_seconds'] = float(np.percentile(values[:, 1], 95))
            summaries.append(row)
        for pair in config['paired']:
            if pair['dataset'] != task:
                continue
            method, reference = pair['method'], pair['reference']
            a, b = local[method], local[reference]
            av, bv = samples[method], samples[reference]
            stats = {}
            for j, metric in enumerate(METRICS):
                if metric not in pair['available_metrics']:
                    stats[metric] = dict(difference=None, ci95=None, ratio=None, ratio_ci95=None,
                                         saving_fraction=None, saving_fraction_ci95=None)
                    continue
                value = dict(difference=float((a[:, j] - b[:, j]).mean()),
                             ci95=np.quantile(av[:, j] - bv[:, j], [.025, .975]).tolist())
                if metric in RATIO_METRICS:
                    ratio = float(a[:, j].mean() / b[:, j].mean()) if b[:, j].mean() > 0 else None
                    bounds = np.quantile(av[:, j] / bv[:, j], [.025, .975]).tolist() if b[:, j].mean() > 0 and np.all(bv[:, j] > 0) else None
                    value.update(ratio=ratio, ratio_ci95=bounds,
                                 saving_fraction=None if ratio is None else 1 - ratio,
                                 saving_fraction_ci95=None if bounds is None else [1 - bounds[1], 1 - bounds[0]])
                stats[metric] = value
            pairs.append(dict(dataset=task, method=method, reference=reference, n=len(chosen), metrics=stats))
    return dict(summaries=summaries, paired=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--series', nargs='+', help='Optional series subset; default: all released series')
    parser.add_argument('--check', action='store_true', help='Verify all computed values against frozen expected.json')
    args = parser.parse_args()
    manifest, metadata, rows, expected = load(args.data)
    configs = [s for s in manifest['series'] if not args.series or s['id'] in args.series]
    require(not args.series or set(args.series) <= {s['id'] for s in configs}, 'Unknown requested series')
    results, checks = {}, {}
    for config in configs:
        name = config['id']
        print('Recomputing %s: %d questions, %d methods, %d executions/question/method' %
              (name, config['questions'], len(config['methods']), config['repetitions']), flush=True)
        result = analyze(config, metadata[name], rows[name], manifest['bootstrap_draws'], manifest['seed'])
        if args.check:
            # Series summaries can contain independently computed extras; match
            # only frozen targets, indexed by their explicit method/task keys.
            actual_s = {(r['dataset'], r['method']): r for r in result['summaries']}
            actual_p = {(r['dataset'], r['method'], r['reference']): r for r in result['paired']}
            count = 0
            for ref in expected[name]['summaries']:
                count += compare_subset(actual_s[ref['dataset'], ref['method']], ref, name + '/summary')
            for ref in expected[name]['paired']:
                count += compare_subset(actual_p[ref['dataset'], ref['method'], ref['reference']], ref, name + '/paired')
            checks[name] = dict(state='passed', numeric_values_checked=count)
            print('  Verified %d numeric values' % count, flush=True)
        results[name] = result
    args.output.mkdir(parents=True, exist_ok=True)
    payload = dict(seed=manifest['seed'], bootstrap_draws=manifest['bootstrap_draws'],
                   interval='Pointwise paired material-group bootstrap stratified by dataset/PW subtype',
                   results=results, checks=checks)
    (args.output / 'summary.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    with (args.output / 'means.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['series', 'dataset', 'method', 'n', 'correct'] + list(METRICS)
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, result in results.items():
            writer.writerows(dict(series=name, **row) for row in result['summaries'])
    print('CPU replay complete; results written to ' + str(args.output), flush=True)


if __name__ == '__main__':
    main()
