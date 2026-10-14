"""Three-task analysis, prepared without opening sealed predictions or labels.

--development-check exercises the actual schema on already observed data.
--analyze requires completed phases AND a separate selection-closure release.
No code here reads official source labels or creates a release automatically.
"""
import argparse
import collections
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import arr_formal_run as formal

rt = formal.rt
ROOT = formal.OUT
OUT = ROOT / 'analysis'
SOURCE = Path(__file__).resolve()
TASKS = ('proofwriter', 'logiqa2', 'arc_challenge')
METRICS = ('accuracy', 'pipeline_seconds', 'physical_output_tokens', 'generation_tokens',
           'full_reads', 'trigger_calls', 'input_tokens', 'cached_input_tokens',
           'reasoning_rate', 'early_stop_rate', 'invalid_rate', 'budget_hit_rate')


def metric_row(r, label):
    prediction = r['prediction']
    cap = r['policy_spec']['cap']
    inputs = r.get('request_input_tokens', r.get('reader_prompt_tokens', 0) + r.get('generator_prompt_tokens', 0))
    cached = r.get('request_cached_tokens', r.get('reader_cached_tokens', 0) + r.get('generator_cached_tokens', 0))
    result = [float(prediction == label), r['pipeline_seconds'], r['physical_output_tokens'],
              r['generation_tokens'], r['full_reads'], r['trigger_calls'], inputs, cached,
              float(r['reasoning_invoked']), float(r['early_stop']), float(prediction is None),
              float(r['generation_tokens'] >= cap and not r['early_stop'])]
    assert all(math.isfinite(float(v)) and v >= 0 for v in result)
    assert r['pipeline_seconds'] > 0 and cached <= inputs
    return result


def bootstrap_indices(ids, metadata, draws, seed):
    rng = np.random.default_rng(seed)
    groups = collections.defaultdict(list)
    for i, sid in enumerate(ids):
        s = metadata[sid]
        groups[(s['dataset'], s.get('stratum', s['dataset']))].append(i)
    # This confirmation cohort has exactly one question per material group.
    assert len({metadata[s]['component_id'] for s in ids}) == len(ids), 'Cluster multiple questions before resampling'
    return np.concatenate([np.asarray(indices)[rng.integers(0, len(indices), size=(draws, len(indices)))]
                           for indices in groups.values()], axis=1)


def summarize(records, metadata, labels, methods, draws=10000):
    ids = list(metadata)
    key = {(r['sample_id'], r['policy']): r for r in records}
    assert all(r['repeat'] == 0 for r in records), 'Do not pool repeated accuracy observations'
    assert len(key) == len(records), 'Duplicate question/method'
    assert set(key) == {(s, m) for s in ids for m in methods}, 'Incomplete or extra comparison matrix'
    assert set(ids) <= set(labels)
    for sid in ids:
        assert 0 <= labels[sid] < metadata[sid]['candidate_count']
        for m in methods:
            pred = key[sid, m]['prediction']
            assert pred is None or isinstance(pred, int) and 0 <= pred < metadata[sid]['candidate_count']
    summaries, pairs, diagnostics = [], [], []
    for task in ('all',) + TASKS:
        chosen = [s for s in ids if task == 'all' or metadata[s]['dataset'] == task]
        if not chosen:
            continue
        if task == 'all':
            assert len(set(collections.Counter(metadata[s]['dataset'] for s in chosen).values())) == 1, 'Task-equal aggregate needs balanced tasks'
        ix = bootstrap_indices(chosen, metadata, draws, 20261010)
        arrays = {m: np.asarray([metric_row(key[s, m], labels[s]) for s in chosen], dtype=float) for m in methods}
        samples = {m: np.stack([a[ix, col].mean(1) for col in range(len(METRICS))], axis=1) for m, a in arrays.items()}
        for m, a in arrays.items():
            mean = a.mean(0)
            summaries.append(dict(dataset=task, method=m, n=len(chosen), correct=int(a[:, 0].sum()),
                **dict(zip(METRICS, mean.tolist())),
                ci95={k: np.quantile(samples[m][:, j], [.025, .975]).tolist() for j, k in enumerate(METRICS)},
                p95_pipeline_seconds=float(np.percentile(a[:, 1], 95)),
                device_ids=sorted({key[s, m]['GPU_UUID'] for s in chosen})))
        for ref in ('full-1024', 'simple-margin-every8-1024', 'ours-entropy70-1024'):
            if ref not in arrays:
                continue
            for m in methods:
                if m == ref:
                    continue
                delta = samples[m] - samples[ref]
                same = all(key[s, m]['GPU_UUID'] == key[s, ref]['GPU_UUID'] for s in chosen)
                pair = dict(dataset=task, method=m, reference=ref, n=len(chosen), same_device=same,
                    accuracy_difference=float(arrays[m][:, 0].mean() - arrays[ref][:, 0].mean()),
                    accuracy_difference_ci95=np.quantile(delta[:, 0], [.025, .975]).tolist(),
                    physical_output_difference=float(arrays[m][:, 2].mean() - arrays[ref][:, 2].mean()),
                    physical_output_difference_ci95=np.quantile(delta[:, 2], [.025, .975]).tolist(),
                    seconds_difference=None, seconds_difference_ci95=None, time_ratio=None, time_ratio_ci95=None)
                if same:
                    pair.update(seconds_difference=float(arrays[m][:, 1].mean() - arrays[ref][:, 1].mean()),
                        seconds_difference_ci95=np.quantile(delta[:, 1], [.025, .975]).tolist(),
                        time_ratio=float(arrays[m][:, 1].mean() / arrays[ref][:, 1].mean()),
                        time_ratio_ci95=np.quantile(samples[m][:, 1] / samples[ref][:, 1], [.025, .975]).tolist())
                pairs.append(pair)
        if 'full-1024' in methods:
            for m in methods:
                counts = collections.Counter()
                extras, after = [], []
                overhead = collections.defaultdict(list)
                for sid in chosen:
                    r, end = key[sid, m], key[sid, 'full-1024']
                    ok, terminal_ok = r['prediction'] == labels[sid], end['prediction'] == labels[sid]
                    counts['routed'] += bool(r['reasoning_invoked'])
                    counts['early_cancelled'] += bool(r['early_stop'])
                    counts['invalid'] += r['prediction'] is None
                    counts['operational_harms_vs_full'] += terminal_ok and not ok
                    counts['operational_rescues_vs_full'] += ok and not terminal_ok
                    if r.get('initial') is not None:
                        ip=r['initial']['probabilities']
                        initial_ok=max(range(len(ip)),key=ip.__getitem__)==labels[sid]
                        gain=not initial_ok and terminal_ok
                        harm=initial_ok and not terminal_ok
                        counts['gain_opportunities']+=gain
                        counts['gains_routed']+=gain and r['reasoning_invoked']
                        counts['harm_opportunities']+=harm
                        counts['harms_avoided_by_routing']+=harm and not r['reasoning_invoked']
                    if r['early_stop']:
                        counts['early_errors'] += not ok
                        counts['early_harms_vs_full'] += terminal_ok and not ok
                        counts['early_rescues_vs_full'] += ok and not terminal_ok
                    if 'prefix_answer_accepted' in r:
                        counts['accepted_after_generation_ended'] += r['prefix_answer_accepted'] and not r['early_stop']
                    if r.get('extra_tokens_since_read_prefix') is not None:
                        extras.append(r['extra_tokens_since_read_prefix'])
                    if r.get('tokens_after_stop_decision') is not None:
                        after.append(r['tokens_after_stop_decision'])
                    for field in ('gate_seconds','detector_host_seconds','trigger_host_seconds','hook_host_seconds','abort_seconds'):
                        if field in r: overhead[field].append(r[field])
                diagnostics.append(dict(dataset=task, method=m, n=len(chosen), counts=dict(counts),
                    early_error_rate=counts['early_errors'] / counts['early_cancelled'] if counts['early_cancelled'] else None,
                    early_stop_rate_among_routed=counts['early_cancelled']/counts['routed'] if counts['routed'] else None,
                    captured_gain_fraction=counts['gains_routed']/counts['gain_opportunities'] if counts['gain_opportunities'] else None,
                    early_harm_rate_all=counts['early_harms_vs_full'] / len(chosen),
                    extra_tokens_since_read_prefix=float(np.mean(extras)) if extras else None,
                    tokens_after_stop_decision=float(np.mean(after)) if after else None,
                    overshoot_available_n=len(extras),
                    post_decision_overshoot_available_n=len(after),
                    overhead_diagnostics={k:dict(mean=float(np.mean(v)),available_n=len(v)) for k,v in overhead.items()},
                    interpretation='Operational separately generated policies; not identical-trace causal stopping losses.'))
    return dict(summaries=summaries, paired=pairs, stopping=diagnostics, bootstrap_draws=draws,
                intervals='Pointwise paired stratified material-group bootstrap', accuracy_loss_tolerance=None)


def write_outputs(dest, result, scope):
    rt.save(dest / 'analysis.json', dict(UTC=rt.now(), scope=scope, **result))
    for family in ('summaries', 'paired'):
        rows = [{k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in result[family]]
        with (dest / (family + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # A reproducible preview only. It uses actual supplied values, never placeholders.
    for cost in ('pipeline_seconds', 'physical_output_tokens'):
        rows = [r for r in result['summaries'] if r['dataset'] == 'all']
        xmax = max(r[cost] for r in rows) * 1.1
        ymax, ymin = min(1., max(r['accuracy'] for r in rows) + .03), max(0., min(r['accuracy'] for r in rows) - .03)
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="540" viewBox="0 0 1100 540">',
                 '<rect width="1100" height="540" fill="white"/>',
                 '<g font-family="sans-serif" font-size="14" fill="#17212b">',
                 '<path d="M70 55 V460 H660" stroke="#17212b" fill="none"/>',
                 f'<text x="70" y="26">{html.escape(scope)}</text>']
        for k in range(6):
            x = 70 + 590 * k / 5; y = 460 - 405 * k / 5
            parts += [f'<text x="{x:.1f}" y="483" text-anchor="middle">{xmax*k/5:.1f}</text>',
                      f'<text x="62" y="{y+5:.1f}" text-anchor="end">{100*(ymin+(ymax-ymin)*k/5):.1f}</text>']
        for i, row in enumerate(rows):
            x=70+590*row[cost]/xmax; y=460-405*(row['accuracy']-ymin)/(ymax-ymin)
            color='#2166ac' if row['method'].startswith('ours') else '#555555'
            parts += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"/>',
                      f'<text x="{x+6:.2f}" y="{y-5:.2f}">{i+1}</text>',
                      f'<text x="690" y="{64+i*28}">{i+1}. {html.escape(row["method"])}</text>']
        parts += [f'<text x="360" y="515" text-anchor="middle">{cost}</text>',
                  '<text x="16" y="265" transform="rotate(-90 16 265)" text-anchor="middle">Accuracy (%)</text>', '</g></svg>']
        (dest / (cost + '.svg')).write_text('\n'.join(parts))


def prepare():
    p, projection = formal.verify_inputs()
    path = OUT / 'protocol.json'
    if path.exists():
        assert rt.read(path)['source_sha256'] == rt.sha(SOURCE), 'Analysis source changed after preparation'
        return
    rt.save(path, dict(UTC=rt.now(), source_sha256=rt.sha(SOURCE), original_protocol_sha256=rt.sha(ROOT / 'protocol.json'),
        inputs_sha256=projection['inputs_sha256'], methods=formal.CORE + formal.BASELINES,
        reference_methods=['full-1024','simple-margin-every8-1024','ours-entropy70-1024'],
        primary='All768 first executions; equal task counts; repetitions do not enter primary estimates',
        bridge='Fixed48 GPU1 core bridge paired with first GPU1 baseline executions; separate table',
        resampling='10000 paired material groups; PW subtype/LQ/ARC strata; seed20261010; pointwise95%',
        gate='No formal scoring before both phases complete and remaining comparison/analysis choices are closed',
        release_required=True, release_created=False, gold_access=False, acceptance_tolerance=None,
        pending=['attention online comparison configuration', 'formal label projection after release'],
        missing_metrics='Unavailable diagnostics remain null; absent data never silently becomes zero'))


def development_check():
    budget = formal.budget
    bp = budget.verify()
    methods = formal.CORE + formal.BASELINES
    metadata = {s['sample_id']: dict(s, stratum=s.get('stratum', s['dataset'])) for s in formal.rows(budget.OUT / 'inputs-no-gold.jsonl')}
    import sentence_online as base
    labels = {s['sample_id']: s['gold'] for s in formal.rows(base.STREAM / 'labels-observed.jsonl')}
    labels.update(budget.arc_labels(bp['sample_ids']))
    records = [rt.read(f) for f in budget.OUT.glob('gpu*/rows/*.json')]
    records = [r for r in records if r['policy'] in methods and r['repeat'] == 0]
    result = summarize(records, metadata, labels, methods)
    expected = {(r['dataset'], r['method']): r for r in rt.read(budget.OUT / 'analysis.json')['summaries']}
    for row in result['summaries']:
        old = expected[row['dataset'], row['method']]
        assert row['correct'] == old['correct'] and abs(row['pipeline_seconds'] - old['pipeline_seconds']) < 1e-8
    # Validate the two scientific bookkeeping failures that would corrupt inference.
    for bad in (records[:-1], records + records[:1]):
        try:
            summarize(bad, metadata, labels, methods, 2)
        except AssertionError:
            pass
        else:
            raise AssertionError('Incomplete/duplicate matrix accepted')
    changed = [dict(r) for r in records]
    for r in changed:
        if r['policy'] == 'native-8192': r['GPU_UUID'] = 'different-device-for-accounting-check'
    check = summarize(changed, metadata, labels, methods, 2)
    assert all(x['time_ratio'] is None for x in check['paired'] if x['method'] == 'native-8192')
    write_outputs(OUT / 'development-check', result, 'Observed development: 48 questions; not confirmation')
    rt.save(OUT / 'development-check/checks.json', dict(UTC=rt.now(), state='passed', questions=48,
        records=len(records), gold_scope='Previously observed development only', formal_predictions_opened=False,
        checks=['reproduces all13 existing development summaries', 'missing/duplicate rejection', 'cross-device ratio suppressed']))
    print('Analysis validated on 624 observed development records; formal predictions unopened')


def analyze(labels_path, release_path):
    prepare()
    for name in ('core-completion.json', 'baselines-completion.json'):
        assert rt.read(ROOT / name)['state'] == 'completed_sealed', 'Incomplete formal phase'
    release = rt.read(release_path)
    assert release['selection_closed'] is True and release['gold_access_authorized'] is True
    assert release['analysis_source_sha256'] == rt.sha(SOURCE)
    assert release['original_protocol_sha256'] == rt.sha(ROOT / 'protocol.json')
    assert release['remaining_comparisons_frozen'] is True
    assert release['attention_decision'] in ('included_with_frozen_addendum', 'explicitly_excluded_with_reason')
    assert release.get('attention_configuration_or_exclusion_reason')
    # The caller supplies an audited label projection only after the release checks.
    labels = rt.read(labels_path)
    metadata = {s['sample_id']: s for s in rt.read(ROOT / 'metadata.json')}
    assert set(labels) == set(metadata)
    rows = [rt.read(f) for folder in ('core', 'baselines') for f in (ROOT / 'sealed' / folder).glob('*.json')]
    assert len(rows) == 11232
    assert all(r['protocol_sha256'] == rt.sha(ROOT / 'protocol.json') for r in rows)
    primary = [r for r in rows if r['repeat'] == 0]
    methods = formal.CORE + formal.BASELINES
    write_outputs(OUT / 'primary', summarize(primary, metadata, labels, methods), 'Three-task confirmation; full-cohort device-specific timing')
    repeats = set(rt.read(ROOT / 'protocol.json')['repeat_ids'])
    bridge = [rt.read(f) for f in (ROOT / 'sealed/gpu1-bridge').glob('*.json')]
    assert len(bridge) == 384
    bridge += [r for r in primary if r['policy'] in formal.BASELINES and r['sample_id'] in repeats]
    assert len({r['GPU_UUID'] for r in bridge}) == 1
    write_outputs(OUT / 'same-device48', summarize(bridge, {s: metadata[s] for s in metadata if s in repeats}, labels, methods),
                  'Preselected48: same-GPU1 paired timing; no extra accuracy sample size')
    stability = []
    for method in methods:
        selected = [r for r in rows if r['policy'] == method and r['sample_id'] in repeats]
        assert len(selected) == 144
        groups = {sid: sorted([r for r in selected if r['sample_id'] == sid], key=lambda r: r['repeat']) for sid in repeats}
        assert all([r['repeat'] for r in runs] == [0,1,2] for runs in groups.values())
        stability.append(dict(method=method, n=48,
            answer_disagreements=sum(len({r['prediction'] for r in runs}) > 1 for runs in groups.values()),
            mean_within_question_seconds_sd=float(np.mean([np.std([r['pipeline_seconds'] for r in runs], ddof=1) for runs in groups.values()]))))
    rt.save(OUT / 'repeat-stability.json', stability)
    core={(r['sample_id'],r['policy']):r for r in primary if r['policy'] in formal.CORE}
    cross_device=[]
    for method in formal.CORE:
        paired=[(r,core[r['sample_id'],method]) for r in bridge if r['policy']==method]
        assert len(paired)==48
        cross_device.append(dict(method=method,n=48,
            answer_disagreements=sum(a['prediction']!=b['prediction'] for a,b in paired),
            generation_disagreements=sum(a['generated_ids']!=b['generated_ids'] for a,b in paired),
            interpretation='Different physical devices and runs; describes reproducibility, not GPU causation.'))
    rt.save(OUT/'cross-device-stability.json',cross_device)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); actions = ap.add_mutually_exclusive_group(required=True)
    for name in ('prepare','development-check','analyze'): actions.add_argument('--'+name, action='store_true')
    ap.add_argument('--labels', type=Path); ap.add_argument('--release', type=Path)
    args = ap.parse_args()
    if args.prepare: prepare()
    elif args.development_check: development_check()
    else:
        assert args.labels and args.release
        analyze(args.labels, args.release)
