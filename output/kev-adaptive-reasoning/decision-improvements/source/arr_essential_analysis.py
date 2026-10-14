"""CPU analysis addendum for the six frozen GPU0 controls.

Preparation and development validation never open confirmation predictions or
labels. Formal scoring requires complete collections and an explicit release.
The frozen collectors and original analysis module are imported, never edited.
"""
import argparse
import collections
import csv
from pathlib import Path

import numpy as np
import arr_formal_analysis as common

formal, rt = common.formal, common.rt
COLLECTION = formal.MAIN / 'paper-essential'
OUT = COLLECTION / 'analysis'
SOURCE = Path(__file__).resolve()
REFERENCE = 'ours-entropy70-1024'
METHODS = (REFERENCE, 'ablate-start', 'ablate-learned-read', 'ablate-stop',
           'route-terminal', 'ablate-benefit-gate-margin')
CONTRASTS = [(REFERENCE, m) for m in METHODS[1:]] + [('ablate-stop', 'route-terminal')]
FIELDS = ('gate_seconds', 'detector_host_seconds', 'trigger_host_seconds',
          'hook_host_seconds', 'abort_seconds', 'extra_tokens_since_read_prefix',
          'tokens_after_stop_decision')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metadata():
    """Merge the bound candidate counts absent from the original metadata file."""
    counts = {s['sample_id']: s['candidate_count']
              for s in formal.rows(formal.OUT / 'inputs-no-gold.jsonl')}
    rows = rt.read(formal.OUT / 'metadata.json')
    require({s['sample_id'] for s in rows} == set(counts), 'Metadata/input mismatch')
    return {s['sample_id']: dict(s, candidate_count=counts[s['sample_id']]) for s in rows}


def prepare(freeze=True):
    original, _ = formal.verify_inputs()
    collection = rt.read(COLLECTION / 'collection-protocol.json')
    for path, digest in collection['bindings'].items():
        require(rt.sha(path) == digest, 'Changed collection binding: ' + path)
    require(tuple(q['id'] for q in collection['policies']) == METHODS, 'Policy inventory changed')
    require(collection['total'] == 5184, 'Unexpected collection size')
    specs = {q['id']: q for q in collection['policies']}
    ref = specs[REFERENCE]
    changes = {'ablate-start': {'force_reasoning_after_gate': True},
               'ablate-learned-read': {'schedule': 'every8'},
               'ablate-stop': {'stop_rule': 'never'},
               'route-terminal': {'schedule': 'terminal'},
               'ablate-benefit-gate-margin': {'gate': 'margin'}}
    for method, change in changes.items():
        require(specs[method] == dict(ref, id=method, **change), 'Unmatched ablation: ' + method)
    meta = metadata()
    require(collection['repeat_ids'] == original['repeat_ids'], 'Repeat cohort changed')
    require(collections.Counter(s['dataset'] for s in meta.values()) ==
            {task: 256 for task in common.TASKS}, 'Unbalanced confirmation cohort')
    require(len({s['component_id'] for s in meta.values()}) == 768, 'Non-independent material groups')
    paths = [SOURCE, SOURCE.with_name('test_arr_essential_analysis.py'), Path(common.__file__), COLLECTION / 'collection-protocol.json',
             COLLECTION / 'collection-jobs.json', formal.OUT / 'protocol.json',
             formal.OUT / 'metadata.json', formal.OUT / 'inputs-no-gold.jsonl',
             formal.OUT / 'analysis/protocol.json',
             formal.OUT / 'gpu1-baselines/execution-amendment.json']
    bindings = {str(p): rt.sha(p) for p in paths}
    path = OUT / 'protocol.json'
    if path.exists():
        saved = rt.read(path)
        require(saved['bindings'] == bindings, 'Analysis changed after freeze; record an explicit amendment')
        return saved
    protocol = dict(UTC=rt.now(), state='prepared_before_confirmation_scoring', bindings=bindings,
        methods=list(METHODS), contrasts=[dict(method=m, reference=r) for m, r in CONTRASTS],
        contrast_direction='method minus reference; ratio=method/reference; saving=1-ratio',
        primary='768 first executions, 256 per task; use GPU0 complete-controller reference for all six controls',
        repeats='Fixed48 questions, runs0/1/2; average timing within question before paired bootstrap. '
                'Repeated answers describe stability; never add accuracy subjects.',
        metrics=list(common.METRICS), bootstrap_draws=10000, seed=20261010,
        resampling='Paired material groups, stratified by PW subtype/LQ/ARC; pointwise95% intervals',
        inference='Five component contrasts and one reading-overhead contrast are all reported. '
                  'Intervals are pointwise, not simultaneous; no superiority claim from selection among intervals.',
        routing='Shared GPU2 direct/full1024 endpoints define operational correction/regression opportunities; '
                'report local-initial agreement and device identities; no cross-device latency ratios.',
        stopping='GPU0 ablate-stop is the main operational endpoint; route-terminal is the secondary endpoint. '
                 'Report actual cancellation and accepted-prefix-after-generation separately. '
                 'Separate-run differences are not identical-trace causal effects.',
        costs='Measured pipeline wall time; physical output, generated and processed input tokens; full reads '
              'and trigger calls. Host timers are overlapping diagnostics, never additive GPU compute or energy.',
        missing='Optional fields retain null and available_n; required accounting omissions fail validation',
        accuracy_loss_tolerance=None, attention='Development adaptation only; excluded from formal inventory '
            'before confirmation scoring under the fixed first-paper scope',
        release_required=True, release_created=False, gold_access=False,
        required_completion=['formal core', 'GPU1 baselines and bridge', 'GPU0 six controls'],
        outputs=['main13', 'same-device48', 'six-policy', 'repeated-timing48',
                 'routing.json', 'stopping.json', 'cost-breakdown.json', 'repeat-stability.json',
                 'cross-device-reference-stability.json', 'question-pairs.csv',
                 'main-repeat-stability.json', 'main-cross-device-stability.json'])
    if freeze:
        rt.save(path, protocol)
    return protocol


def validate_matrix(records, meta, methods, repeat_ids=(), with_repeats=False, specs=None, jobs=None):
    keys = [(r['sample_id'], r['policy'], r['repeat']) for r in records]
    expected = {(sid, m, 0) for sid in meta for m in methods}
    if with_repeats:
        expected |= {(sid, m, rep) for sid in repeat_ids for m in methods for rep in (1, 2)}
    require(len(set(keys)) == len(keys) and set(keys) == expected, 'Incomplete/duplicate/extra matrix')
    by_job = {j['case_id']: j for j in jobs} if jobs is not None else None
    if by_job is not None:
        require(len(by_job) == len(jobs), 'Duplicate expected job id')
        require({r['case_id'] for r in records} == set(by_job), 'Job-id mismatch')
    for r in records:
        s = meta[r['sample_id']]
        require(r['dataset'] == s['dataset'], 'Dataset mismatch')
        if specs is not None:
            formal.row_checks(r, s, specs[r['policy']])
        if by_job is not None:
            j = by_job[r['case_id']]
            require(all(r[k] == j[k] for k in ('sample_id', 'policy', 'repeat')), 'Job record mismatch')
        pred = r['prediction']
        require(pred is None or type(pred) is int and 0 <= pred < s['candidate_count'], 'Invalid answer index')
        for field, alternatives in [
            ('request_input_tokens', ('reader_prompt_tokens', 'generator_prompt_tokens')),
            ('request_cached_tokens', ('reader_cached_tokens', 'generator_cached_tokens'))]:
            require(field in r or all(k in r for k in alternatives), 'Missing required cost: ' + field)
        common.metric_row(r, 0)
        require(bool(r.get('GPU_UUID')), 'Missing device identity')


def validate_labels(labels, meta):
    require(set(labels) == set(meta), 'Label projection must cover exactly the frozen cohort')
    for sid, gold in labels.items():
        require(type(gold) is int and 0 <= gold < meta[sid]['candidate_count'], 'Invalid zero-based gold')


def paired_comparisons(records, meta, labels, contrasts=CONTRASTS, draws=10000):
    key = {(r['sample_id'], r['policy']): r for r in records}
    require(all(r['repeat'] == 0 for r in records) and len(key) == len(records), 'Primary runs only')
    output = []
    for task in ('all',) + common.TASKS:
        ids = [s for s in meta if task == 'all' or meta[s]['dataset'] == task]
        if not ids:
            continue
        ix = common.bootstrap_indices(ids, meta, draws, 20261010)
        for method, reference in contrasts:
            a = np.array([common.metric_row(key[s, method], labels[s]) for s in ids])
            b = np.array([common.metric_row(key[s, reference], labels[s]) for s in ids])
            same = all(key[s, method]['GPU_UUID'] == key[s, reference]['GPU_UUID'] for s in ids)
            stats = {}
            for col, metric in enumerate(common.METRICS):
                if metric == 'pipeline_seconds' and not same:
                    stats[metric] = dict(difference=None, ci95=None, ratio=None, ratio_ci95=None,
                                         saving_fraction=None, saving_fraction_ci95=None)
                    continue
                av, bv = a[ix, col].mean(1), b[ix, col].mean(1)
                stats[metric] = dict(difference=float((a[:, col] - b[:, col]).mean()),
                    ci95=np.quantile(av - bv, [.025, .975]).tolist())
                if metric in ('pipeline_seconds', 'physical_output_tokens', 'generation_tokens',
                              'full_reads', 'trigger_calls', 'input_tokens'):
                    valid = b[:, col].mean() > 0 and np.all(bv > 0)
                    ratio = float(a[:, col].mean() / b[:, col].mean()) if b[:, col].mean() > 0 else None
                    interval = np.quantile(av / bv, [.025, .975]).tolist() if valid else None
                    stats[metric].update(ratio=ratio, ratio_ci95=interval,
                        saving_fraction=None if ratio is None else 1 - ratio,
                        saving_fraction_ci95=None if interval is None else [1 - interval[1], 1 - interval[0]])
            output.append(dict(dataset=task, method=method, reference=reference, n=len(ids),
                same_device=same, metrics=stats,
                corrected=int(((a[:, 0] == 1) & (b[:, 0] == 0)).sum()),
                regressed=int(((a[:, 0] == 0) & (b[:, 0] == 1)).sum()),
                answer_changes=sum(key[s, method]['prediction'] != key[s, reference]['prediction'] for s in ids),
                routing_changes=sum(key[s, method]['reasoning_invoked'] != key[s, reference]['reasoning_invoked'] for s in ids)))
    return output


def rate(numerator, denominator):
    return numerator / denominator if denominator else None


def question_pairs(records, meta, labels, contrasts=CONTRASTS):
    key = {(r['sample_id'], r['policy']): r for r in records}
    out = []
    for sid in meta:
        for method, reference in contrasts:
            a, b = key[sid, method], key[sid, reference]
            same = a['GPU_UUID'] == b['GPU_UUID']
            diff = {name + '_difference': float(x - y) for name, x, y in zip(
                common.METRICS, common.metric_row(a, labels[sid]), common.metric_row(b, labels[sid]))}
            if not same:
                diff['pipeline_seconds_difference'] = None
            out.append(dict(sample_id=sid, component_id=meta[sid]['component_id'],
                dataset=meta[sid]['dataset'], method=method, reference=reference,
                method_device=a['GPU_UUID'], reference_device=b['GPU_UUID'], same_device=same,
                gold=labels[sid], prediction=a['prediction'], reference_prediction=b['prediction'],
                routed=a['reasoning_invoked'], reference_routed=b['reasoning_invoked'],
                early_stop=a['early_stop'], reference_early_stop=b['early_stop'], **diff))
    return out


def write_question_pairs(path, records):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def diagnostics(records, meta, labels, endpoints, stop_refs=('ablate-stop', 'route-terminal')):
    key = {(r['sample_id'], r['policy']): r for r in records}
    end = {(r['sample_id'], r['policy']): r for r in endpoints}
    methods = list(dict.fromkeys(r['policy'] for r in records))
    routing, stopping, costs = [], [], []
    for task in ('all',) + common.TASKS:
        ids = [s for s in meta if task == 'all' or meta[s]['dataset'] == task]
        if not ids:
            continue
        for method in methods:
            rows = [key[s, method] for s in ids]
            c = dict.fromkeys(('gain_opportunities', 'gains_routed', 'gain_opportunities_skipped',
                'regression_opportunities', 'regressions_skipped', 'regressions_routed',
                'unchanged_opportunities', 'unchanged_routed', 'routed', 'initial_available',
                'initial_disagrees_with_shared_direct', 'actual_corrections_vs_direct',
                'actual_regressions_vs_direct', 'routed_gains_actually_correct'), 0)
            for sid, r in zip(ids, rows):
                d, f = end[sid, 'direct'], end[sid, 'full-1024']
                direct_ok, full_ok = d['prediction'] == labels[sid], f['prediction'] == labels[sid]
                ok, routed = r['prediction'] == labels[sid], r['reasoning_invoked']
                gain, harm = not direct_ok and full_ok, direct_ok and not full_ok
                c['gain_opportunities'] += gain
                c['gains_routed'] += gain and routed
                c['gain_opportunities_skipped'] += gain and not routed
                c['regression_opportunities'] += harm
                c['regressions_skipped'] += harm and not routed
                c['regressions_routed'] += harm and routed
                c['unchanged_opportunities'] += direct_ok == full_ok
                c['unchanged_routed'] += direct_ok == full_ok and routed
                c['routed'] += bool(routed)
                c['routed_gains_actually_correct'] += gain and routed and ok
                c['actual_corrections_vs_direct'] += not direct_ok and ok
                c['actual_regressions_vs_direct'] += direct_ok and not ok
                if r.get('initial') is not None:
                    p = r['initial']['probabilities']
                    c['initial_available'] += 1
                    c['initial_disagrees_with_shared_direct'] += p.index(max(p)) != d['prediction']
            routing.append(dict(dataset=task, method=method, n=len(ids), counts=c,
                gain_capture=rate(c['gains_routed'], c['gain_opportunities']),
                regression_avoidance=rate(c['regressions_skipped'], c['regression_opportunities']),
                gain_precision_among_routed=rate(c['gains_routed'], c['routed']),
                same_device_as_endpoints=all(r['GPU_UUID'] == end[s, 'direct']['GPU_UUID'] ==
                    end[s, 'full-1024']['GPU_UUID'] for s, r in zip(ids, rows)),
                interpretation='Shared separately generated endpoint labels; capture is not guaranteed actual rescue.'))
            for stop_ref in stop_refs:
                if stop_ref == method:
                    continue
                c = dict.fromkeys(('routed', 'cancelled', 'cancelled_errors', 'cancelled_harms',
                    'cancelled_rescues', 'accepted_after_generation', 'late_acceptance_errors',
                    'late_acceptance_harms', 'late_acceptance_rescues', 'route_disagreements',
                    'initial_disagreements', 'matched_routed_cancelled', 'matched_routed_harms',
                    'matched_routed_rescues', 'cancelled_reference_correct'), 0)
                for sid, r in zip(ids, rows):
                    ref = key[sid, stop_ref]
                    ok, ref_ok = r['prediction'] == labels[sid], ref['prediction'] == labels[sid]
                    c['routed'] += bool(r['reasoning_invoked'])
                    c['route_disagreements'] += r['reasoning_invoked'] != ref['reasoning_invoked']
                    if r.get('initial') and ref.get('initial'):
                        a, b = r['initial']['probabilities'], ref['initial']['probabilities']
                        c['initial_disagreements'] += a.index(max(a)) != b.index(max(b))
                    if r['early_stop']:
                        c['cancelled'] += 1
                        c['cancelled_errors'] += not ok
                        c['cancelled_harms'] += ref_ok and not ok
                        c['cancelled_rescues'] += ok and not ref_ok
                        c['cancelled_reference_correct'] += ref_ok
                        if ref['reasoning_invoked'] and r['reasoning_invoked']:
                            c['matched_routed_cancelled'] += 1
                            c['matched_routed_harms'] += ref_ok and not ok
                            c['matched_routed_rescues'] += ok and not ref_ok
                    if r.get('prefix_answer_accepted', False) and not r['early_stop']:
                        c['accepted_after_generation'] += 1
                        c['late_acceptance_errors'] += not ok
                        c['late_acceptance_harms'] += ref_ok and not ok
                        c['late_acceptance_rescues'] += ok and not ref_ok
                stopping.append(dict(dataset=task, method=method, reference=stop_ref, n=len(ids), counts=c,
                    cancelled_error_rate=rate(c['cancelled_errors'], c['cancelled']),
                    cancelled_harm_rate_all=rate(c['cancelled_harms'], len(ids)),
                    cancelled_harm_rate_among_cancelled=rate(c['cancelled_harms'], c['cancelled']),
                    stop_rate_among_routed=rate(c['cancelled'], c['routed']),
                    same_device=all(r['GPU_UUID'] == key[s, stop_ref]['GPU_UUID'] for s, r in zip(ids, rows)),
                    interpretation='Operational separate-run comparison; inspect route/initial disagreement and matched-routed counts.'))
            components = collections.defaultdict(list)
            for r in rows:
                for field in FIELDS:
                    if r.get(field) is not None:
                        components[field].append(r[field])
                checks = r.get('checks')
                if checks is not None:
                    components['intermediate_reads'].append(sum(not q['terminal'] for q in checks))
                    components['terminal_reads'].append(sum(q['terminal'] for q in checks))
                if 'initial_reads' in r:
                    components['initial_reads'].append(r['initial_reads'])
                if r.get('initial') and 'seconds' in r['initial']:
                    components['initial_response_seconds'].append(r['initial']['seconds'])
                if checks is not None and all('response_seconds' in q for q in checks):
                    components['sum_read_response_seconds'].append(sum(q['response_seconds'] for q in checks))
            names = FIELDS + ('initial_reads', 'intermediate_reads', 'terminal_reads',
                             'initial_response_seconds', 'sum_read_response_seconds')
            costs.append(dict(dataset=task, method=method, n=len(ids), fields={k: dict(
                mean=float(np.mean(components[k])) if components[k] else None,
                available_n=len(components[k])) for k in names},
                interpretation='Response/host timers overlap generation and each other; do not sum into pipeline time.'))
    return dict(routing=routing, stopping=stopping, cost_breakdown=costs)


def repeated_timing(records, meta, labels, methods, repeat_ids, contrasts=CONTRASTS, draws=10000):
    selected = {s: meta[s] for s in meta if s in set(repeat_ids)}
    key = {(r['sample_id'], r['policy'], r['repeat']): r for r in records}
    means, stability = [], []
    for method in methods:
        groups = [[key[s, method, rep] for rep in (0, 1, 2)] for s in selected]
        stability.append(dict(method=method, n=len(groups),
            answer_disagreements=sum(len({r['prediction'] for r in runs}) > 1 for runs in groups),
            route_disagreements=sum(len({r['reasoning_invoked'] for r in runs}) > 1 for runs in groups),
            generation_disagreements=sum(len({tuple(r['generated_ids']) for r in runs}) > 1 for runs in groups),
            mean_within_question_seconds_sd=float(np.mean([
                np.std([r['pipeline_seconds'] for r in runs], ddof=1) for runs in groups]))))
        for runs in groups:
            require(len({r['GPU_UUID'] for r in runs}) == 1, 'Repeat device changed')
            means.append(dict(runs[0], pipeline_seconds=float(np.mean([r['pipeline_seconds'] for r in runs]))))
    # Only wall time is aggregated across repeats; do not emit hybrid accuracy/token summaries.
    comparisons = paired_comparisons(means, selected, labels, contrasts, draws)
    timing = [dict(dataset=r['dataset'], method=r['method'], reference=r['reference'],
        n=r['n'], same_device=r['same_device'], **r['metrics']['pipeline_seconds']) for r in comparisons]
    return dict(timing=timing, stability=stability,
        interpretation='Three times averaged within each fixed question, then paired question bootstrap. Accuracy remains first-run only.')


def write_table(dest, result):
    lines = ['# Six-policy matched results', '',
             '|Task|Policy|Correct/n|Accuracy|Seconds|Output tokens|Reads|Reasoning rate|Early-stop rate|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in result['summaries']:
        lines.append(f'|{r["dataset"]}|{r["method"]}|{r["correct"]}/{r["n"]}|{r["accuracy"]:.4f}|'
                     f'{r["pipeline_seconds"]:.4f}|{r["physical_output_tokens"]:.2f}|{r["full_reads"]:.3f}|'
                     f'{r["reasoning_rate"]:.4f}|{r["early_stop_rate"]:.4f}|')
    (dest / 'table.md').write_text('\n'.join(lines) + '\n')


def release_checks(labels_path, release_path):
    prepare()
    for path, state, total in [
        (formal.OUT / 'core-completion.json', 'completed_sealed', 6912),
        (formal.OUT / 'baselines-completion.json', 'completed_sealed', 4704),
        (COLLECTION / 'completion.json', 'completed_unscored', 5184)]:
        require(path.exists(), 'Collection incomplete: ' + str(path))
        value = rt.read(path)
        require(value['state'] == state and value.get('completed', value.get('cases')) == total
                and value.get('total', value.get('cases')) == total,
                'Incomplete collection: ' + str(path))
    release = rt.read(release_path)
    for field in ('selection_closed', 'gold_access_authorized', 'remaining_comparisons_frozen'):
        require(release.get(field) is True, 'Release missing: ' + field)
    for field, path in [
        ('analysis_source_sha256', Path(common.__file__)), ('essential_analysis_source_sha256', SOURCE),
        ('original_protocol_sha256', formal.OUT / 'protocol.json'),
        ('essential_collection_protocol_sha256', COLLECTION / 'collection-protocol.json'),
        ('essential_analysis_protocol_sha256', OUT / 'protocol.json'), ('labels_sha256', labels_path)]:
        require(release.get(field) == rt.sha(path), 'Release binding mismatch: ' + field)
    require(release.get('attention_decision') == 'explicitly_excluded_with_reason' and
            bool(release.get('attention_configuration_or_exclusion_reason')), 'Attention scope not recorded')
    require(release.get('label_projection_audited') is True, 'Label projection not audited')


def load_rows(folder, jobs, meta, specs, digest_field, digest, gpu, extra=None):
    files = sorted(folder.glob('*.json'))
    require({p.stem for p in files} == {j['case_id'] for j in jobs}, 'Missing or extra sealed records')
    rows = [rt.read(p) for p in files]
    require(all(r[digest_field] == digest and r['GPU_UUID'] == rt.GPUS[gpu] for r in rows),
            'Record provenance mismatch')
    if extra:
        require(all(all(r[k] == v for k, v in extra.items()) for r in rows), 'Execution amendment mismatch')
    repeats = {j['sample_id'] for j in jobs if j['repeat'] > 0}
    selected = {s: meta[s] for s in meta if s in {j['sample_id'] for j in jobs}}
    methods = list(dict.fromkeys(j['policy'] for j in jobs))
    validate_matrix(rows, selected, methods, repeats, bool(repeats), specs, jobs)
    return rows


def analyze(labels_path, release_path):
    # Completion and authorization are checked before opening any sealed row or gold.
    release_checks(labels_path, release_path)
    meta = metadata()
    labels = rt.read(labels_path)
    validate_labels(labels, meta)
    protocol = rt.read(formal.OUT / 'protocol.json')
    jobs = rt.read(formal.OUT / 'jobs.json')
    specs = {q['id']: q for q in protocol['policies']}
    original_hash = rt.sha(formal.OUT / 'protocol.json')
    core = load_rows(formal.OUT / 'sealed/core', jobs['core'], meta, specs,
                     'protocol_sha256', original_hash, 2)
    amendment_dir = formal.OUT / 'gpu1-baselines'
    amendment = rt.read(amendment_dir / 'execution-amendment.json')
    for path, digest in amendment['bindings'].items():
        require(rt.sha(path) == digest, 'Changed GPU1 binding: ' + path)
    baseline = load_rows(formal.OUT / 'sealed/baselines', jobs['baselines'], meta, specs,
        'protocol_sha256', original_hash, 1,
        {'execution_amendment_sha256': rt.sha(amendment_dir / 'execution-amendment.json')})
    collection = rt.read(COLLECTION / 'collection-protocol.json')
    controls = load_rows(COLLECTION / 'sealed', rt.read(COLLECTION / 'collection-jobs.json'), meta,
        {q['id']: q for q in collection['policies']}, 'collection_protocol_sha256',
        rt.sha(COLLECTION / 'collection-protocol.json'), 0)
    primary = [r for r in controls if r['repeat'] == 0]
    main_primary = [r for r in core + baseline if r['repeat'] == 0]
    result = common.summarize(primary, meta, labels, METHODS)
    result['paired'] = paired_comparisons(primary, meta, labels)
    common.write_outputs(OUT / 'six-policy', result, '768 confirmation questions; matched GPU0 controls')
    write_table(OUT / 'six-policy', result)
    write_question_pairs(OUT / 'question-pairs.csv', question_pairs(primary, meta, labels))
    endpoints = [r for r in main_primary if r['policy'] in ('direct', 'full-1024')]
    for name, value in diagnostics(primary, meta, labels, endpoints).items():
        rt.save(OUT / (name.replace('_', '-') + '.json'), value)
    repeat = repeated_timing(controls, meta, labels, METHODS, collection['repeat_ids'])
    rt.save(OUT / 'repeated-timing48.json', repeat['timing'])
    rt.save(OUT / 'repeat-stability.json', repeat['stability'])
    # Integrate the unchanged 13-policy analysis with candidate counts from bound inputs.
    common.write_outputs(OUT / 'main13', common.summarize(main_primary, meta, labels,
        formal.CORE + formal.BASELINES), '768 confirmation questions; device-specific costs')
    bridge_jobs = [j for j in rt.read(amendment_dir / 'jobs.json') if j['role'] == 'bridge']
    bridge = load_rows(formal.OUT / 'sealed/gpu1-bridge', bridge_jobs, meta, specs,
        'protocol_sha256', original_hash, 1,
        {'execution_amendment_sha256': rt.sha(amendment_dir / 'execution-amendment.json')})
    repeat_ids = set(protocol['repeat_ids'])
    bridge += [r for r in main_primary if r['policy'] in formal.BASELINES and r['sample_id'] in repeat_ids]
    common.write_outputs(OUT / 'same-device48', common.summarize(bridge,
        {s: meta[s] for s in meta if s in repeat_ids}, labels, formal.CORE + formal.BASELINES),
        'Fixed48 questions; paired GPU1 comparison; not extra accuracy subjects')
    main_methods = formal.CORE + formal.BASELINES
    main_repeats = repeated_timing(core + baseline, meta, labels, main_methods, protocol['repeat_ids'],
        [(m, REFERENCE) for m in main_methods if m != REFERENCE])
    rt.save(OUT / 'main-repeat-stability.json', main_repeats['stability'])
    rt.save(OUT / 'main-repeated-timing48.json', main_repeats['timing'])
    original_core = {(r['sample_id'], r['policy']): r for r in main_primary if r['policy'] in formal.CORE}
    cross = []
    for method in formal.CORE:
        pairs = [(r, original_core[r['sample_id'], method]) for r in bridge if r['policy'] == method]
        require(len(pairs) == 48, 'Incomplete original same-device bridge')
        cross.append(dict(method=method, n=48,
            answer_disagreements=sum(a['prediction'] != b['prediction'] for a, b in pairs),
            generation_disagreements=sum(a['generated_ids'] != b['generated_ids'] for a, b in pairs),
            interpretation='GPU1/GPU2 repeated configuration; not a causal estimate of device effects.'))
    rt.save(OUT / 'main-cross-device-stability.json', cross)
    a = {r['sample_id']: r for r in primary if r['policy'] == REFERENCE}
    b = {r['sample_id']: r for r in main_primary if r['policy'] == REFERENCE}
    rt.save(OUT / 'cross-device-reference-stability.json', dict(n=768,
        answer_disagreements=sum(a[s]['prediction'] != b[s]['prediction'] for s in meta),
        generation_disagreements=sum(a[s]['generated_ids'] != b[s]['generated_ids'] for s in meta),
        route_disagreements=sum(a[s]['reasoning_invoked'] != b[s]['reasoning_invoked'] for s in meta),
        interpretation='GPU0/GPU2 complete-policy reproducibility; no cross-device speedup inference.'))
    rt.save(OUT / 'completion.json', dict(UTC=rt.now(), state='scored_after_release',
        primary_questions=768, control_records=len(controls), release_sha256=rt.sha(release_path),
        labels_sha256=rt.sha(labels_path), analysis_protocol_sha256=rt.sha(OUT / 'protocol.json')))


def development_check():
    """Real observed data keep their original names; no proxy six-policy results."""
    prepare(freeze=False)
    budget = formal.budget
    bp = budget.verify()
    meta = {s['sample_id']: s for s in formal.rows(budget.OUT / 'inputs-no-gold.jsonl')}
    import sentence_online as original
    labels = {s['sample_id']: s['gold'] for s in formal.rows(original.STREAM / 'labels-observed.jsonl')
              if s['sample_id'] in meta}
    labels.update(budget.arc_labels(bp['sample_ids']))
    methods = formal.CORE + formal.BASELINES
    rows = [rt.read(p) for p in budget.OUT.glob('gpu*/rows/*.json')]
    rows = [r for r in rows if r['policy'] in methods and r['repeat'] == 0]
    validate_labels(labels, meta)
    validate_matrix(rows, meta, methods)
    result = common.summarize(rows, meta, labels, methods, draws=200)
    expected = {(r['dataset'], r['method']): r for r in rt.read(budget.OUT / 'analysis.json')['summaries']}
    for r in result['summaries']:
        old = expected[r['dataset'], r['method']]
        require(r['correct'] == old['correct'] and abs(r['pipeline_seconds'] - old['pipeline_seconds']) < 1e-8,
                'Observed development reproduction failed')
    pairs = [(REFERENCE, 'full-1024'), (REFERENCE, 'simple-margin-every8-1024')]
    result['paired'] = paired_comparisons(rows, meta, labels, pairs, draws=200)
    control = next(x for x in result['paired'] if x['dataset'] == 'all' and x['reference'] == 'full-1024')
    require(abs(control['metrics']['accuracy']['difference'] - 1 / 48) < 1e-12, 'Paired direction error')
    subset = [r for r in rows if r['policy'] == REFERENCE]
    diagnostic_rows = subset + [r for r in rows if r['policy'] == 'full-1024']
    ds = diagnostics(diagnostic_rows, meta, labels,
        [r for r in rows if r['policy'] in ('direct', 'full-1024')], stop_refs=('full-1024',))
    common.write_outputs(OUT / 'development-check', result,
        'Observed48 development only; original policy names; 200-draw code check')
    write_question_pairs(OUT / 'development-check/question-pairs.csv', question_pairs(rows, meta, labels, pairs))
    rt.save(OUT / 'development-check/diagnostics.json', ds)
    rt.save(OUT / 'development-check/checks.json', dict(UTC=rt.now(), state='passed',
        questions=48, records=624, formal_predictions_opened=False, formal_labels_opened=False,
        checks=['all13 original accuracy/time summaries reproduced', 'paired difference sign',
                'real-record routing/stopping/cost schema'],
        six_policy_outcomes='Not available; synthetic unit checks test arithmetic only'))
    print('Passed: 624 observed development rows; no confirmation predictions or labels opened')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'development-check', 'analyze'):
        action.add_argument('--' + name, action='store_true')
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--release', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        print('Six-policy analysis registered; no GPU work, formal scoring or release')
    elif args.development_check:
        development_check()
    else:
        require(args.labels is not None and args.release is not None, 'Supply labels and release explicitly')
        analyze(args.labels, args.release)
