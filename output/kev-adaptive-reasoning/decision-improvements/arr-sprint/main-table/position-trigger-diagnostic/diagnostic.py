"""Isolated CPU diagnostics on existing sentence caches. No legacy imports/writes."""
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Require CPU-only visibility'
import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[5]
BASE = PROJECT / 'output/kev-adaptive-reasoning/decision-improvements'
COLLECT = BASE / 'sentence-supervision'
STREAM = BASE.parent / 'stream-readout'
MODELS = ['hidden', 'position_lr', 'position_lookup', 'constant']
SEED = 20261012


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, value):
    p = OUT / name
    tmp = p.with_suffix(p.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def status(phase, **extra):
    value = dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase, **extra)
    save('status.json', value)
    print(value, flush=True)


def entropy_certainty(p):
    p = np.asarray(p, dtype=float)
    return float(1 + np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p)))


def load_data():
    assert read(COLLECT / 'completion.json')['state'] == 'completed'
    work = read(COLLECT / 'work.json')
    feature_index = {r['index']: r for r in read(COLLECT / 'feature-index.json')}
    labels = {r['sample_id']: r['gold'] for r in rows(STREAM / 'labels-observed.jsonl')}
    source_samples = rows(STREAM / 'samples.jsonl')
    train = [r for r in source_samples if r['split'] == 'train']
    dev = [r for r in source_samples if r['split'] == 'cal']
    a = torch.load(STREAM / 'stopping-grid/oof-probabilities.pt', map_location='cpu', weights_only=True)
    b = torch.load(STREAM / 'training/answer_position_linear-s29.pt', map_location='cpu', weights_only=True)
    assert a['sample_ids'] == [r['sample_id'] for r in train]
    assert b['sample_ids'] == [r['sample_id'] for r in dev]
    terminals = {r['sample_id']: p[-1, :r['candidate_count']].numpy().astype(float)
                 for samples, saved in [(train, a), (dev, b)]
                 for r, p in zip(samples, saved['probabilities'])}
    assert {r['sample_id'] for r in work} == set(terminals)
    assert not ({r['component_id'] for r in work if r['split'] == 'train'} &
                {r['component_id'] for r in work if r['split'] == 'cal'})
    trigger = read(BASE / 'arr-sprint/main-table/task-agnostic-trigger/trigger.json')
    assert trigger['entropy_threshold'] == .85 and trigger['uses_task_identity'] is False
    hidden_w = np.array(trigger['hidden_weight'], dtype=float)
    all_events, samples = [], []
    coalesced = excluded = 0
    for r in work:
        path = COLLECT / 'features' / f"{r['index']:04}.pt"
        assert sha(path) == feature_index[r['index']]['sha256']
        f = torch.load(path, map_location='cpu', weights_only=True)
        assert f['sample_id'] == r['sample_id'] and f['events'] == r['events']
        assert f['input_hashes'] == [ev['input_sha256'] for ev in r['events']]
        assert f['probabilities'].shape == (len(r['events']), r['candidate_count'])
        assert torch.isfinite(f['probabilities']).all() and torch.isfinite(f['natural_states']).all()
        if r['split'] == 'cal':
            assert r['available_sentence_events'] == len(r['events']), 'Need complete dev sentence order'
        valid = [j for j, e in enumerate(r['events']) if e['observable_after_tokens'] < r['terminal_tokens']]
        excluded += len(r['events']) - len(valid)
        selected = sorted({r['events'][j]['observable_after_tokens']: j for j in valid}.values())
        coalesced += len(valid) - len(selected)
        denom = sum(r['events'][j]['training_weight'] for j in selected) if r['split'] == 'train' else 1
        sample = {k: r[k] for k in ['sample_id', 'component_id', 'split', 'dataset', 'candidate_count', 'terminal_tokens']}
        sample.update(gold=labels[r['sample_id']], terminal_probabilities=terminals[r['sample_id']], points=[])
        sample['terminal_correct'] = bool(sample['terminal_probabilities'].argmax() == sample['gold'])
        natural = f['natural_states'].float().numpy().astype(np.float64)
        pos = np.array([ev['prefix_tokens'] / 1024 for ev in r['events']])
        logits = natural @ hidden_w + pos * trigger['position_weight'] + trigger['bias']
        scores = 1 / (1 + np.exp(-logits))
        for j in selected:
            ev = r['events'][j]
            p = f['probabilities'][j].numpy().astype(float)
            assert abs(p.sum() - 1) < 1e-5
            gid = len(all_events)
            item = dict(sample_id=r['sample_id'], component_id=r['component_id'], dataset=r['dataset'],
                        train=r['split'] == 'train', prefix_tokens=ev['prefix_tokens'],
                        ordinal=j + 1 if r['split'] == 'cal' else None,
                        observable_after_tokens=ev['observable_after_tokens'],
                        train_weight=float(ev['training_weight'] / denom) if r['split'] == 'train' else 0,
                        eval_weight=1 / len(selected), certainty=entropy_certainty(p),
                        correct=bool(p.argmax() == sample['gold']), pred=int(p.argmax()), hidden=float(scores[j]))
            sample['points'].append(gid)
            all_events.append(item)
        if sample['points'] and r['split'] == 'train':
            assert abs(sum(all_events[i]['train_weight'] for i in sample['points']) - 1) < 1e-6
        samples.append(sample)
        if len(samples) % 256 == 0:
            status('loading', completed=len(samples), total=len(work))
    assert sum(s['terminal_correct'] for s in samples if s['split'] == 'cal') == 203
    assert sum(s['split'] == 'train' for s in samples) == 1024
    assert sum(s['split'] == 'cal' for s in samples) == 256
    save('data-audit.json', dict(questions=len(samples), events=len(all_events),
        train_events=sum(r['train'] for r in all_events), dev_events=sum(not r['train'] for r in all_events),
        coalesced_same_observable_time=coalesced, excluded_not_early=excluded,
        feature_hashes_verified=True, material_groups_disjoint=True,
        dev_full_sentence_order_verified=True, train_ordinals_unassigned=True,
        sentence_backend='RTX3080', terminal_backend='A100', fresh_or_arc_access=False))
    return all_events, samples


def metrics(y, score, weight):
    if weight.sum() == 0:
        return dict(logloss=None, auroc=None, ap=None)
    both = len(np.unique(y[weight > 0])) > 1
    return dict(logloss=float(log_loss(y, score, sample_weight=weight, labels=[0, 1])),
                auroc=float(roc_auc_score(y, score, sample_weight=weight)) if both else None,
                ap=float(average_precision_score(y, score, sample_weight=weight)) if np.any(y[weight > 0]) else None)


def fit_and_evaluate(events):
    train = np.array([r['train'] for r in events])
    pos = np.array([r['prefix_tokens'] / 1024 for r in events])
    bins = np.minimum((pos * 16).astype(int), 15)
    tw = np.array([r['train_weight'] for r in events])
    qw = np.array([r['eval_weight'] for r in events])
    correct = np.array([r['correct'] for r in events])
    certainty = np.array([r['certainty'] for r in events])
    y85 = (correct & (certainty >= .85)).astype(int)
    model = LogisticRegression(C=.01, solver='lbfgs', max_iter=3000, tol=1e-7, random_state=20261009)
    model.fit(pos[train, None], y85[train], sample_weight=tw[train])
    table = [(float(np.sum(tw[train & (bins == b)] * y85[train & (bins == b)])) + 1) /
             (float(tw[train & (bins == b)].sum()) + 2) for b in range(16)]
    constant = float(np.average(y85[train], weights=tw[train]))
    scores = dict(hidden=np.array([r['hidden'] for r in events]),
                  position_lr=model.predict_proba(pos[:, None])[:, 1],
                  position_lookup=np.array(table)[bins], constant=np.full(len(events), constant))
    joblib.dump(model, OUT / 'position-lr.joblib')
    save('fitted-baselines.json', dict(target=.85, C=.01, position_divisor=1024,
        lr_coefficient=float(model.coef_[0, 0]), lr_intercept=float(model.intercept_[0]),
        lookup_token_width=64, lookup_smoothing='(weighted positive mass + 1)/(weighted mass + 2)',
        lookup_values=table, constant=constant, fitted_training_questions=1024))
    summaries, band_rows = [], []
    datasets = np.array([r['dataset'] for r in events])
    for target in [.85, .70]:
        y = (correct & (certainty >= target)).astype(int)
        for split, splitmask in [('train', train), ('development', ~train)]:
            for dataset in ['all', 'proofwriter', 'logiqa2']:
                mask = splitmask & ((datasets == dataset) if dataset != 'all' else True)
                for name, score in scores.items():
                    summaries.append(dict(target=target, split=split, dataset=dataset, controller=name,
                        questions=len({r['sample_id'] for r, k in zip(events, mask) if k}), events=int(mask.sum()),
                        weighted_prevalence=float(np.average(y[mask], weights=qw[mask])),
                        **metrics(y[mask], score[mask], qw[mask])))
        for dataset in ['all', 'proofwriter', 'logiqa2']:
            mask = ~train & ((datasets == dataset) if dataset != 'all' else True)
            for b in range(16):
                bm = mask & (bins == b)
                if not bm.any():
                    continue
                # Renormalize each question within each band for descriptive band metrics.
                counts = Counter(r['sample_id'] for r, k in zip(events, bm) if k)
                bw = np.array([1 / counts[r['sample_id']] for r, k in zip(events, bm) if k])
                for name, score in scores.items():
                    band_rows.append(dict(target=target, dataset=dataset, token_lo=64 * b, token_hi=64 * (b + 1),
                        controller=name, questions=len(counts), events=int(bm.sum()),
                        positive_weight=float(bw @ y[bm]), negative_weight=float(bw @ (1-y[bm])),
                        **metrics(y[bm], score[bm], bw)))
    save('predictive-metrics.json', summaries)
    save('length-band-metrics.json', band_rows)
    conditioned = []
    for target in [.85, .70]:
        for dataset in ['all', 'proofwriter', 'logiqa2']:
            for name in MODELS:
                chosen = [r for r in band_rows if r['target'] == target and r['dataset'] == dataset and
                          r['controller'] == name and r['auroc'] is not None]
                pairmass = np.array([r['positive_weight'] * r['negative_weight'] for r in chosen])
                conditioned.append(dict(target=target, dataset=dataset, controller=name,
                    valid_bands=len(chosen), pooled_within_band_auroc=float(np.average([r['auroc'] for r in chosen], weights=pairmass)),
                    definition='64-token bands; question-normalized within each band; aggregate AUC by within-band positive-negative pair mass'))
    save('length-conditioned-summary.json', conditioned)
    return train, certainty, correct, qw, scores, summaries, conditioned


def paired_bootstrap(events, train, certainty, correct, qw, scores):
    rng = np.random.default_rng(SEED)
    results = []
    for dataset in ['all', 'proofwriter', 'logiqa2']:
        idx = np.array([i for i, r in enumerate(events) if not train[i] and (dataset == 'all' or r['dataset'] == dataset)])
        groups = sorted({events[i]['component_id'] for i in idx})
        group_index = {g: j for j, g in enumerate(groups)}
        event_group = np.array([group_index[events[i]['component_id']] for i in idx])
        strata = [np.array([group_index[g] for g in groups if any(events[i]['dataset'] == d and events[i]['component_id'] == g for i in idx)])
                  for d in ['proofwriter', 'logiqa2']]
        strata = [s for s in strata if len(s)]
        assert sum(map(len, strata)) == len(groups)
        for target in [.85, .70]:
            y = (correct[idx] & (certainty[idx] >= target)).astype(int)
            base_weight = qw[idx]
            point = {name: metrics(y, scores[name][idx], base_weight) for name in MODELS[:3]}
            draws = {base: {m: [] for m in ['logloss', 'auroc', 'ap']} for base in MODELS[1:3]}
            for repeat in range(1000):
                multiplicity = np.zeros(len(groups), dtype=int)
                for stratum in strata:
                    multiplicity += np.bincount(rng.choice(stratum, size=len(stratum), replace=True), minlength=len(groups))
                weight = base_weight * multiplicity[event_group]
                measured = {name: metrics(y, scores[name][idx], weight) for name in MODELS[:3]}
                for base in MODELS[1:3]:
                    for metric in draws[base]:
                        if measured['hidden'][metric] is not None and measured[base][metric] is not None:
                            draws[base][metric].append(measured['hidden'][metric] - measured[base][metric])
            for base in MODELS[1:3]:
                for metric, vals in draws[base].items():
                    results.append(dict(target=target, dataset=dataset, comparison='hidden minus ' + base,
                        metric=metric, difference=point['hidden'][metric] - point[base][metric],
                        ci95=np.quantile(vals, [.025, .975]).tolist(), replicates=len(vals), material_groups=len(groups),
                        paired=True, bootstrap_unit='material_group', dataset_stratified=True))
            status('paired_predictive_bootstrap', dataset=dataset, target=target)
    save('paired-predictive-ci.json', results)
    return results


def replay(events, samples, scores):
    details, summary = [], []
    configs = [('terminal', None), ('every4', None)] + [(m, t) for m in MODELS[:3] for t in [0., .1, .25, .5, .75, .9, 1.]]
    for stop in [.85, .70]:
        for controller, tau in configs:
            decisions = []
            for s in samples:
                if s['split'] != 'cal':
                    continue
                reads, trigger_calls = 0, 0
                stop_gid = None
                if controller != 'terminal':
                    for gid in s['points']:
                        ev = events[gid]
                        assert ev['ordinal'] is not None
                        if ev['ordinal'] % 4:
                            continue
                        if controller != 'every4':
                            trigger_calls += 1
                            if scores[controller][gid] < tau:
                                continue
                        reads += 1
                        if ev['certainty'] >= stop:
                            stop_gid = gid
                            break
                if stop_gid is None:
                    reads += 1
                    pred = int(s['terminal_probabilities'].argmax())
                    tokens = s['terminal_tokens']
                else:
                    pred = events[stop_gid]['pred']
                    tokens = events[stop_gid]['observable_after_tokens']
                decisions.append(dict(sample_id=s['sample_id'], component_id=s['component_id'], dataset=s['dataset'],
                    stopping_entropy=stop, controller=controller, trigger_threshold=tau, reads=reads,
                    trigger_calls=trigger_calls, tokens=tokens, correct=pred == s['gold'],
                    early=stop_gid is not None, terminal_correct=s['terminal_correct']))
            details.extend(decisions)
            for dataset in ['all', 'proofwriter', 'logiqa2']:
                chosen = [r for r in decisions if dataset == 'all' or r['dataset'] == dataset]
                summary.append(dict(stopping_entropy=stop, controller=controller, trigger_threshold=tau,
                    dataset=dataset, n=len(chosen), correct=sum(r['correct'] for r in chosen),
                    accuracy=float(np.mean([r['correct'] for r in chosen])),
                    mean_reads=float(np.mean([r['reads'] for r in chosen])),
                    mean_trigger_calls=float(np.mean([r['trigger_calls'] for r in chosen])),
                    mean_tokens=float(np.mean([r['tokens'] for r in chosen])), early=sum(r['early'] for r in chosen),
                    gained=sum(r['correct'] and not r['terminal_correct'] for r in chosen),
                    lost=sum(not r['correct'] and r['terminal_correct'] for r in chosen)))
    save('replay-curve.json', summary)
    with (OUT / 'replay-decisions.jsonl').open('w') as f:
        for r in details:
            f.write(json.dumps(r) + '\n')
    rng = np.random.default_rng(SEED + 1)
    ci = []
    for stop in [.85, .70]:
        for dataset in ['all', 'proofwriter', 'logiqa2']:
            selected = [r for r in details if r['stopping_entropy'] == stop and (dataset == 'all' or r['dataset'] == dataset)]
            question_ids = sorted({r['sample_id'] for r in selected})
            lookup = {(r['controller'], r['trigger_threshold'], r['sample_id']): r for r in selected}
            reference = [lookup['hidden', .25, q] for q in question_ids]
            groups = sorted({r['component_id'] for r in reference})
            gi = {g: i for i, g in enumerate(groups)}
            qg = np.array([gi[r['component_id']] for r in reference])
            strata = [np.array([gi[g] for g in groups if any(r['dataset'] == d and r['component_id'] == g for r in reference)]) for d in ['proofwriter', 'logiqa2']]
            strata = [s for s in strata if len(s)]
            for base, tau in [('every4', None), ('position_lr', .25), ('position_lookup', .25)]:
                other = [lookup[base, tau, q] for q in question_ids]
                delta = np.array([[float(a[k]) - float(b[k]) for k in ['correct', 'reads', 'tokens']]
                                  for a, b in zip(reference, other)])
                boot = []
                for repeat in range(1000):
                    counts = np.zeros(len(groups), dtype=int)
                    for stratum in strata:
                        counts += np.bincount(rng.choice(stratum, size=len(stratum), replace=True), minlength=len(groups))
                    boot.append(np.average(delta, axis=0, weights=counts[qg]))
                bounds = np.quantile(boot, [.025, .975], axis=0)
                for j, name in enumerate(['accuracy', 'mean_reads', 'mean_tokens']):
                    ci.append(dict(stopping_entropy=stop, dataset=dataset, comparison=f'hidden tau=.25 minus {base} tau={tau}',
                        metric=name, difference=float(delta[:, j].mean()), ci95=bounds[:, j].tolist(),
                        paired=True, replicates=1000, material_groups=len(groups)))
    save('paired-replay-ci.json', ci)
    return summary, ci


def write_report(summaries, conditioned, pred_ci, curves, replay_ci, elapsed):
    def tab(values, fields):
        lines = ['|' + '|'.join(fields) + '|', '|' + '|'.join(['---'] * len(fields)) + '|']
        for r in values:
            lines.append('|' + '|'.join(f'{r[k]:.6f}' if isinstance(r[k], float) else str(r[k]) for k in fields) + '|')
        return lines
    lines = ['# Frozen position-control CPU diagnostic', '',
        'Existing train1024/dev256 only. All models retain the original correctness + entropy-certainty ≥ .85 training target. Position logistic and the fixed 64-token lookup are independently fitted with the original training question weights; the hidden controller is the frozen taskless trigger. Development metrics assign equal total weight to each question.', '',
        'Primary .85 development target:', '']
    lines += tab([r for r in summaries if r['split'] == 'development' and r['target'] == .85], ['dataset', 'controller', 'questions', 'events', 'weighted_prevalence', 'logloss', 'auroc', 'ap'])
    lines += ['', 'Length-conditioned discrimination (64-token bands, question-normalized within band):', '']
    lines += tab([r for r in conditioned if r['target'] == .85], ['dataset', 'controller', 'valid_bands', 'pooled_within_band_auroc'])
    lines += ['', 'Paired material-group 95% bootstrap CIs, primary .85 target:', '']
    lines += tab([r for r in pred_ci if r['target'] == .85], ['dataset', 'comparison', 'metric', 'difference', 'ci95'])
    lines += ['', 'Fixed-trajectory replay at absolute sentence ordinal s mod4=0; all trigger variants share checkpoints. Main tau=.25 descriptive comparison:', '']
    lines += tab([r for r in curves if r['stopping_entropy'] == .85 and r['controller'] in ['terminal', 'every4'] or r['stopping_entropy'] == .85 and r['trigger_threshold'] == .25], ['dataset', 'controller', 'trigger_threshold', 'correct', 'accuracy', 'mean_reads', 'mean_tokens', 'early', 'gained', 'lost'])
    lines += ['', 'The complete threshold sweep and .70 supplementary diagnostics are in the JSON artifacts. The .70 target uses the same .85-trained scores; no controller was selected or refit using these diagnostics. Bootstrap samples whole material groups within dataset and keeps identical sampled groups for every compared controller.', '',
        'Scope: retrospective trigger discrimination and frozen-trajectory query/token outcomes on ProofWriter and LogiQA2. Sentence readouts were collected on RTX3080; terminal readouts came from A100. These CPU computations provide no online latency estimate, no same-device inference comparison, no ARC result, and no fresh confirmation claim. Replay may expose a changed answer at a saved checkpoint but cannot establish the trajectory that a new online system would generate.', '',
        f'Execution wall time: {elapsed:.2f} seconds. The only new source and outputs are in this isolated diagnostic directory. Existing training, protocols, results, controllers, and online code were preserved.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    with (OUT / 'development-primary-metrics.csv').open('w', newline='') as f:
        chosen = [r for r in summaries if r['split'] == 'development' and r['target'] == .85]
        writer = csv.DictWriter(f, fieldnames=list(chosen[0]))
        writer.writeheader()
        writer.writerows(chosen)


def main():
    started = time.monotonic()
    protocol = read(OUT / 'protocol.json')
    for name, digest in protocol['source_bindings'].items():
        assert sha(PROJECT / name) == digest, name
    protected = [BASE / 'sentence-policies' / n for n in ['data-audit.json', 'status.json', 'protocol.json', 'trigger-models.joblib']]
    protected += [COLLECT / n for n in ['work.json', 'feature-index.json', 'status.json', 'protocol.json', 'completion.json']]
    before = {str(p): dict(sha256=sha(p), mtime_ns=p.stat().st_mtime_ns) for p in protected if p.exists()}
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        events, samples = load_data()
        status('fitting_and_predictive_evaluation')
        train, certainty, correct, qw, scores, summaries, conditioned = fit_and_evaluate(events)
        status('predictive_results_ready')
        pred_ci = paired_bootstrap(events, train, certainty, correct, qw, scores)
        status('replay')
        curves, replay_ci = replay(events, samples, scores)
    for p, previous in before.items():
        assert sha(Path(p)) == previous['sha256'] and Path(p).stat().st_mtime_ns == previous['mtime_ns'], p
    for name, digest in protocol['source_bindings'].items():
        assert sha(PROJECT / name) == digest, name
    elapsed = time.monotonic() - started
    save('preservation-audit.json', dict(protected_artifacts=before, unchanged=True, protocol_sources_unchanged=True,
                                       legacy_loader_imported=False, CUDA_initialized=torch.cuda.is_initialized()))
    write_report(summaries, conditioned, pred_ci, curves, replay_ci, elapsed)
    save('completion.json', dict(state='completed', elapsed_wall_seconds=elapsed, train_questions=1024, dev_questions=256,
        GPU=False, source_sha256=sha(Path(__file__)), protocol_sha256=sha(OUT / 'protocol.json'),
        source_artifacts_preserved=True, fresh_access=False, ARC_access=False, python=sys.version,
        numpy=np.__version__, torch=torch.__version__, sklearn=sklearn.__version__))
    status('completed', elapsed_wall_seconds=elapsed)


if __name__ == '__main__':
    main()
