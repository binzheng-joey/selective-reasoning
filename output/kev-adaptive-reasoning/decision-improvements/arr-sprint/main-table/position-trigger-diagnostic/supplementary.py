"""Post-primary read-only exploration: checkpoint and within-question controls."""
import time
import json
from collections import Counter
import diagnostic as d

np = d.np
OUT = d.OUT


def strata_weights(records, rng):
    groups = sorted({r['component_id'] for r in records})
    gi = {g: i for i, g in enumerate(groups)}
    assignment = np.array([gi[r['component_id']] for r in records])
    strata = [np.array([gi[g] for g in groups if any(r['component_id'] == g and r['dataset'] == ds for r in records)])
              for ds in ['proofwriter', 'logiqa2']]
    strata = [s for s in strata if len(s)]
    assert sum(map(len, strata)) == len(groups)
    draws = []
    for _ in range(1000):
        multiplicity = np.zeros(len(groups), dtype=int)
        for s in strata:
            multiplicity += np.bincount(rng.choice(s, size=len(s), replace=True), minlength=len(groups))
        draws.append(multiplicity[assignment])
    return np.array(draws)


def main():
    started = time.monotonic()
    assert d.read(OUT / 'completion.json')['state'] == 'completed'
    d.status = lambda phase, **kw: print(dict(phase=phase, **kw), flush=True)
    original_save = d.save
    # Load cache only. Preserve primary audit/status and all previous results.
    d.save = lambda name, value: None
    d.torch.set_num_threads(2)
    with d.threadpool_limits(limits=2):
        events, samples = d.load_data()
    d.save = original_save
    train = np.array([r['train'] for r in events])
    checkpoint = np.array([not r['train'] and r['ordinal'] % 4 == 0 for r in events])
    dataset_array = np.array([r['dataset'] for r in events])
    pos = np.array([r['prefix_tokens'] / 1024 for r in events])
    bins = np.minimum((pos * 16).astype(int), 15)
    fit = d.read(OUT / 'fitted-baselines.json')
    lr = d.joblib.load(OUT / 'position-lr.joblib')
    scores = dict(hidden=np.array([r['hidden'] for r in events]),
        position_lr=lr.predict_proba(pos[:, None])[:, 1],
        position_lookup=np.array(fit['lookup_values'])[bins], constant=np.full(len(events), fit['constant']))
    certainty = np.array([r['certainty'] for r in events])
    correct = np.array([r['correct'] for r in events])
    checkpoint_counts = Counter(r['sample_id'] for r, k in zip(events, checkpoint) if k)
    checkpoint_qw = np.array([1 / checkpoint_counts[r['sample_id']] if k else 0 for r, k in zip(events, checkpoint)])
    result, within, within_ci = [], [], []
    dev_samples = [s for s in samples if s['split'] == 'cal']
    for target in [.85, .70]:
        y = (correct & (certainty >= target)).astype(int)
        for dataset in ['all', 'proofwriter', 'logiqa2']:
            mask = checkpoint & ((dataset_array == dataset) if dataset != 'all' else True)
            for name in d.MODELS:
                result.append(dict(target=target, dataset=dataset, controller=name,
                    questions=len({r['sample_id'] for r, k in zip(events, mask) if k}), events=int(mask.sum()),
                    weighted_prevalence=float(np.average(y[mask], weights=checkpoint_qw[mask])),
                    **d.metrics(y[mask], scores[name][mask], checkpoint_qw[mask])))
        for subset in ['all_events', 'modulo4']:
            for dataset in ['all', 'proofwriter', 'logiqa2']:
                chosen = [s for s in dev_samples if dataset == 'all' or s['dataset'] == dataset]
                question_rows = []
                for s in chosen:
                    idx = np.array([i for i in s['points'] if subset == 'all_events' or checkpoint[i]])
                    if not len(idx) or len(np.unique(y[idx])) < 2:
                        continue
                    q = dict(sample_id=s['sample_id'], component_id=s['component_id'], dataset=s['dataset'])
                    for name in d.MODELS:
                        q[name] = float(d.roc_auc_score(y[idx], scores[name][idx]))
                    question_rows.append(q)
                if not question_rows:
                    continue
                for name in d.MODELS:
                    within.append(dict(target=target, subset=subset, dataset=dataset, controller=name,
                        eligible_questions=len(question_rows), total_questions=len(chosen),
                        coverage=len(question_rows) / len(chosen),
                        mean_within_question_auroc=float(np.mean([q[name] for q in question_rows]))))
                weights = strata_weights(question_rows, np.random.default_rng(d.SEED + 2))
                for base in ['position_lr', 'position_lookup']:
                    delta = np.array([q['hidden'] - q[base] for q in question_rows])
                    boot = (weights @ delta) / weights.sum(axis=1)
                    within_ci.append(dict(target=target, subset=subset, dataset=dataset,
                        comparison='hidden minus ' + base, mean_auroc_difference=float(delta.mean()),
                        ci95=np.quantile(boot, [.025, .975]).tolist(), eligible_questions=len(question_rows),
                        total_questions=len(chosen), replicates=1000, paired=True, bootstrap_unit='material_group'))
    d.save('checkpoint-predictive-metrics.json', result)
    d.save('within-question-auroc.json', within)
    d.save('within-question-paired-ci.json', within_ci)
    print('Checkpoint and within-question metrics ready', flush=True)
    # Match budgets exactly within each question; full future trajectory is used.
    budget_results = []
    for target in [.85, .70]:
        y = (correct & (certainty >= target)).astype(int)
        for fraction in [.1, .25, .5, .75, 1.]:
            question_values = []
            for s in dev_samples:
                idx = np.array([i for i in s['points'] if checkpoint[i]])
                n = len(idx)
                k = int(np.ceil(fraction * n))
                for name in d.MODELS:
                    # Stable sort resolves tied scores by the earlier absolute sentence.
                    selected = idx[np.argsort(-scores[name][idx], kind='stable')[:k]]
                    positives = int(y[idx].sum())
                    hits = int(y[selected].sum())
                    question_values.append(dict(dataset=s['dataset'], controller=name, checkpoints=n, budget=k,
                        positives=positives, hits=hits, recall=hits / positives if positives else None,
                        precision=hits / k if k else None, any_hit=hits > 0))
            for dataset in ['all', 'proofwriter', 'logiqa2']:
                for name in d.MODELS:
                    chosen = [q for q in question_values if q['controller'] == name and (dataset == 'all' or q['dataset'] == dataset)]
                    eligible = [q for q in chosen if q['positives']]
                    budget_results.append(dict(target=target, dataset=dataset, controller=name, budget_fraction=fraction,
                        questions=len(chosen), positive_questions=len(eligible),
                        mean_checkpoint_budget=float(np.mean([q['budget'] for q in chosen])),
                        target_questions_hit=sum(q['any_hit'] for q in chosen),
                        positive_question_hit_rate=float(np.mean([q['any_hit'] for q in eligible])),
                        mean_recall_positive_questions=float(np.mean([q['recall'] for q in eligible])),
                        mean_precision=float(np.mean([q['precision'] for q in chosen if q['precision'] is not None])),
                        full_future_trajectory_used=True, deployable=False))
    d.save('retrospective-matched-budget-ranking.json', budget_results)
    # Preserve primary bootstrap output; evaluate only the uniform modulo-four candidates.
    d.save = lambda name, value: original_save('checkpoint-' + name, value)
    with d.threadpool_limits(limits=2):
        checkpoint_ci = d.paired_bootstrap(events, ~checkpoint, certainty, correct, checkpoint_qw, scores)
    d.save = original_save
    lines = ['# Additional exploratory controls', '',
        'Added after the primary CPU analysis. No model refitting or selection. Every value uses the same .85-trained hidden/position controllers. Primary protocol and results are preserved.', '',
        '|Subset|Dataset|Controller|Eligible/total questions|Mean within-question AUROC|',
        '|---|---|---|---|---|']
    for r in within:
        if r['target'] == .85:
            lines.append(f"|{r['subset']}|{r['dataset']}|{r['controller']}|{r['eligible_questions']}/{r['total_questions']}|{r['mean_within_question_auroc']:.6f}|")
    lines += ['', 'A question is eligible only if its target events include both positive and negative labels. This statistic compares sentence ranking within the same question, reducing the role of fixed question difficulty. Paired material-group confidence intervals are supplied in within-question-paired-ci.json.', '',
        'checkpoint-predictive-metrics.json and checkpoint-paired-predictive-ci.json restrict evaluation to absolute sentence ordinal mod4=0, with equal total weight per question after restriction.', '',
        'retrospective-matched-budget-ranking.json uses whole-trajectory scores to retain the same ceil(fraction × checkpoint count) candidates per question. This is a non-deployable ranking diagnostic that requires future states; it does not define an online policy and is excluded from the main result.', '',
        'The same original RTX3080 sentence / A100 terminal, two-task, fixed-trajectory scope applies. No fresh or ARC data were accessed.', '']
    (OUT / 'supplementary-report.md').write_text('\n'.join(lines))
    d.save('supplementary-completion.json', dict(state='completed', elapsed_wall_seconds=time.monotonic() - started,
        primary_results_preserved=True, source_sha256=d.sha(d.Path(__file__)), GPU=False, fresh_access=False, ARC_access=False,
        checkpoint_questions=len(checkpoint_counts), checkpoint_events=int(checkpoint.sum())))
    print('Supplement completed', flush=True)


if __name__ == '__main__':
    main()
