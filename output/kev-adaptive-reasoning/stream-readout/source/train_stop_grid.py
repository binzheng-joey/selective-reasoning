"""CPU cross-fitted stopping pilot on existing answer-position features.

All questions enter reasoning. The 128/256/512 grid isolates stopping from
sentence scheduling and initial routing. No new generation or online claims.
"""
import collections
import datetime
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
OUT = ROOT / 'stopping-grid'
THRESHOLDS = [.50, .60, .70, .80, .85, .90, .95, .98, .99]
PRICES = [0., .025, .05, .10, .20]
GRID = [0, 128, 256, 512, 1024]
FEATURES = ['p0', 'p1', 'p2', 'p3', 'max_probability', 'margin', 'normalized_entropy',
            'configured_budget_fraction', 'candidate_count_div4', 'previous_max_probability',
            'confidence_change', 'same_answer_as_previous_read', 'previous_read_exists']


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    path = OUT / name
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def vector(ps, budget, count, previous):
    top = sorted(ps, reverse=True)
    confidence = top[0]
    entropy = -sum(p * math.log(p) for p in ps if p > 0) / math.log(count)
    pred = max(range(4), key=ps.__getitem__)
    before = max(previous) if previous is not None else 0.
    same = int(previous is not None and pred == max(range(4), key=previous.__getitem__))
    return list(ps) + [confidence, top[0] - top[1], entropy, budget / 1024, count / 4,
                       before, confidence - before if previous is not None else 0., same, int(previous is not None)]


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only pilot'
    import numpy as np
    import torch
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import joblib
    torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    assert not (OUT / 'completion.json').exists(), 'Pilot already completed'
    start = time.monotonic()
    samples = rows(ROOT / 'samples.jsonl')
    train = [r for r in samples if r['split'] == 'train']
    dev = [r for r in samples if r['split'] == 'cal']
    folds = {r['sample_id']: r['held_out_fold'] for r in rows(ROOT / 'stopping-plan/oof-folds.jsonl')}
    chosen = read(ROOT / 'training/selected.json')['answer_position_linear-s29']
    assert chosen['lr'] == .0003 and chosen['epoch'] == 20
    dependencies = [Path(__file__), ROOT / 'samples.jsonl', ROOT / 'labels-observed.jsonl',
                    ROOT / 'stopping-plan/oof-folds.jsonl', ROOT / 'training/answer_position_linear-s29.pt',
                    ROOT / 'training/selected.json']
    protocol = dict(UTC=now(), state='frozen_before_stopping_fit', role='Development pilot, not independent confirmation',
        reader='answer_position_linear-s29, fixed completed study weights for dev; four cross-fitted heads for train',
        selection_reason='Natural-state readouts lost substantial accuracy in completed matched comparison; retain answer-position interface.',
        n_train=1024, n_observed_dev=256, seed=29, head_learning_rate=.0003, head_epochs=20, head_batch=16,
        oof='Four material-group folds; head fitting excludes each evaluated training group. Hyperparameters inherited from prior observed-dev selection.',
        stop_model='StandardScaler + L2 logistic regression, C=1, lbfgs, max_iter=2000; one binary-current and one joint-current/terminal model',
        labels='Joint label = 2*current_correct + terminal_correct. Continue means frozen reader at cap/EOS, not best future point.',
        sample_weight='Each eligible training question contributes total weight one over its intermediate positions',
        features=FEATURES, checkpoints=[128, 256, 512], terminal_cap=1024,
        initial_routing='All questions enter reasoning; no zero-token reads or initial exits',
        scheduler='Fixed token grid; all earlier grid reads were actually performed before a later decision',
        arms=['raw_confidence', 'raw_margin', 'one_minus_entropy', 'calibrated_current', 'current_and_continuation_value'],
        confidence_or_certainty_thresholds=THRESHOLDS, token_utility_prices=PRICES,
        stop_rule='For joint model, P(current correct)>=threshold AND P(wrong->right)-P(right->wrong)<=price*(1-t/1024)',
        risk_scope='No calibrated correctness guarantee; prices are utility per configured token budget, not measured latency or GPU time.',
        cost_scope='Ideal generation tokens and performed read counts only; no branch/readback/cache/latency cost inferred',
        source_sha256={str(p): sha(p) for p in dependencies},
        protected='Do not read fresh512 inputs/gold or parent confirmation inputs/gold; no GPU or generator changes')
    write('protocol.json', protocol)
    write('status.json', dict(state='loading_existing_features', UTC=now()))
    label = {r['sample_id']: r['gold'] for r in rows(ROOT / 'labels-observed.jsonl')}
    xs = []
    for r in train:
        path = PROJECT / r['answer_feature']
        assert sha(path) == r['answer_feature_sha256']
        value = torch.load(path, map_location='cpu', weights_only=True)
        assert value['sample_id'] == r['sample_id'] and tuple(value['states'].shape) == (5, 2560)
        xs.append(value['states'].float())
    x = torch.stack(xs)
    counts = torch.tensor([r['candidate_count'] for r in train])
    labels = torch.tensor([label[r['sample_id']] for r in train])
    oof = torch.empty((1024, 5, 4))
    fold_runs = []
    for fold in range(4):
        assert time.monotonic() - start < 1800
        fit = [i for i, r in enumerate(train) if folds[r['sample_id']] != fold]
        held = [i for i, r in enumerate(train) if folds[r['sample_id']] == fold]
        assert len(fit) == 768 and len(held) == 256
        assert not {train[i]['component_id'] for i in fit} & {train[i]['component_id'] for i in held}
        torch.manual_seed(29)
        head = torch.nn.Linear(2560, 4, bias=False)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.0003, weight_decay=.01)
        generator = torch.Generator().manual_seed(20261009 + 29 + fold)
        mask = torch.arange(4)[None, None, :] >= counts[:, None, None]
        for epoch in range(20):
            order = [fit[j] for j in torch.randperm(len(fit), generator=generator).tolist()]
            for begin in range(0, len(order), 16):
                ix = order[begin:begin+16]
                logits = head(x[ix]).masked_fill(mask[ix], -torch.inf)
                loss = F.cross_entropy(logits.reshape(-1, 4), labels[ix, None].expand(-1, 5).reshape(-1))
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); optimizer.step()
        with torch.no_grad():
            ps = head(x[held]).masked_fill(mask[held], -torch.inf).softmax(-1)
            oof[held] = ps
        torch.save(dict(fold=fold, fit_sample_ids=[train[i]['sample_id'] for i in fit],
                        held_sample_ids=[train[i]['sample_id'] for i in held], state_dict=head.state_dict()),
                   OUT / f'oof-reader-{fold}.pt')
        fold_runs.append(dict(fold=fold, terminal_correct=int((ps[:, -1].argmax(-1) == labels[held]).sum()), n=256))
        write('status.json', dict(state='cross_fitting_reader', UTC=now(), completed_folds=fold+1, total_folds=4))
    assert torch.isfinite(oof).all()
    torch.save(dict(sample_ids=[r['sample_id'] for r in train], probabilities=oof), OUT / 'oof-probabilities.pt')
    saved = torch.load(ROOT / 'training/answer_position_linear-s29.pt', map_location='cpu', weights_only=True)
    assert saved['sample_ids'] == [r['sample_id'] for r in dev]

    def events(manifest, probabilities):
        output = []
        for r, ps in zip(manifest, probabilities):
            g = label[r['sample_id']]; end = min(1024, r['generated_tokens'])
            terminal = max(range(4), key=ps[-1].__getitem__) == g
            valid = [j for j in [1, 2, 3] if GRID[j] < end]
            previous = None
            for j in valid:
                pred = max(range(4), key=ps[j].__getitem__)
                output.append(dict(sample_id=r['sample_id'], budget=GRID[j],
                    x=vector(ps[j], GRID[j], r['candidate_count'], previous),
                    current_correct=pred == g, terminal_correct=terminal,
                    joint_label=2*int(pred == g)+int(terminal), weight=1/len(valid)))
                previous = ps[j]
        return output

    training = events(train, oof.tolist())
    evaluation = events(dev, saved['probabilities'].tolist())
    tx = np.array([r['x'] for r in training]); dx = np.array([r['x'] for r in evaluation])
    scaler = StandardScaler().fit(tx)
    weight = np.array([r['weight'] for r in training])
    models = {}
    for name, target in [('current', 'current_correct'), ('joint', 'joint_label')]:
        model = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, random_state=29)
        model.fit(scaler.transform(tx), np.array([r[target] for r in training]), sample_weight=weight)
        assert int(model.n_iter_.max()) < 2000, 'Stop classifier failed to converge'
        models[name] = model
    joblib.dump(dict(scaler=scaler, models=models, features=FEATURES), OUT / 'stop-models.joblib')
    current = models['current'].predict_proba(scaler.transform(dx))[:, list(models['current'].classes_).index(True)]
    joint = models['joint'].predict_proba(scaler.transform(dx))
    q = np.zeros((len(evaluation), 4))
    for j, label_class in enumerate(models['joint'].classes_): q[:, int(label_class)] = joint[:, j]
    predicted = collections.defaultdict(list)
    for i, r in enumerate(evaluation):
        r.update(calibrated_correct=float(current[i]), joint_current_correct=float(q[i, 2]+q[i, 3]),
                 continuation_delta=float(q[i, 1]-q[i, 2]), lost_correct_probability=float(q[i, 1]))
        predicted[r['sample_id']].append(r)
    baselines = {}
    for r, ps in zip(dev, saved['probabilities'].tolist()):
        baselines[r['sample_id']] = dict(dataset=r['dataset'], correct=max(range(4), key=ps[-1].__getitem__) == label[r['sample_id']],
                                        tokens=min(1024, r['generated_tokens']))
    assert sum(r['correct'] for r in baselines.values()) == 203
    configurations = [(name, t, None) for name in ['raw_confidence', 'raw_margin', 'one_minus_entropy', 'calibrated_current'] for t in THRESHOLDS]
    configurations += [('current_and_continuation_value', t, price) for t in THRESHOLDS for price in PRICES]
    curve = []
    decisions = []
    for name, threshold, price in configurations:
        results = []
        for sid, baseline in baselines.items():
            used, exit_at = 0, None
            for ev in predicted[sid]:
                used += 1
                score = {'raw_confidence': ev['x'][4], 'raw_margin': ev['x'][5],
                         'one_minus_entropy': 1-ev['x'][6], 'calibrated_current': ev['calibrated_correct'],
                         'current_and_continuation_value': ev['joint_current_correct']}[name]
                allowed = score >= threshold
                if name == 'current_and_continuation_value':
                    allowed = allowed and ev['continuation_delta'] <= price*(1-ev['budget']/1024)
                if allowed:
                    exit_at = ev; break
            result = dict(sample_id=sid, dataset=baseline['dataset'],
                          correct=exit_at['current_correct'] if exit_at else baseline['correct'],
                          terminal_correct=baseline['correct'], early=exit_at is not None,
                          tokens=exit_at['budget'] if exit_at else baseline['tokens'], reads=used+(exit_at is None))
            results.append(result)
            decisions.append(dict(arm=name, threshold=threshold, price=price, **result))
        for dataset in ['all', 'proofwriter', 'logiqa2']:
            subset = [r for r in results if dataset == 'all' or r['dataset'] == dataset]
            curve.append(dict(arm=name, threshold=threshold, price=price, dataset=dataset, n=len(subset),
                correct=sum(r['correct'] for r in subset), terminal_correct=sum(r['terminal_correct'] for r in subset),
                gained_correct=sum(r['correct'] and not r['terminal_correct'] for r in subset),
                lost_correct=sum(not r['correct'] and r['terminal_correct'] for r in subset),
                early_exits=sum(r['early'] for r in subset), ideal_mean_generated_tokens=statistics.mean(r['tokens'] for r in subset),
                mean_reads=statistics.mean(r['reads'] for r in subset)))
    write('analysis.json', dict(UTC=now(), scope=protocol['role'], folds=fold_runs,
          terminal_only=dict(correct=203, n=256, mean_tokens=statistics.mean(r['tokens'] for r in baselines.values()), mean_reads=1),
          train_events=len(training), dev_events=len(evaluation), curves=curve,
          claims='Stopping pilot on token checkpoints; no sentence scheduling, initial routing, online speedup, or independent-test claim'))
    (OUT / 'dev-events.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in evaluation))
    (OUT / 'policy-decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decisions))
    lines = ['# 答案位置读取器的折外停止训练', '',
             '原训练1024题做材料组四折读取器训练，折外预测用于训练停止模型；原已观察开发256题用于展示曲线。底座、生成文本和开发集读取头保持不变。',
             '全部题进入推理，检查点固定为128/256/512 token。尚未验证句末调度、初始分流或在线耗时。', '',
             '末尾只读一次：203/256正确，平均读取1次。以下token是假设读取零耗时的理想生成量，不能当作实际资源节省。', '',
             '|策略|阈值|token收益价格|正确/256|新增正确|损失正确|提前退出|平均读取|理想生成token|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in curve:
        if r['dataset'] != 'all' or r['threshold'] not in [.90, .95]: continue
        if r['arm'] in ['raw_margin', 'one_minus_entropy']: continue
        if r['price'] not in [None, 0., .05]: continue
        price = '-' if r['price'] is None else str(r['price'])
        lines.append(f"|{r['arm']}|{r['threshold']}|{price}|{r['correct']}|{r['gained_correct']}|{r['lost_correct']}|{r['early_exits']}|{r['mean_reads']:.2f}|{r['ideal_mean_generated_tokens']:.1f}|")
    lines += ['', '完整置信度、margin、entropy、校准正确性与继续收益曲线见analysis.json。上表展示事前规定的0.90/0.95阈值和0/0.05价格，未按开发gold挑选“最好”点。',
              '继续收益以共同预算终点正确性减当前正确性定义；“预测收益低”还需满足当前正确性门限，不能自动当作可靠答案。',
              '每题训练权重归一，不把多个前缀当成独立题。折外读取减轻训练题读数过乐观问题，但仍存在768题读取头与完整1024题读取头之间的分布差异。',
              '开发集曾用于读取头选型，后续改进需要新的独立确认。当前结果不提供准确率不下降保证，也不代表在线缓存已经实现。']
    (OUT / 'report.md').write_text('\n'.join(lines)+'\n')
    for name, digest in protocol['source_sha256'].items(): assert sha(name) == digest, name
    done = dict(state='completed', UTC=now(), wall_seconds=time.monotonic()-start, GPU_used=False,
                reader_crossfit_models=4, stop_models=2, new_generated_tokens=0, new_gold_decodes=0)
    write('completion.json', done); write('status.json', done)
    print(json.dumps(done))


if __name__ == '__main__':
    main()
