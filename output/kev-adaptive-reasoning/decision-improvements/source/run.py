"""CPU-only component experiments. Existing development data; no GPU/test access."""
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import joblib
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
GATE = ROOT.parent / 'prethink-hidden-gate'
STREAM = ROOT.parent / 'stream-readout'
GRID = np.array([0, 128, 256, 512, 1024])
CS = [.001, .01, .1, 1.]
PRICES = [0., .025, .05, .1, .2]
THRESHOLDS = [.5, .6, .7, .8, .85, .9, .95, .98, .99]
SEED = 20261009


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def status(phase, **kw):
    value = dict(UTC=now(), phase=phase, pid=os.getpid(), **kw)
    write('status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fit_lr(x, y, C=1., weight=None):
    assert len(np.unique(y)) > 1
    m = LogisticRegression(C=C, solver='lbfgs', max_iter=3000, tol=1e-7, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        m.fit(x, y, sample_weight=weight)
    return m


def four_probs(model, x):
    q = np.zeros((len(x), 4))
    for j, c in enumerate(model.classes_):
        q[:, int(c)] = model.predict_proba(x)[:, j]
    return q


def split_audit(groups):
    for i, (a, ga) in enumerate(groups.items()):
        for b, gb in list(groups.items())[i+1:]:
            assert not (ga & gb), (a, b, 'shared material group')


def route_metrics(mask, data, costs):
    initial = np.array([r['initial_correct'] for r in data], dtype=int)
    terminal = np.array([r['reasoning_correct'] for r in data], dtype=int)
    delta = terminal - initial
    return dict(n=len(data), correct=int(np.where(mask, terminal, initial).sum()),
                calls=int(mask.sum()), gain=int(((delta == 1) & mask).sum()),
                harm=int(((delta == -1) & mask).sum()),
                generated_tokens=int(sum(r['reasoning_tokens'] for r, yes in zip(data, mask) if yes)),
                replay_incremental_seconds=float(np.mean(costs * mask)))


def run_gate():
    status('gate_loading', state='running')
    data = rows(GATE / 'paired-targets-cpu.jsonl')
    inputs = rows(GATE / 'initial-inputs.jsonl')
    assert [r['key'] for r in data] == [r['key'] for r in inputs]
    h, four, hashes = [], [], {}
    for r, inp in zip(data, inputs):
        f = GATE / 'features' / (r['key'] + '.npz')
        with np.load(f) as a:
            h.append(a['hidden'])
        hashes[str(f.relative_to(PROJECT))] = sha(f)
        p = np.array(inp['reference_probabilities'])
        z = np.sort(p)
        four.append([z[-1], z[-1]-z[-2], -(p*np.log(np.maximum(p, 1e-30))).sum()/np.log(len(p)),
                     np.log1p(len(inp['encoding']['ids']))])
    x = np.column_stack([four, h])
    ix = {s: np.array([i for i, r in enumerate(data) if r['split'] == s]) for s in ['train', 'cal', 'eval']}
    split_audit({s: {data[i]['component_id'] for i in ids} for s, ids in ix.items()})
    tr, ca = ix['train'], ix['cal']
    scaler = StandardScaler().fit(x[tr])
    z = scaler.transform(x)
    a = np.array([r['initial_correct'] for r in data], dtype=int)
    b = np.array([r['reasoning_correct'] for r in data], dtype=int)
    joint = 2*a+b
    gain, harm = (b-a == 1).astype(int), (b-a == -1).astype(int)
    chosen, scores, selections = {}, {}, []
    for family in ['two_binary', 'joint_four']:
        candidates = []
        for C in CS:
            status('gate_fit', family=family, C=C, state='running')
            if family == 'two_binary':
                models = [fit_lr(z[tr], target[tr], C) for target in [gain, harm]]
                pg, ph = [m.predict_proba(z)[:, 1] for m in models]
            else:
                models = [fit_lr(z[tr], joint[tr], C)]
                q = four_probs(models[0], z)
                pg, ph = q[:, 1], q[:, 2]
            loss = float(((pg[ca]-gain[ca])**2+(ph[ca]-harm[ca])**2).mean()/2)
            candidates.append((loss, C, models, pg-ph))
            selections.append(dict(family=family, C=C, calibration_Brier=loss))
        loss, C, models, score = min(candidates, key=lambda v:(v[0], v[1]))
        chosen[family] = dict(C=C, Brier=loss, models=models)
        scores[family] = score
    costs = np.array([r['reasoning_seconds'] + r['readout_seconds'] for r in data])
    target = np.log1p(costs)
    cost_candidates = []
    for alpha in [1., 10., 100.]:
        m = Ridge(alpha=alpha, solver='lsqr', tol=1e-6, max_iter=3000).fit(z[tr], target[tr])
        loss = float(np.mean((m.predict(z[ca])-target[ca])**2))
        cost_candidates.append((loss, alpha, m))
    loss, alpha, cost_model = min(cost_candidates, key=lambda v:(v[0],v[1]))
    predicted_cost = np.expm1(np.clip(cost_model.predict(z), 0., np.log1p(max(costs[tr])*10)))
    norm = float(costs[tr].mean())
    rules = {}
    for family, benefit in scores.items():
        for price in PRICES:
            score = benefit-price*predicted_cost/norm
            name = family + '_cost' + str(price)
            rules[name] = dict(score=score, family=family, price=price)
    for name, score in [('confidence', -np.array(four)[:,0]), ('margin', -np.array(four)[:,1]), ('entropy', np.array(four)[:,2])]:
        rules[name] = dict(score=score, family=name, price=None)
    thresholds = {}
    for name, item in rules.items():
        ranked = np.sort(item['score'][ca])[::-1]
        thresholds[name] = {str(f): (None if f in [0., 1.] else float((ranked[int(f*len(ca))-1]+ranked[int(f*len(ca))])/2))
                            for f in [0., .25, .5, .75, 1.]}
    write('gate/selection.json', dict(UTC=now(), selected={k:{'C':v['C'],'Brier':v['Brier']} for k,v in chosen.items()},
          candidates=selections, cost_alpha=alpha, cost_log_MSE=loss, thresholds=thresholds,
          cost_normalizer_seconds=norm, observed_eval_not_used_to_select=True))
    joblib.dump(dict(scaler=scaler, classifiers=chosen, cost_model=cost_model), ROOT/'gate/models.joblib')
    curves = []
    for split in ['cal', 'eval']:
        idx = ix[split]
        for name, item in rules.items():
            for fraction, threshold in thresholds[name].items():
                f = float(fraction)
                mask = np.ones(len(idx),dtype=bool) if f == 1 else (np.zeros(len(idx),dtype=bool) if f == 0 else item['score'][idx] > threshold)
                for dataset in ['all','proofwriter','logiqa2']:
                    take = np.array([j for j, i in enumerate(idx) if dataset=='all' or data[i]['dataset']==dataset])
                    metrics = route_metrics(mask[take], [data[idx[j]] for j in take], costs[idx[take]])
                    curves.append(dict(split=split, dataset=dataset, policy=name, fraction=f, **metrics))
    write('gate/analysis.json', dict(scope='Existing observed development; endpoint Kev, cap8192; historical costs only',
          curves=curves, cost_calibration_MAE_seconds=float(np.abs(predicted_cost[ca]-costs[ca]).mean()),
          n={s:len(ids) for s,ids in ix.items()}, train_joint_class_counts=np.bincount(joint[tr],minlength=4).tolist()))
    write('gate/feature-bindings.json', hashes)
    return curves


def stream_data():
    data = rows(STREAM/'samples.jsonl')
    labels = {r['sample_id']:r['gold'] for r in rows(STREAM/'labels-observed.jsonl')}
    train = [r for r in data if r['split']=='train']
    dev = [r for r in data if r['split']=='cal']
    split_audit({'train':{r['component_id'] for r in train},'dev':{r['component_id'] for r in dev}})
    oof = torch.load(STREAM/'stopping-grid/oof-probabilities.pt',map_location='cpu',weights_only=True)
    saved = torch.load(STREAM/'training/answer_position_linear-s29.pt',map_location='cpu',weights_only=True)
    assert oof['sample_ids']==[r['sample_id'] for r in train]
    assert saved['sample_ids']==[r['sample_id'] for r in dev]
    ps = np.concatenate([oof['probabilities'].numpy(),saved['probabilities'].numpy()])
    ordered = train+dev
    gold = np.array([labels[r['sample_id']] for r in ordered])
    correct = ps.argmax(-1)==gold[:,None]
    assert int(correct[len(train):,-1].sum())==203
    return ordered, ps, correct, len(train)


def certainty(p, signal):
    p = np.asarray(p)
    z = np.sort(p)
    if signal=='confidence':return float(z[-1])
    if signal=='margin':return float(z[-1]-z[-2])
    return float(1+np.sum(p[p>0]*np.log(p[p>0]))/np.log(len(p)))


def feature(p, j, count, previous):
    before = float(np.max(previous)) if previous is not None else 0.
    mx = float(np.max(p))
    return list(p)+[mx, certainty(p[:count],'margin'), 1-certainty(p[:count],'entropy'),
                   GRID[j]/1024, count/4, before, mx-before if previous is not None else 0.,
                   int(previous is not None and p.argmax()==previous.argmax()),int(previous is not None)]


def aggregate(decisions, samples):
    result = []
    for dataset in ['all','proofwriter','logiqa2']:
        chosen = [d for d,r in zip(decisions,samples) if dataset=='all' or r['dataset']==dataset]
        result.append(dict(dataset=dataset,n=len(chosen),correct=sum(d['correct'] for d in chosen),
            gained=sum(d['correct'] and not d['terminal_correct'] for d in chosen),
            lost=sum(not d['correct'] and d['terminal_correct'] for d in chosen),
            early=sum(d['tokens']<d.get('terminal_tokens',1024) for d in chosen),
            ideal_tokens=float(np.mean([d['tokens'] for d in chosen])),
            mean_reads=float(np.mean([d['reads'] for d in chosen])),
            mean_trigger_calls=float(np.mean([d.get('trigger_calls',0) for d in chosen]))))
    return result


def simulate_reads(observe, trigger, signal, threshold, schedule, trigger_threshold, end=1024):
    used=[];calls=0;next_j=1;last=None;lastlast=None
    for j in [j for j in [1,2,3] if GRID[j]<end]+[4]:
        perform = j==4
        if j<4:
            if schedule=='dense':perform=True
            elif schedule=='terminal':perform=False
            elif schedule=='fixed_128_512':perform=j in [1,3]
            elif schedule=='fixed_256_512':perform=j in [2,3]
            elif schedule=='feedback':perform=j==next_j
            elif schedule=='learned':
                calls+=1;perform=trigger(j)>=trigger_threshold
        if not perform:continue
        p=observe(j);used.append(j)
        s=certainty(p,signal)
        if j==4 or s>=threshold:
            return dict(at=j,pred=int(np.argmax(p)),reads=len(used),trigger_calls=calls,read_indices=used)
        if schedule=='feedback':
            stable = last is not None and np.argmax(p)==np.argmax(last) and abs(s-certainty(last,signal))<.03
            next_j = min(4,j+(2 if s<threshold-.1 or stable else 1))
        lastlast,last=last,p
    raise AssertionError('Missing terminal read')


def run_reading(samples, ps, correct, ntrain):
    status('read_trigger_features', state='running')
    natural=[]
    feature_index={r['index']:r for r in rows(STREAM/'features/index.jsonl')}
    for k,r in enumerate(samples):
        f=STREAM/'features'/f"{r['index']:04}.pt"
        assert sha(f)==feature_index[r['index']]['sha256']
        a=torch.load(f,map_location='cpu',weights_only=True)
        assert a['sample_id']==r['sample_id']
        natural.append(a['last'][1:4].float().numpy())
        del a
        if (k+1)%128==0:status('read_trigger_features', state='running',completed=k+1,total=len(samples))
    hidden=np.stack(natural).astype(np.float64); del natural
    flat=hidden.reshape(-1,hidden.shape[-1])
    valid=np.array([[GRID[j]<min(1024,r['generated_tokens']) for j in [1,2,3]] for r in samples])
    training_mask=valid.reshape(-1).copy();training_mask[ntrain*3:]=False
    weights=np.repeat(1/np.maximum(valid.sum(1),1),3)
    scaler=StandardScaler().fit(flat[training_mask])
    pca=PCA(n_components=64,svd_solver='randomized',random_state=SEED).fit(scaler.transform(flat[training_mask]))
    reduced=pca.transform(scaler.transform(flat))
    position=np.tile(GRID[1:4]/1024,len(samples))[:,None]
    x=np.column_stack([reduced,position])
    models={}; predictions={}; diagnostics={}
    for threshold in [.8,.85]:
        passed=np.array([[certainty(ps[i,j,:r['candidate_count']],'entropy')>=threshold for j in [1,2,3]] for i,r in enumerate(samples)])
        target=(passed & correct[:,1:4]).reshape(-1).astype(int)
        model=fit_lr(x[training_mask],target[training_mask],C=.01,weight=weights[training_mask])
        pred=model.predict_proba(x)[:,1].reshape(len(samples),3)
        # PCA+scaling+linear classifier can be folded into one small affine trigger.
        wh=(model.coef_[0,:64]@pca.components_)/scaler.scale_
        intercept=float(model.intercept_[0]-(model.coef_[0,:64]@pca.components_)@pca.mean_-wh@scaler.mean_)
        logits=flat@wh+model.coef_[0,64]*position[:,0]+intercept
        assert np.max(np.abs(logits-model.decision_function(x)))<1e-7
        models[str(threshold)]=dict(model=model,hidden_weight=wh,position_weight=float(model.coef_[0,64]),intercept=intercept)
        predictions[threshold]=pred
        diagnostics[str(threshold)]=dict(training_positive_rate=float(target[training_mask].mean()),
                                        training_events=int(training_mask.sum()),
                                        trigger_parameters=len(wh)+2,folded_logit_agreement=True)
    (ROOT/'reading').mkdir(exist_ok=True)
    joblib.dump(dict(scaler=scaler,pca=pca,models=models),ROOT/'reading/trigger-models.joblib')
    # Actual skipped expensive read probabilities are not exposed to the scheduler.
    for hidden_middle in [[.99,.01],[.01,.99]]:
        values={1:[.5,.5],2:hidden_middle,3:[.99,.01],4:[.1,.9]};seen=[]
        def see(j):seen.append(j);return values[j]
        out=simulate_reads(see,None,'confidence',.95,'feedback',None)
        assert seen==[1,3] and out['pred']==0
    curves=[];details=[]
    configurations=[('terminal','entropy',.85,None)]
    for sig,thr in [('entropy',.8),('entropy',.85),('confidence',.95),('margin',.9)]:
        configurations.extend((s,sig,thr,None) for s in ['dense','fixed_128_512','fixed_256_512','feedback'])
    configurations.extend(('learned','entropy',t,q) for t in [.8,.85] for q in [0.,.1,.25,.5,.75,.9,1.01])
    dev=samples[ntrain:]
    for schedule,sig,thr,tau in configurations:
        decisions=[]
        for i,r in enumerate(dev,ntrain):
            observed=[]
            def observe(j):observed.append(j);return ps[i,j,:r['candidate_count']]
            def cheap(j):return float(predictions[thr][i,j-1])
            end=min(1024,r['generated_tokens'])
            out=simulate_reads(observe,cheap,sig,thr,schedule,tau,end=end)
            assert observed==out['read_indices']
            decision=dict(sample_id=r['sample_id'],correct=bool(correct[i,out['at']]),terminal_correct=bool(correct[i,4]),
                          tokens=int(end if out['at']==4 else GRID[out['at']]),terminal_tokens=end,
                          reads=out['reads'],trigger_calls=out['trigger_calls'],read_indices=observed)
            decisions.append(decision)
        meta=dict(schedule=schedule,signal=sig,stop_threshold=thr,trigger_threshold=tau)
        curves.extend(dict(**meta,**m) for m in aggregate(decisions,dev))
        details.extend(dict(**meta,**d) for d in decisions)
    write('reading/analysis.json',dict(scope='Observed dev256; token grid only; no live cost/causal equivalence claim',
          diagnostics=diagnostics,curves=curves,unobserved_probability_invariance=True,
          cost='Heavy reads and cheap trigger calls charged separately; trigger and hidden export latency not measured'))
    write('reading/decisions.json',details)
    return curves


def run_stopping(samples, ps, correct, ntrain):
    status('multi_horizon_stopping_fit', state='running')
    tx=[];targets=[];weights=[]
    devx=[];mapping=[]
    for i,r in enumerate(samples):
        previous=None
        end=min(1024,r['generated_tokens'])
        pair_count=sum(4-j for j in [1,2,3] if GRID[j]<end)
        for j in [1,2,3]:
            base=feature(ps[i,j],j,r['candidate_count'],previous)
            for h in range(j+1,5):
                vector=base+[(GRID[h]-GRID[j])/1024,GRID[h]/1024]
                if i<ntrain:
                    if GRID[j]<end:
                        tx.append(vector);targets.append(2*int(correct[i,j])+int(correct[i,h]));weights.append(1/pair_count)
                else:devx.append(vector);mapping.append((i-ntrain,j,h))
            previous=ps[i,j]
    tx=np.array(tx);devx=np.array(devx)
    scaler=StandardScaler().fit(tx)
    model=fit_lr(scaler.transform(tx),np.array(targets),C=1.,weight=np.array(weights))
    q=four_probs(model,scaler.transform(devx))
    gains={key:float(v[1]-v[2]) for key,v in zip(mapping,q)}
    existing=joblib.load(STREAM/'stopping-grid/stop-models.joblib')
    bases=[]
    for i,r in enumerate(samples[ntrain:],ntrain):
        previous=None
        for j in [1,2,3]:
            bases.append(feature(ps[i,j],j,r['candidate_count'],previous));previous=ps[i,j]
    transformed=existing['scaler'].transform(bases)
    current=existing['models']['current'].predict_proba(transformed)[:,list(existing['models']['current'].classes_).index(True)].reshape(-1,3)
    oldjoint=four_probs(existing['models']['joint'],transformed).reshape(-1,3,4)
    (ROOT/'stopping').mkdir(exist_ok=True)
    joblib.dump(dict(scaler=scaler,model=model),ROOT/'stopping/multi-horizon-model.joblib')
    configurations=[('terminal',None,None)]
    configurations.extend((arm,t,None) for arm in ['confidence','entropy','margin','consistent_entropy'] for t in THRESHOLDS)
    configurations.extend((arm,t,c) for arm in ['old_joint','same_current_terminal','same_current_multi'] for t in THRESHOLDS for c in PRICES)
    curves=[];details=[];dev=samples[ntrain:]
    for arm,threshold,price in configurations:
        decisions=[]
        for di,r in enumerate(dev):
            i=ntrain+di;at=4;count=0;previous=None
            end=min(1024,r['generated_tokens'])
            if arm!='terminal':
                for j in [j for j in [1,2,3] if GRID[j]<end]:
                    count+=1;p=ps[i,j,:r['candidate_count']]
                    if arm in ['confidence','entropy','margin']:
                        stop=certainty(p,arm)>=threshold
                    elif arm=='consistent_entropy':
                        stop=previous is not None and p.argmax()==previous.argmax() and certainty(p,'entropy')>=threshold and certainty(previous,'entropy')>=threshold
                    elif arm=='old_joint':
                        v=oldjoint[di,j-1];stop=v[2]+v[3]>=threshold and v[1]-v[2]<=price*(1-GRID[j]/1024)
                    elif arm=='same_current_terminal':
                        v=oldjoint[di,j-1];stop=current[di,j-1]>=threshold and v[1]-v[2]<=price*(1-GRID[j]/1024)
                    else:
                        net=max(gains[di,j,h]-price*(GRID[h]-GRID[j])/1024 for h in range(j+1,5))
                        stop=current[di,j-1]>=threshold and net<=0
                    previous=p
                    if stop:at=j;break
            if at==4:count+=1
            decisions.append(dict(sample_id=r['sample_id'],correct=bool(correct[i,at]),terminal_correct=bool(correct[i,4]),
                                  tokens=int(end if at==4 else GRID[at]),terminal_tokens=end,reads=count))
        meta=dict(arm=arm,threshold=threshold,price=price)
        curves.extend(dict(**meta,**m) for m in aggregate(decisions,dev))
        details.extend(dict(**meta,**d) for d in decisions)
    write('stopping/analysis.json',dict(scope='Observed dev256; all reason, fixed dense grid; no latency guarantee',
          train_pairs=len(targets),question_weight_sum=1.,curves=curves,
          multi_horizon_rule='Current-correct probability >=threshold and every configured continuation gain-price*length<=0; includes terminal horizon'))
    write('stopping/decisions.json',details)
    return curves


def report(gate,reading,stopping):
    lines=['# 三个决策的第一轮改进实验','',
      'CPU复用已有特征和折外读数；无新增生成、无GPU使用、无独立测试。三个模块分别实验，未确定最终方法。','',
      '## 启动：校准75%调用档位，历史已观察eval256','',
      '|模型|正确/256|调用|生成token|增量回放秒/题|','|---|---:|---:|---:|---:|']
    for r in gate:
        if r['split']=='eval' and r['dataset']=='all' and r['fraction']==.75:
            lines.append(f"|{r['policy']}|{r['correct']}|{r['calls']}|{r['generated_tokens']}|{r['replay_incremental_seconds']:.3f}|")
    lines+=['','## 读取：全部事前指定策略点','',
      '|调度|停止信号/阈值|触发阈值|正确/256|理想token|重读取|轻触发|','|---|---|---|---:|---:|---:|---:|']
    for r in reading:
        if r['dataset']=='all':lines.append(f"|{r['schedule']}|{r['signal']}/{r['stop_threshold']}|{r['trigger_threshold']}|{r['correct']}|{r['ideal_tokens']:.1f}|{r['mean_reads']:.3f}|{r['mean_trigger_calls']:.3f}|")
    lines+=['','## 停止：预设0.85/0.95和费用0/0.05代表点，完整曲线见JSON','',
      '|规则|阈值|费用系数|正确/256|纠错/退化|理想token|读取|','|---|---|---|---:|---|---:|---:|']
    for r in stopping:
        if r['dataset']=='all' and (r['arm']=='terminal' or (r['threshold'] in [.85,.95] and r['price'] in [None,0.,.05])):
            lines.append(f"|{r['arm']}|{r['threshold']}|{r['price']}|{r['correct']}|{r['gained']}/{r['lost']}|{r['ideal_tokens']:.1f}|{r['mean_reads']:.3f}|")
    lines+=['','读取/停止的理想token不包括触发器、缓存、异步超发与调度费用。启动成本为历史轨迹回放；不可相互合并或视作实测加速。',
      '轻量触发器只在三个token位置验证，句末密集监督与在线实现仍待完成。所有学习式模型在历史训练数据拟合，开发结果只用于筛查，需后续冻结和新数据确认。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only experiment'
    assert not (ROOT/'completion.json').exists(), 'Already completed; do not overwrite evidence'
    torch.set_num_threads(2)
    sources=[Path(__file__),ROOT/'protocol.md',GATE/'paired-targets-cpu.jsonl',GATE/'initial-inputs.jsonl',
             STREAM/'samples.jsonl',STREAM/'labels-observed.jsonl',STREAM/'stopping-grid/oof-probabilities.pt',
             STREAM/'training/answer_position_linear-s29.pt',STREAM/'stopping-grid/stop-models.joblib',STREAM/'features/index.jsonl']
    bindings={str(p.relative_to(PROJECT)):sha(p) for p in sources}
    write('protocol.json',dict(UTC=now(),state='fixed_before_new_fits',seed=SEED,C=CS,prices=PRICES,thresholds=THRESHOLDS,
          source_sha256=bindings,CPU_threads=2,GPU=False,independent_test=False,new_generation=0,
          protected=['prethink-hidden-gate/fresh-validation','prefix-readout/confirmation'],
          versions=dict(torch=torch.__version__,numpy=np.__version__)))
    start=time.perf_counter()
    with threadpool_limits(limits=2):
        repair=ROOT/'engineering/affine-fix.json'
        if repair.exists():
            for rel,digest in read(repair)['retained_artifact_hashes'].items():assert sha(ROOT/rel)==digest,rel
            gate=read(ROOT/'gate/analysis.json')['curves']
        else:gate=run_gate()
        samples,ps,correct,ntrain=stream_data()
        stopping=read(ROOT/'stopping/analysis.json')['curves'] if repair.exists() else run_stopping(samples,ps,correct,ntrain)
        reading=run_reading(samples,ps,correct,ntrain)
        report(gate,reading,stopping)
    for rel,digest in bindings.items():assert sha(PROJECT/rel)==digest,rel
    write('completion.json',dict(UTC=now(),state='completed',CPU_seconds=time.perf_counter()-start,
          new_generation=0,GPU_used=False,modules=['gate','reading','stopping'],
          curve_rows=dict(gate=len(gate),reading=len(reading),stopping=len(stopping))))
    status('completed',state='completed',elapsed_seconds=time.perf_counter()-start)


if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',state='failed',error=traceback.format_exc())
        raise
