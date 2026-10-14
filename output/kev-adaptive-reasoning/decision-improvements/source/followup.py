"""Finite CPU diagnosis and label-free sentence workload preparation."""
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run as e


def save(name,value):e.write('followup/'+name,value)


def status(phase,**kw):
    value=dict(UTC=e.now(),phase=phase,**kw)
    save('status.json',value);print(json.dumps(value),flush=True)


def gate_diagnosis():
    status('gate_diagnosis')
    data=e.rows(e.GATE/'paired-targets-cpu.jsonl'); inputs=e.rows(e.GATE/'initial-inputs.jsonl')
    bindings=e.read(e.ROOT/'gate/feature-bindings.json')
    assert [r['key'] for r in data]==[r['key'] for r in inputs]
    features=[]
    for r,x in zip(data,inputs):
        file=e.GATE/'features'/(r['key']+'.npz')
        assert e.sha(file)==bindings[str(file.relative_to(e.PROJECT))]
        with np.load(file) as a:h=a['hidden'].copy()
        p=np.array(x['reference_probabilities']);s=np.sort(p)
        four=[s[-1],s[-1]-s[-2],-(p*np.log(np.maximum(p,1e-30))).sum()/np.log(len(p)),np.log1p(len(x['encoding']['ids']))]
        features.append(np.concatenate([four,h]))
    saved=joblib.load(e.ROOT/'gate/models.joblib');z=saved['scaler'].transform(features)
    gain,harm=[m.predict_proba(z)[:,1] for m in saved['classifiers']['two_binary']['models']]
    selection=e.read(e.ROOT/'gate/selection.json');threshold=selection['thresholds']['two_binary_cost0.0']['0.75']
    route=gain-harm>threshold
    initial=np.array([r['initial_correct'] for r in data],dtype=bool)
    terminal=np.array([r['reasoning_correct'] for r in data],dtype=bool)
    costs=np.array([r['reasoning_seconds']+r['readout_seconds'] for r in data])
    train=np.array([r['split']=='train' for r in data])
    predicted=np.expm1(np.clip(saved['cost_model'].predict(z),0,np.log1p(max(costs[train])*10)))
    summaries=[];cases=[]
    for split in ['cal','eval']:
        for dataset in ['all','proofwriter','logiqa2']:
            idx=np.array([i for i,r in enumerate(data) if r['split']==split and (dataset=='all' or r['dataset']==dataset)])
            a,b,m=initial[idx],terminal[idx],route[idx]
            actual=np.where(m,b,a);missed=(~a&b&~m);damaged=(a&~b&m)
            oracle=int((a|b).sum())
            assert oracle-int(actual.sum())==int(missed.sum()+damaged.sum())
            summaries.append(dict(split=split,dataset=dataset,n=len(idx),direct_correct=int(a.sum()),
                all_reason_correct=int(b.sum()),gate_correct=int(actual.sum()),oracle_correct=oracle,
                recoverable_total=int((~a&b).sum()),recoverable_captured=int((~a&b&m).sum()),
                missed_recoverable=int(missed.sum()),introduced_harm=int(damaged.sum()),calls=int(m.sum()),
                both_wrong=int((~a&~b).sum()),learned_cost_MAE=float(np.abs(predicted[idx]-costs[idx]).mean()),
                constant_train_median_MAE=float(np.abs(np.median(costs[train])-costs[idx]).mean())))
            if dataset=='all':
                for i in idx:
                    r=data[i]
                    if (~initial[i]&terminal[i]&~route[i]) or (initial[i]&~terminal[i]&route[i]):
                        cases.append(dict(key=r['key'],split=split,dataset=r['dataset'],
                            category='missed_recoverable' if not route[i] else 'introduced_harm',
                            predicted_gain=float(gain[i]),predicted_harm=float(harm[i]),
                            score=float(gain[i]-harm[i]),threshold=threshold))
    save('gate.json',dict(summaries=summaries,cases=cases,selection='Frozen calibration-75% threshold',
         oracle='Both path outcomes known after execution; diagnostic only, not deployable'))
    return summaries


def stopping_diagnosis():
    status('stopping_diagnosis')
    samples,ps,correct,ntrain=e.stream_data();dev=samples[ntrain:]
    saved=joblib.load(e.STREAM/'stopping-grid/stop-models.joblib')
    feats=[]
    for i,r in enumerate(dev,ntrain):
        previous=None
        for j in [1,2,3]:
            feats.append(e.feature(ps[i,j],j,r['candidate_count'],previous));previous=ps[i,j]
    z=saved['scaler'].transform(feats)
    current=saved['models']['current'].predict_proba(z)[:,list(saved['models']['current'].classes_).index(True)].reshape(-1,3)
    joint=e.four_probs(saved['models']['joint'],z).reshape(-1,3,4)
    curves=[];cases=[]
    settings=[(sig,thr,None) for sig in ['confidence','entropy','margin'] for thr in e.THRESHOLDS]
    settings += [('rescue_risk',thr,risk) for thr in [.8,.9,.95] for risk in [0.,.025,.05,.1,.2]]
    for arm,thr,risk in settings:
        decisions=[]
        for di,r in enumerate(dev):
            i=ntrain+di;at=4;reads=0;end=min(1024,r['generated_tokens'])
            for j in [j for j in [1,2,3] if e.GRID[j]<end]:
                reads+=1
                stop=(current[di,j-1]>=thr and joint[di,j-1,1]<=risk) if arm=='rescue_risk' else e.certainty(ps[i,j,:r['candidate_count']],arm)>=thr
                if stop:at=j;break
            if at==4:reads+=1
            d=dict(sample_id=r['sample_id'],dataset=r['dataset'],correct=bool(correct[i,at]),terminal_correct=bool(correct[i,4]),
                   tokens=int(end if at==4 else e.GRID[at]),terminal_tokens=end,reads=reads,at=at)
            decisions.append(d)
            if at<4 and not d['correct']:
                cases.append(dict(arm=arm,threshold=thr,risk_bound=risk,**d))
        for summary in e.aggregate(decisions,dev):
            subset=[d for d in decisions if summary['dataset']=='all' or d['dataset']==summary['dataset']]
            early=[d for d in subset if d['at']<4]
            summary.update(arm=arm,threshold=thr,risk_bound=risk,
                 wrong_early=sum(not d['correct'] for d in early),early_error_rate=sum(not d['correct'] for d in early)/len(early) if early else None)
            curves.append(summary)
    original=e.read(e.ROOT/'stopping/analysis.json')['curves']
    for r in curves:
        if r['arm']=='rescue_risk':continue
        old=next(v for v in original if v['arm']==r['arm'] and v['dataset']==r['dataset'] and v['threshold']==r['threshold'])
        for key in ['correct','gained','lost','early','ideal_tokens','mean_reads']:assert r[key]==old[key],(key,r,old)
    train_profiles=[]
    for dataset in ['proofwriter','logiqa2']:
        for j in [1,2,3]:
            ix=[i for i,r in enumerate(samples[:ntrain]) if r['dataset']==dataset and e.GRID[j]<min(1024,r['generated_tokens'])
                and e.certainty(ps[i,j,:r['candidate_count']],'entropy')>=.8]
            train_profiles.append(dict(dataset=dataset,tokens=int(e.GRID[j]),eligible=len(ix),
                correct=sum(bool(correct[i,j]) for i in ix),rescued_by_terminal=sum(not correct[i,j] and bool(correct[i,4]) for i in ix)))
    save('stopping.json',dict(curves=curves,wrong_exit_cases=cases,training_OOF_profiles=train_profiles,
         rule='Current correctness >= threshold AND P(current wrong, terminal correct) <= risk_bound; estimated probability, not a guarantee'))
    return curves


def sentence_manifest():
    status('sentence_manifest')
    from transformers import AutoTokenizer
    sys.path.insert(0,str(e.STREAM/'source'))
    from sentences import points
    model=e.PROJECT/'experiments/kev-adaptive-reasoning/models/Qwen3.5-4B'
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    samples=e.rows(e.STREAM/'samples.jsonl');inputs=e.rows(e.STREAM/'inputs-no-gold.jsonl')
    folds={r['sample_id']:r['held_out_fold'] for r in e.rows(e.STREAM/'stopping-plan/oof-folds.jsonl')}
    dev_existing={r['sample_id']:r['points'] for r in e.rows(e.STREAM/'sentence-evaluation/trajectories.jsonl') if r['reader']=='stream_last_linear'}
    manifest=[];statistics=Counter()
    for k,(r,x) in enumerate(zip(samples,inputs)):
        assert r['sample_id']==x['sample_id'];end=len(x['generated_ids'])
        if r['split']=='cal':
            available=[{key:p[key] for key in ['prefix_tokens','observable_after_tokens']} for p in dev_existing[r['sample_id']]
                       if 'sentence' in p['roles'] and p['observable_after_tokens']<end]
        else:
            detected,skipped=points(tok,x['generated_ids']);statistics['token_alignment_omissions_train']+=skipped
            available=[{key:p[key] for key in ['prefix_tokens','observable_after_tokens']} for p in detected if p['observable_after_tokens']<end]
        chosen=[]
        for band in range(4):
            group=[p for p in available if min(p['prefix_tokens']//256,3)==band]
            count=len(group)
            if not count:continue
            chosen_group=sorted(group,key=lambda p:hashlib.sha256(f"20261009:{r['sample_id']}:{p['prefix_tokens']}".encode()).digest())[:2] if r['split']=='train' else group
            for p in chosen_group:
                chosen.append(dict(**p,training_weight=count/len(chosen_group)/len(available) if r['split']=='train' else None))
        chosen.sort(key=lambda p:p['prefix_tokens'])
        if r['split']=='train' and chosen:assert abs(sum(p['training_weight'] for p in chosen)-1)<1e-12
        assert all(0<p['prefix_tokens']<=p['observable_after_tokens']<end for p in chosen)
        head=e.STREAM/'stopping-grid'/f"oof-reader-{folds[r['sample_id']]}.pt" if r['split']=='train' else e.STREAM/'training/answer_position_linear-s29.pt'
        manifest.append(dict(index=r['index'],sample_id=r['sample_id'],split=r['split'],dataset=r['dataset'],component_id=r['component_id'],
                    candidate_count=r['candidate_count'],prompt_tokens=len(x['prompt_ids']),terminal_tokens=end,
                    head=str(head.relative_to(e.PROJECT)),held_out_fold=folds.get(r['sample_id']),
                    available_sentence_events=len(available),events=chosen))
        statistics[r['split']+'_questions']+=1;statistics[r['split']+'_available_events']+=len(available);statistics[r['split']+'_selected_events']+=len(chosen)
        if (k+1)%64==0:status('sentence_manifest',completed=k+1,total=len(samples))
    e.split_audit({s:{r['component_id'] for r in samples if r['split']==s} for s in ['train','cal']})
    save('sentence-manifest.json',dict(samples=manifest,counts=dict(statistics),
         state='prepared_only_not_GPU_extracted',new_generated_tokens=0,
         required_next='Frozen-backbone close-if-needed + FINAL answer-position forwards; OOF heads for training, s29 for dev. Measure deployment later.'))
    return statistics


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (e.ROOT/'followup/completion.json').exists()
    sources=[Path(__file__),e.ROOT/'source/run.py',e.ROOT/'followup/protocol.md',e.ROOT/'gate/models.joblib',
             e.ROOT/'gate/selection.json',e.GATE/'paired-targets-cpu.jsonl',e.GATE/'initial-inputs.jsonl',
             e.STREAM/'samples.jsonl',e.STREAM/'inputs-no-gold.jsonl',e.STREAM/'labels-observed.jsonl',
             e.STREAM/'stopping-plan/oof-folds.jsonl',e.STREAM/'stopping-grid/oof-probabilities.pt',
             e.STREAM/'stopping-grid/stop-models.joblib',e.STREAM/'training/answer_position_linear-s29.pt',
             e.STREAM/'sentence-evaluation/trajectories.jsonl',e.STREAM/'source/sentences.py',e.STREAM/'source/common.py']
    bindings={str(p.relative_to(e.PROJECT)):e.sha(p) for p in sources}
    save('bindings.json',bindings)
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        gate=gate_diagnosis();stopping=stopping_diagnosis();counts=sentence_manifest()
    lines=['# 组件改进第二步','', '已有开发数据上的诊断；未访问GPU2确认gold，未新增生成。','',
           '|启动评估集|任务|直接/全部推理/门控正确|理想上界|漏救回|误推理退化|成本回归MAE/常量MAE秒|',
           '|---|---|---|---:|---:|---:|---:|']
    for r in gate:
        lines.append(f"|{r['split']}|{r['dataset']}|{r['direct_correct']}/{r['all_reason_correct']}/{r['gate_correct']}|{r['oracle_correct']}|{r['missed_recoverable']}|{r['introduced_harm']}|{r['learned_cost_MAE']:.2f}/{r['constant_train_median_MAE']:.2f}|")
    lines+=['','|停止规则|正确性阈值|可救回概率上限|正确/256|错停损失|全部错停|理想token|完整读取|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in stopping:
        if r['dataset']=='all' and (r['arm']=='rescue_risk' or (r['arm']=='entropy' and r['threshold'] in [.8,.85])):
            lines.append(f"|{r['arm']}|{r['threshold']}|{r['risk_bound']}|{r['correct']}|{r['lost']}|{r['wrong_early']}|{r['ideal_tokens']:.1f}|{r['mean_reads']:.3f}|")
    lines+=['', '句末清单计数：'+json.dumps(dict(counts),ensure_ascii=False)+'。',
            '训练清单最多每题8个句末，按已知进度区间无gold抽样；开发保留所有可在终点前观察到的句末。清单已准备，尚未做答案位置GPU前向、句末触发器训练或在线性能验证。',
            'GPU2运行到何时停止的建议另见gpu2-status.json；没有修改其控制器或STOP标记。']
    (e.ROOT/'followup/report.md').write_text('\n'.join(lines)+'\n')
    for rel,digest in bindings.items():assert e.sha(e.PROJECT/rel)==digest,rel
    save('completion.json',dict(UTC=e.now(),state='completed',GPU_used=False,new_generation=0,
         sentence_stage='manifest_only',input_sha256=bindings))
    status('completed',state='completed')


if __name__=='__main__':main()
