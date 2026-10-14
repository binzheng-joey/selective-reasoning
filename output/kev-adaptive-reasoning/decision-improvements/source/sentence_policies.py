"""Sentence-level trigger/stopping experiments on frozen observed development.

No new generation/GPU use; no access to independent confirmation data.
"""
from collections import Counter
import hashlib
import os
from pathlib import Path
import time
import traceback
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run as e

OUT=e.ROOT/'sentence-policies'
COLLECT=e.ROOT/'sentence-supervision'
THRESHOLDS=[.7,.8,.85,.9,.95,.98]
TRIGGERS=[.1,.25,.5,.75,.9]
PRICES=[0.,.025,.05,.1]


def save(name,value):
    e.write('sentence-policies/'+name,value)


def status(phase,**kw):
    v=dict(UTC=e.now(),state='running',phase=phase,pid=os.getpid(),**kw)
    save('status.json',v);print(v,flush=True)


def cert(p,kind):
    z=np.sort(p)
    if kind=='confidence':return float(z[-1])
    if kind=='margin':return float(z[-1]-z[-2])
    return float(1+sum(float(v)*np.log(float(v)) for v in p if v>0)/np.log(len(p)))


def vector(p,t,count,task):
    padded=np.zeros(4);padded[:count]=p
    return np.r_[padded,cert(p,'confidence'),cert(p,'margin'),1-cert(p,'entropy'),t/1024,count/4,float(task=='logiqa2')]


def simulate(events,end,observe,terminal,stop,threshold,schedule,trigger=None,tau=None,
             learned_stop=None,price=0.):
    """The scheduler can access cheap trigger output but not skipped readouts."""
    previous=None;previous_ordinal=0;reads=[];calls=0;gap=4
    for ev in events:
        assert ev['observable_after_tokens']<end
        ordinal=ev['ordinal'];gid=ev['gid']
        if schedule=='terminal':continue
        if schedule=='dense':perform=True
        elif schedule=='every4':perform=ordinal-previous_ordinal>=4
        elif schedule=='every8':perform=ordinal-previous_ordinal>=8
        elif schedule=='feedback':perform=ordinal-previous_ordinal>=gap
        elif schedule in ['learned','learned_gap4','task_progress','task_progress_gap4']:
            if schedule.endswith('gap4') and ordinal-previous_ordinal<4:continue
            calls+=1;perform=trigger(gid)>=tau
        else:raise ValueError(schedule)
        if not perform:continue
        p=observe(gid);reads.append(gid)
        if stop in ['confidence','margin','entropy']:
            halt=cert(p,stop)>=threshold
        elif stop=='consistent_entropy':
            halt=previous is not None and p.argmax()==previous.argmax() and cert(p,'entropy')>=threshold and cert(previous,'entropy')>=threshold
        elif stop in ['current_raw','current_calibrated']:
            halt=learned_stop(gid,stop)>=threshold
        elif stop=='continuation_gain':
            values=learned_stop(gid,stop)
            halt=values[0]>=threshold and values[1]<=price*(1-ev['prefix_tokens']/1024)
        else:raise ValueError(stop)
        if halt:
            return dict(pred=int(p.argmax()),at=gid,tokens=ev['observable_after_tokens'],prefix_tokens=ev['prefix_tokens'],
                        reads=len(reads),read_indices=reads,trigger_calls=calls,sentence_opportunities=ordinal)
        if schedule=='feedback':
            stable=previous is not None and p.argmax()==previous.argmax() and abs(cert(p,'entropy')-cert(previous,'entropy'))<.03
            gap=8 if cert(p,'entropy')<threshold-.1 or stable else 4
        previous=p;previous_ordinal=ordinal
    p=terminal()
    return dict(pred=int(p.argmax()),at=None,tokens=end,prefix_tokens=end,reads=len(reads)+1,
                read_indices=reads,trigger_calls=calls,sentence_opportunities=events[-1]['ordinal'] if events else 0)


def simulator_checks():
    events=[dict(gid=j,ordinal=j+1,prefix_tokens=20*(j+1),observable_after_tokens=20*(j+1)+1) for j in range(5)]
    saved=[]
    for alternate in [np.array([.999,.001]),np.array([.001,.999])]:
        seen=[]
        def observe(gid):
            seen.append(gid)
            return np.array([.99,.01]) if gid==3 else alternate
        result=simulate(events,128,observe,lambda:np.array([.1,.9]),'confidence',.95,'every4')
        assert seen==[3] and result['tokens']==81 and result['pred']==0
        saved.append(result)
    assert saved[0]==saved[1]
    seen=[];called=[]
    def observe(gid):seen.append(gid);return np.array([.99,.01])
    def trigger(gid):called.append(gid);return 0.
    result=simulate(events,128,observe,lambda:np.array([.1,.9]),'confidence',.95,'learned',trigger,.5)
    assert seen==[] and result['reads']==1 and result['tokens']==128 and result['pred']==1
    assert called==list(range(5)) and result['trigger_calls']==5
    return dict(skipped_probability_invariance=True,terminal_fallback=True,observable_time_charged=True,
                trigger_calls_charged=True)


def load_data():
    assert e.read(COLLECT/'completion.json')['state']=='completed'
    work=e.read(COLLECT/'work.json');indices={x['index']:x for x in e.read(COLLECT/'feature-index.json')}
    labels={r['sample_id']:r['gold'] for r in e.rows(e.STREAM/'labels-observed.jsonl')}
    old_samples,old_probs,_,_=e.stream_data()
    terminals={r['sample_id']:p[-1,:r['candidate_count']] for r,p in zip(old_samples,old_probs)}
    assert {r['sample_id'] for r in work}==set(terminals)
    e.split_audit({split:{r['component_id'] for r in work if r['split']==split} for split in ['train','cal']})
    all_hidden=[];all_probs=[];all_x=[];all_weights=[];all_y=[];all_terminal=[];all_meta=[];samples=[]
    gid=0;coalesced=0;not_early=0
    for r in work:
        f=COLLECT/'features'/f"{r['index']:04}.pt"
        assert e.sha(f)==indices[r['index']]['sha256']
        a=torch.load(f,map_location='cpu',weights_only=True)
        assert a['sample_id']==r['sample_id'] and a['events']==r['events']
        assert a['input_hashes']==[ev['input_sha256'] for ev in r['events']]
        assert a['probabilities'].shape==(len(r['events']),r['candidate_count'])
        assert torch.isfinite(a['probabilities']).all() and torch.isfinite(a['natural_states']).all()
        gold=labels[r['sample_id']];terminal=terminals[r['sample_id']].astype(float)
        valid=[j for j,ev in enumerate(r['events']) if ev['observable_after_tokens']<r['terminal_tokens']]
        not_early+=len(r['events'])-len(valid)
        # At a common observable time, use only the latest complete prefix.
        latest={r['events'][j]['observable_after_tokens']:j for j in valid}
        selected=sorted(latest.values())
        coalesced+=len(valid)-len(selected)
        weight_sum=sum(r['events'][j]['training_weight'] for j in selected) if r['split']=='train' else 1.
        sample={**r,'gold':gold,'terminal_probabilities':terminal,'terminal_correct':bool(terminal.argmax()==gold),'points':[]}
        for j in selected:
            ev=r['events'][j];p=a['probabilities'][j].numpy().astype(float)
            assert abs(p.sum()-1)<1e-5
            weight=float(ev['training_weight']/weight_sum) if r['split']=='train' else 0.
            point=dict(gid=gid,ordinal=j+1,prefix_tokens=ev['prefix_tokens'],observable_after_tokens=ev['observable_after_tokens'])
            sample['points'].append(point)
            all_hidden.append(a['natural_states'][j].float().numpy())
            padded=np.zeros(4);padded[:r['candidate_count']]=p;all_probs.append(padded)
            all_x.append(vector(p,ev['prefix_tokens'],r['candidate_count'],r['dataset']))
            all_weights.append(weight);all_y.append(p.argmax()==gold);all_terminal.append(sample['terminal_correct'])
            all_meta.append(dict(index=r['index'],sample_id=r['sample_id'],component_id=r['component_id'],
                split=r['split'],dataset=r['dataset'],candidate_count=r['candidate_count'],prefix_tokens=ev['prefix_tokens']))
            gid+=1
        samples.append(sample)
        if len(samples)%256==0:status('loading_sentence_features',completed=len(samples),total=len(work))
    data=dict(samples=samples,hidden=np.array(all_hidden,dtype=np.float64),probabilities=np.array(all_probs),
              stop_x=np.array(all_x),weights=np.array(all_weights),correct=np.array(all_y,dtype=int),
              terminal_correct=np.array(all_terminal,dtype=int),meta=all_meta)
    data['train_mask']=np.array([r['split']=='train' for r in all_meta])
    data['task']=np.array([r['dataset']=='logiqa2' for r in all_meta],dtype=int)
    data['position']=np.array([r['prefix_tokens']/1024 for r in all_meta])
    for s in samples:
        if s['split']=='train' and s['points']:
            assert abs(sum(data['weights'][p['gid']] for p in s['points'])-1)<1e-6
    assert sum(s['terminal_correct'] for s in samples if s['split']=='cal')==203
    save('data-audit.json',dict(UTC=e.now(),questions=len(samples),train_questions=sum(s['split']=='train' for s in samples),
         dev_questions=sum(s['split']=='cal' for s in samples),collected_events=sum(len(r['events']) for r in work),
         usable_events=gid,coalesced_same_observable_time=coalesced,excluded_not_early=not_early,
         train_events=int(data['train_mask'].sum()),dev_events=int((~data['train_mask']).sum()),
         material_groups_disjoint=True,feature_hashes_verified=True,question_weights_normalized=True,
         terminal_reader_backend='Existing A100 fixed-budget readouts; sentence readouts RTX3080; A100 bridge reported separately'))
    return data


def opportunity(d):
    detail=[]
    for s in d['samples']:
        if s['split']!='cal':continue
        points=s['points'];ids=[x['gid'] for x in points]
        c=[bool(d['correct'][i]) for i in ids]+[s['terminal_correct']]
        preds=[int(d['probabilities'][i].argmax()) for i in ids]+[int(s['terminal_probabilities'].argmax())]
        first=next((points[j]['observable_after_tokens'] for j,v in enumerate(c[:-1]) if v),None)
        stable=next((points[j]['observable_after_tokens'] for j in range(len(points)) if all(c[j:])),None)
        row=dict(sample_id=s['sample_id'],dataset=s['dataset'],sentence_points=len(points),terminal_correct=s['terminal_correct'],
            any_prefix_correct=any(c[:-1]),first_correct_observable_tokens=first,first_stably_correct_tokens=stable,
            answer_changes=sum(a!=b for a,b in zip(preds,preds[1:])),
            wrong_terminal_with_correct_prefix=not s['terminal_correct'] and any(c[:-1]))
        detail.append(row)
    summaries=[]
    for task in ['all','proofwriter','logiqa2']:
        chosen=[r for r in detail if task=='all' or r['dataset']==task]
        counts=[r['sentence_points'] for r in chosen]
        summaries.append(dict(dataset=task,n=len(chosen),terminal_correct=sum(r['terminal_correct'] for r in chosen),
            any_prefix_correct=sum(r['any_prefix_correct'] for r in chosen),
            oracle_union_correct=sum(r['any_prefix_correct'] or r['terminal_correct'] for r in chosen),
            mean_sentence_points=float(np.mean(counts)),median_sentence_points=float(np.median(counts)),
            p90_sentence_points=float(np.quantile(counts,.9)),
            first_correct_by_256=sum(r['first_correct_observable_tokens'] is not None and r['first_correct_observable_tokens']<=256 for r in chosen),
            first_stably_correct_by_512=sum(r['first_stably_correct_tokens'] is not None and r['first_stably_correct_tokens']<=512 for r in chosen),
            wrong_terminal_with_correct_prefix=sum(r['wrong_terminal_with_correct_prefix'] for r in chosen)))
    save('opportunity.json',dict(scope='Observed dev256, retrospective oracle diagnostics only; not executable policies',
                               summaries=summaries,details=detail))
    return summaries


def fit_triggers(d):
    status('fitting_read_triggers')
    mask=d['train_mask'];h=d['hidden'];w=d['weights']
    scaler=StandardScaler().fit(h[mask],sample_weight=w[mask])
    pca=PCA(n_components=64,svd_solver='randomized',random_state=e.SEED).fit(scaler.transform(h[mask]))
    reduced=pca.transform(scaler.transform(h))
    x=np.column_stack([reduced,d['position'],d['task']])
    models={};scores={};tables={};checks={}
    entropy=np.array([cert(p[:r['candidate_count']],'entropy') for p,r in zip(d['probabilities'],d['meta'])])
    for threshold in [.8,.85]:
        target=((entropy>=threshold)&d['correct'].astype(bool)).astype(int)
        model=e.fit_lr(x[mask],target[mask],C=.01,weight=w[mask])
        logits=model.decision_function(x)
        projection=model.coef_[0,:64]@pca.components_
        hidden_weight=projection/scaler.scale_
        bias=float(model.intercept_[0]-projection@pca.mean_-hidden_weight@scaler.mean_)
        folded=h@hidden_weight+x[:,-2:]@model.coef_[0,-2:]+bias
        error=float(np.max(np.abs(logits-folded)));assert error<1e-7
        prob=model.predict_proba(x)[:,1]
        table={};lookup=[]
        for task in [0,1]:
            for band in range(4):
                idx=mask&(d['task']==task)&(np.minimum((d['position']*4).astype(int),3)==band)
                table[f'{task}/{band}']=float((np.sum(w[idx]*target[idx])+1)/(w[idx].sum()+2))
        for task,pos in zip(d['task'],d['position']):lookup.append(table[f'{task}/{min(int(pos*4),3)}'])
        models[str(threshold)]=dict(model=model,hidden_weight=hidden_weight,bias=bias,
                                    position_weight=float(model.coef_[0,-2]),task_weight=float(model.coef_[0,-1]))
        scores[threshold]=prob;tables[threshold]=np.array(lookup)
        checks[str(threshold)]=dict(folded_max_logit_difference=error,train_positive_weighted_rate=float(np.average(target[mask],weights=w[mask])),task_progress_table=table)
    joblib.dump(dict(scaler=scaler,pca=pca,models=models),OUT/'trigger-models.joblib')
    save('trigger-fit.json',checks)
    return scores,tables


def fit_stoppers(d):
    status('fitting_stopping_models')
    fit_groups=set()
    for task in ['proofwriter','logiqa2']:
        groups=sorted({r['component_id'] for r in d['meta'] if r['split']=='train' and r['dataset']==task},
            key=lambda g:hashlib.sha256(('sentence-stop-cal-20261009:'+g).encode()).hexdigest())
        fit_groups.update(g for j,g in enumerate(groups) if j%4!=0)
    fit=d['train_mask']&np.array([r['component_id'] in fit_groups for r in d['meta']])
    calibrate=d['train_mask']&~fit
    assert fit.any() and calibrate.any()
    scaler=StandardScaler().fit(d['stop_x'][fit],sample_weight=d['weights'][fit])
    x=scaler.transform(d['stop_x'])
    current=e.fit_lr(x[fit],d['correct'][fit],C=1.,weight=d['weights'][fit])
    raw=current.predict_proba(x)[:,1]
    logit=current.decision_function(x)[:,None]
    calibration=e.fit_lr(logit[calibrate],d['correct'][calibrate],C=1.,weight=d['weights'][calibrate])
    calibrated=calibration.predict_proba(logit)[:,1]
    joint=e.fit_lr(x[fit],2*d['correct'][fit]+d['terminal_correct'][fit],C=1.,weight=d['weights'][fit])
    q=e.four_probs(joint,x)
    joblib.dump(dict(scaler=scaler,current=current,calibration=calibration,joint=joint),OUT/'stop-models.joblib')
    diag=dict(fit_questions=len({d['meta'][i]['component_id'] for i in np.flatnonzero(fit)}),
         calibration_questions=len({d['meta'][i]['component_id'] for i in np.flatnonzero(calibrate)}),
         disjoint_groups=True,features='Current readout probabilities, certainty, progress and task; no future or skipped readouts',
         current_calibration='Sigmoid fitted on separate training material groups; empirical reliability, not a risk guarantee')
    reliability=[]
    for name,values in [('current_raw',raw),('current_calibrated',calibrated)]:
        for task in ['all','proofwriter','logiqa2']:
            chosen=np.array([r['split']=='cal' and (task=='all' or r['dataset']==task) for r in d['meta']])
            for lo,hi in [(0.,.5),(.5,.7),(.7,.8),(.8,.9),(.9,1.01)]:
                take=chosen&(values>=lo)&(values<hi)
                reliability.append(dict(model=name,dataset=task,lower=lo,upper=hi,n_events=int(take.sum()),
                    mean_score=float(values[take].mean()) if take.any() else None,
                    observed_correct_fraction=float(d['correct'][take].mean()) if take.any() else None))
    save('stop-fit.json',dict(**diag,event_reliability=reliability,
        reliability_scope='Correlated prefix events, descriptive only; policy-level error rates reported separately'))
    return dict(current_raw=raw,current_calibrated=calibrated,continuation_gain=np.column_stack([calibrated,q[:,1]-q[:,2]]))


def aggregate(decisions):
    rows=[]
    for task in ['all','proofwriter','logiqa2']:
        subset=[r for r in decisions if task=='all' or r['dataset']==task]
        early=[r for r in subset if r['tokens']<r['terminal_tokens']]
        rows.append(dict(dataset=task,n=len(subset),correct=sum(r['correct'] for r in subset),
            terminal_correct=sum(r['terminal_correct'] for r in subset),
            gained=sum(r['correct'] and not r['terminal_correct'] for r in subset),
            lost=sum(not r['correct'] and r['terminal_correct'] for r in subset),
            early=len(early),early_wrong=sum(not r['correct'] for r in early),
            early_error_rate=sum(not r['correct'] for r in early)/len(early) if early else None,
            ideal_tokens=float(np.mean([r['tokens'] for r in subset])),mean_reads=float(np.mean([r['reads'] for r in subset])),
            mean_trigger_calls=float(np.mean([r['trigger_calls'] for r in subset])),
            total_reads=sum(r['reads'] for r in subset),total_trigger_calls=sum(r['trigger_calls'] for r in subset)))
    return rows


def evaluate(d,trigger_scores,tables,stop_scores):
    status('simulating_policies')
    configs=[dict(family='reading',schedule='terminal',stop='entropy',threshold=.85,tau=None,price=0.)]
    for threshold in [.8,.85]:
        for schedule in ['dense','every4','every8','feedback']:
            configs.append(dict(family='reading',schedule=schedule,stop='entropy',threshold=threshold,tau=None,price=0.))
        for schedule in ['learned','learned_gap4','task_progress','task_progress_gap4']:
            for tau in TRIGGERS:
                configs.append(dict(family='reading',schedule=schedule,stop='entropy',threshold=threshold,tau=tau,price=0.))
    for stop in ['confidence','margin','entropy','consistent_entropy','current_raw','current_calibrated']:
        for threshold in THRESHOLDS:
            configs.append(dict(family='stopping',schedule='every4',stop=stop,threshold=threshold,tau=None,price=0.))
    for threshold in THRESHOLDS:
        for price in PRICES:
            configs.append(dict(family='stopping',schedule='every4',stop='continuation_gain',threshold=threshold,tau=None,price=price))
    assert len(configs)==109
    curves=[];all_decisions=[];dev=[s for s in d['samples'] if s['split']=='cal']
    for cid,conf in enumerate(configs):
        decisions=[]
        for s in dev:
            observed=[]
            def observe(gid):
                observed.append(gid)
                return d['probabilities'][gid,:s['candidate_count']]
            def terminal():return s['terminal_probabilities']
            def trigger(gid):
                scores=tables if conf['schedule'].startswith('task_progress') else trigger_scores
                return float(scores[conf['threshold']][gid])
            def learned_stop(gid,rule):return stop_scores[rule][gid]
            result=simulate(s['points'],s['terminal_tokens'],observe,terminal,conf['stop'],conf['threshold'],
                conf['schedule'],trigger,conf['tau'],learned_stop,conf['price'])
            assert observed==result['read_indices']
            assert result['tokens']<=s['terminal_tokens']
            decisions.append(dict(sample_id=s['sample_id'],dataset=s['dataset'],correct=result['pred']==s['gold'],
                 terminal_correct=s['terminal_correct'],terminal_tokens=s['terminal_tokens'],**result))
        policy=f's{cid:03d}'
        curves.extend(dict(policy=policy,**conf,**r) for r in aggregate(decisions))
        all_decisions.extend(dict(policy=policy,**conf,**r) for r in decisions)
        if cid%20==0:status('simulating_policies',completed=cid+1,total=len(configs))
    # Same rule in the two isolated experiment blocks must give the same per-question results.
    for threshold in [.8,.85]:
        left=[r for r in all_decisions if r['family']=='reading' and r['schedule']=='every4' and r['threshold']==threshold]
        right=[r for r in all_decisions if r['family']=='stopping' and r['stop']=='entropy' and r['threshold']==threshold]
        assert [(r['sample_id'],r['pred'],r['tokens'],r['reads']) for r in left]==[(r['sample_id'],r['pred'],r['tokens'],r['reads']) for r in right]
    for r in curves:
        assert r['correct']==r['terminal_correct']+r['gained']-r['lost']
    save('analysis.json',dict(UTC=e.now(),curves=curves,
        scope='Observed dev256 sentence policies, all start reasoning; no new online timing',
        cost='Observable token count (includes sentence-boundary lookahead), heavy reads and cheap calls; excludes async overshoot and execution overhead',
        backend='RTX3080 sentence readouts plus existing A100 natural states and terminal readouts; A100 bridge separately assessed'))
    save('decisions.json',all_decisions)
    return curves,all_decisions


def report(opportunity,curves):
    lines=['# 句末读取调度与停止规则实验','',
        '1024训练题采用折外答案读取头，256题为已观察开发资料；冻结生成内容、底座和答案头。当前比较全部进入推理，不与启动门控收益合并。','',
        '## 事后信息空间','',
        '|任务|题数|终点正确|任一句末正确|句末或终点理想上界|平均句末数|首次正确≤256|后续持续正确起点≤512|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in opportunity:
        lines.append(f"|{r['dataset']}|{r['n']}|{r['terminal_correct']}|{r['any_prefix_correct']}|{r['oracle_union_correct']}|{r['mean_sentence_points']:.2f}|{r['first_correct_by_256']}|{r['first_stably_correct_by_512']}|")
    lines+=['','这些是事后诊断，不是控制器可获知的标签或可实现准确率。','',
        '## 读取调度：事先固定熵确定性0.85，学习/查表触发阈值0.25','',
        '|策略|正确/256|理想token|完整读取|轻触发|早停|错停损失|早停纠错|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in curves:
        if r['dataset']=='all' and r['family']=='reading' and r['threshold']==.85 and r['tau'] in [None,.25]:
            lines.append(f"|{r['schedule']}|{r['correct']}|{r['ideal_tokens']:.1f}|{r['mean_reads']:.3f}|{r['mean_trigger_calls']:.2f}|{r['early']}|{r['lost']}|{r['gained']}|")
    lines+=['','## 停止：固定每新增4句读取，事先固定代表阈值0.85/0.95、继续费用0/0.05','',
        '|规则|阈值|费用|正确/256|理想token|完整读取|早停|早停错误|错停损失|早停纠错|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in curves:
        if r['dataset']=='all' and r['family']=='stopping' and r['threshold'] in [.85,.95] and r['price'] in [0.,.05]:
            lines.append(f"|{r['stop']}|{r['threshold']}|{r['price']}|{r['correct']}|{r['ideal_tokens']:.1f}|{r['mean_reads']:.3f}|{r['early']}|{r['early_wrong']}|{r['lost']}|{r['gained']}|")
    lines+=['','完整曲线、分任务结果与逐题决定见analysis.json和decisions.json。代表点在执行前指定，不据开发最优点宣称胜出。',
        '读取训练使用每题抽样权重；触发器预测此刻昂贵读取既高确定又正确的可能性，在线不能看到跳过的昂贵读数。',
        '停止模型仅用当前实际读取及位置/任务特征，继续收益目标为同一题既有终点，未来答案只用于训练。',
        'CPU回放未包括自然状态导出、缓存分支、异步取消及设备争用成本；不能由读取次数推断端到端速度。',
        '句末读数来自RTX3080，自然状态和终点为历史A100资料；正式A100评估应重跑候选，数值桥接检查另见sentence-a100-check。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'CPU-only policies'
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Do not overwrite frozen experiment'
    checks=simulator_checks()
    sources=[Path(__file__),e.ROOT/'source/run.py',COLLECT/'protocol.json',COLLECT/'multigpu/protocol.json',
       COLLECT/'work.json',COLLECT/'feature-index.json',e.STREAM/'labels-observed.jsonl',
       e.STREAM/'stopping-grid/oof-probabilities.pt',e.STREAM/'training/answer_position_linear-s29.pt']
    bindings={str(p.relative_to(e.PROJECT)):e.sha(p) for p in sources}
    save('protocol.json',dict(UTC=e.now(),source_bindings=bindings,thresholds=THRESHOLDS,trigger_thresholds=TRIGGERS,prices=PRICES,
        train='1024 question-disjoint OOF readout labels; trigger uses all train; stop fitting/calibration use disjoint 3/4 and1/4 training groups',
        dev='256 observed development questions, full curves reported, not independent confirmation',
        trigger='Training-only PCA64 + logistic C0.01, natural prefix hidden state + token position + task; affine fold checked',
        task_progress='Training-only weighted smoothed high-certainty-correct rate by task and token quartile',
        stop='Current correctness logistic C1, separate-group sigmoid calibration; joint current/terminal outcomes logistic C1',
        weights='Inverse-probability sentence sampling weights normalized to total1 per training question',
        controls='Reading holds entropy rule fixed; stopping holds every4-sentence schedule fixed',
        cost='Generation to observable complete-sentence time, actual performed heavy reads, cheap triggers counted separately',
        representative_points='Reading entropy0.85 and trigger0.25; stopping thresholds0.85/0.95 with prices0/0.05',
        GPU=False,new_generation=0,confirmation_access=False,engineering_checks=checks))
    start=time.perf_counter();torch.set_num_threads(2)
    status('loading_sentence_features')
    with threadpool_limits(limits=2):
        d=load_data();op=opportunity(d)
        triggers,tables=fit_triggers(d);stops=fit_stoppers(d)
        curves,decisions=evaluate(d,triggers,tables,stops)
        report(op,curves)
    for rel,h in bindings.items():assert e.sha(e.PROJECT/rel)==h,rel
    save('checks.json',dict(**checks,matching_reading_stopping_controls=True,
        correctness_gain_loss_identity=True,source_hashes_unchanged=True))
    done=dict(UTC=e.now(),state='completed',wall_seconds=time.perf_counter()-start,policies=109,
        curve_rows=len(curves),decision_rows=len(decisions),GPU_used=False,confirmation_access=False)
    save('completion.json',done);save('status.json',done);print(done,flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        save('status.json',dict(UTC=e.now(),state='failed',error=traceback.format_exc()))
        raise
