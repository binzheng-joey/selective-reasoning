"""CPU startup gating for one native backbone/linear reader at 0 and 1024.

Existing OOF train readouts; native initial states only, no later-state features.
Calibration and threshold fitting use separate training material groups.
Observed dev256 is reported in full and is never used for fitting here.
"""
import collections
import hashlib
import os
import traceback
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import run as e
import gate_refinement as gr

OUT=e.ROOT/'gate-unified-endpoint'


def save(name,obj):e.write(OUT/name,obj)


def binary(x,y,ids):
    if len(np.unique(y[ids]))<2:return dict(constant=float((y[ids].sum()+1)/(len(ids)+2)))
    return e.fit_lr(x[ids],y[ids],C=.01)


def predict(model,x):
    return np.full(len(x),model['constant']) if isinstance(model,dict) else gr.probability(model,x)


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
    sources=[__file__,e.ROOT/'source/run.py',e.ROOT/'source/gate_refinement.py',e.STREAM/'samples.jsonl',
        e.STREAM/'stopping-grid/oof-probabilities.pt',e.STREAM/'training/answer_position_linear-s29.pt',e.STREAM/'labels-observed.jsonl']
    bindings={str(p):e.sha(p) for p in sources}
    save('protocol.json',dict(UTC=e.now(),bindings=bindings,
        endpoint='Native Qwen3.5-4B + FP32 answer-position linear at zero and1024; one fixed generation trace',
        partition='Within each task, training groups sorted by fixed SHA256: first75% fit, next12.5% calibrate, last12.5% thresholds; dev256 untouched by new fitting',
        model='Fixed C=.01 logistic gain/harm; initial-error predictor same C; no hyperparameter selection on observed dev',
        variants=['benefit_raw','benefit_global_cal','benefit_task_cal','initial_error','confidence','margin','entropy','task_mean'],
        fractions=gr.FRACTIONS,cost='Recorded generated tokens at1024; no end-to-end latency claim',
        targets='OOF reader predictions for train; deployed reader for observed dev',confirmation_access=False))
    save('status.json',dict(UTC=e.now(),state='running',phase='loading',pid=os.getpid()))
    samples,ps,corr,ntrain=e.stream_data();inputs={r['sample_id']:r for r in e.rows(e.STREAM/'inputs-no-gold.jsonl')}
    train=np.arange(ntrain);dev=np.arange(ntrain,len(samples));task=np.array([s['dataset']=='logiqa2' for s in samples],int)
    fit=[];cal=[];tune=[]
    for t in [0,1]:
        ids=sorted([int(i) for i in train if task[i]==t],key=lambda i:hashlib.sha256(('unified-gate-20261009:'+samples[i]['component_id']).encode()).hexdigest())
        assert len(ids)==512
        fit+=ids[:384];cal+=ids[384:448];tune+=ids[448:]
    fit,cal,tune=map(np.array,[fit,cal,tune])
    e.split_audit({name:{samples[i]['component_id'] for i in ids} for name,ids in [('fit',fit),('cal',cal),('tune',tune),('dev',dev)]})
    save('split.json',{name:[samples[i]['sample_id'] for i in ids] for name,ids in [('fit',fit),('cal',cal),('tune',tune),('dev',dev)]})
    h=[];four=[];features={};tokens=[]
    for i,s in enumerate(samples):
        path=e.PROJECT/s['answer_feature'];assert e.sha(path)==s['answer_feature_sha256']
        feature=e.torch.load(path,map_location='cpu',weights_only=True)
        h.append(feature['states'][0].float().numpy());features[str(path)]=s['answer_feature_sha256']
        p=ps[i,0,:s['candidate_count']].astype(float);z=np.sort(p)
        four.append([z[-1],z[-1]-z[-2],-sum(v*np.log(v) for v in p if v>0)/np.log(len(p)),np.log1p(len(inputs[s['sample_id']]['prompt_ids']))])
        tokens.append(min(1024,len(inputs[s['sample_id']]['generated_ids'])))
    save('feature-bindings.json',features)
    four=np.array(four);x=np.column_stack([four,h]);scaler=StandardScaler().fit(x[fit]);z=scaler.transform(x)
    a=corr[:,0].astype(int);b=corr[:,-1].astype(int);gain=(b-a==1).astype(int);harm=(b-a==-1).astype(int)
    models=[binary(z,y,fit) for y in [gain,harm,1-a]]
    pg,ph,wrong=[predict(m,z) for m in models]
    rules=dict(benefit_raw=pg-ph,initial_error=wrong,confidence=-four[:,0],margin=-four[:,1],entropy=four[:,2]);calibrators={}
    for conditional in [False,True]:
        values,ms=gr.calibrate(pg,ph,task,cal,gain,harm,conditional)
        name='benefit_task_cal' if conditional else 'benefit_global_cal';rules[name]=values[0]-values[1];calibrators[name]=ms
    means={t:float((b-a)[fit[task[fit]==t]].mean()) for t in [0,1]}
    jitter=np.array([int(hashlib.sha256(('unified-tie:'+s['sample_id']).encode()).hexdigest()[:12],16)/16**12 for s in samples])
    rules['task_mean']=np.array([means[t] for t in task])+1e-10*jitter
    curves=[];decisions={};tokens=np.array(tokens);thresholds={}
    for name,score in rules.items():
        ordered=np.sort(score[tune])[::-1]
        for frac in gr.FRACTIONS:
            threshold=None
            if frac in [0.,1.]:mask=np.full(len(samples),frac==1.,bool)
            else:
                n=max(1,min(len(ordered)-1,round(frac*len(ordered))));threshold=float((ordered[n-1]+ordered[n])/2);mask=score>threshold
            key=name+'/'+str(frac);thresholds[key]=threshold;decisions[key]=mask[dev].tolist()
            for dataset in ['all','proofwriter','logiqa2']:
                ids=np.array([i for i in dev if dataset=='all' or samples[i]['dataset']==dataset]);m=mask[ids];aa=a[ids];bb=b[ids]
                correct=int(np.where(m,bb,aa).sum());rescues=int(((bb-aa==1)&m).sum());harms=int(((bb-aa==-1)&m).sum())
                assert correct==int(aa.sum())+rescues-harms
                curves.append(dict(policy=name,fraction=frac,dataset=dataset,n=len(ids),correct=correct,accuracy=correct/len(ids),
                    calls=int(m.sum()),generated_tokens=int(tokens[ids][m].sum()),rescues=rescues,harms=harms,
                    missed_rescues=int(((bb-aa==1)&~m).sum()),unchanged_calls=int(((bb==aa)&m).sum()),
                    direct_correct=int(aa.sum()),always_correct=int(bb.sum()),oracle_correct=int(np.maximum(aa,bb).sum())))
    joblib.dump(dict(scaler=scaler,models=models,calibrators=calibrators,task_means=means,thresholds=thresholds),OUT/'models.joblib')
    save('analysis.json',dict(curves=curves,counts=dict(fit=len(fit),cal=len(cal),threshold=len(tune),dev=len(dev)),
        dev_direct_correct=int(a[dev].sum()),dev_always_correct=int(b[dev].sum()),dev_oracle_correct=int(np.maximum(a,b)[dev].sum()),
        train_outcomes=dict(collections.Counter(map(str,2*a[train]+b[train]))),initial='Native zero-token linear reader',terminal='Native1024 linear reader'))
    save('decisions.json',dict(sample_ids=[samples[i]['sample_id'] for i in dev],decisions=decisions))
    lines=['# 统一原生底座与终点的启动门控','',
        '初始与终点均使用原生Qwen3.5-4B的线性答案头，终点预算1024。训练题的答案使用折外预测。门控拟合768题、校准128题、阈值128题，均来自原训练1024题；开发256题不参与本轮拟合。','',
        f'开发集：直接答对{a[dev].sum()}/256；全部推理答对{b[dev].sum()}/256；两路理想选择上界{np.maximum(a,b)[dev].sum()}/256。','',
        '下表为预定75%阈值档位；完整曲线见analysis.json。生成量未包含在线读取和控制开销。','',
        '|策略|正确/256|推理调用|生成token|救回|退化|','|---|---:|---:|---:|---:|---:|']
    for r in curves:
        if r['dataset']=='all' and r['fraction']==.75:lines.append(f"|{r['policy']}|{r['correct']}|{r['calls']}|{r['generated_tokens']}|{r['rescues']}|{r['harms']}|")
    lines+=['','本结果属于已观察开发集上的端点统一实验。比较旧8192/Kev门控时，起点、终点和门控训练划分都发生改变，不能把数值差异归因于门控本身。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    for p,hsh in bindings.items():assert e.sha(p)==hsh
    done=dict(UTC=e.now(),state='completed',policies=len(rules),curve_points=len(curves),GPU=False,confirmation_access=False)
    save('completion.json',done);save('status.json',done);print(done,flush=True)


if __name__=='__main__':
    try:
        with threadpool_limits(limits=2):main()
    except BaseException:
        save('status.json',dict(UTC=e.now(),state='failed',error=traceback.format_exc()));raise
