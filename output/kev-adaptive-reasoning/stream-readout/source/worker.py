"""GPU feature replay or head training. One owned GPU, no generation or backbone update."""
import argparse
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path
from common import ROOT, PROJECT, PR, GRID, GPUS, now, read, rows, sha, write, write_rows


def guard():
    uuid=os.environ['CUDA_VISIBLE_DEVICES']
    assert uuid in GPUS.values()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[]
    for line in apps.splitlines():
        a=[s.strip() for s in line.split(',')]
        if a[0]==uuid and int(a[1])!=os.getpid():foreign.append(line)
    assert not foreign,('Another owner on selected GPU',foreign)


def protocol():
    p=read(ROOT/'protocol.json')
    for name,digest in read(ROOT/'source-bindings.json').items():assert sha(PROJECT/name)==digest,name
    for name,digest in p['bindings'].items():assert sha(PROJECT/name)==digest,name
    return p


def setup():
    import torch
    guard()
    torch.set_num_threads(4)
    torch.manual_seed(20261009)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return torch


def extract():
    p=protocol();torch=setup()
    from transformers import AutoModelForImageTextToText
    folder=ROOT/'features';folder.mkdir(exist_ok=True)
    start=time.perf_counter();index=[]
    status=dict(state='loading_model',UTC=now(),pid=os.getpid(),GPU_UUID=os.environ['CUDA_VISIBLE_DEVICES'],completed=0,total=1280)
    write(folder/'status.json',status)
    for name,digest in p['model_files'].items():assert sha(PROJECT/name)==digest,name
    model=AutoModelForImageTextToText.from_pretrained(PROJECT/p['model'],dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',local_files_only=True).eval().requires_grad_(False)
    lm=model.model.language_model
    data=rows(ROOT/'inputs-no-gold.jsonl');samples=rows(ROOT/'samples.jsonl')
    engineering=[];seen_tasks=set()
    forward_seconds=0
    with torch.inference_mode():
        for b,x in zip(samples,data):
            assert b['sample_id']==x['sample_id']
            assert time.perf_counter()-start<p['limits']['feature_wall_seconds']
            if b['index']%8==0:guard()
            file=folder/f"{b['index']:04}.pt"
            if file.exists():
                value=torch.load(file,map_location='cpu',weights_only=True)
                assert value['sample_id']==b['sample_id'] and value['source_sha256']==b['source_sha256']
                elapsed=value['forward_seconds']
                del value
            else:
                ids=x['prompt_ids']+x['generated_ids']
                tensor=torch.tensor([ids],device='cuda')
                torch.cuda.synchronize();st=time.perf_counter()
                h=lm(input_ids=tensor,use_cache=False).last_hidden_state[0]
                torch.cuda.synchronize();elapsed=time.perf_counter()-st
                assert h.shape==(len(ids),2560) and torch.isfinite(h).all()
                # No gold. Verify causal-prefix invariance when future tokens change at fixed layout.
                if b['split']=='train' and b['dataset'] not in seen_tasks:
                    seen_tasks.add(b['dataset'])
                    cut=len(x['prompt_ids'])+min(128,len(x['generated_ids'])//2)
                    changed=tensor.clone();changed[0,cut:]=13
                    future=lm(input_ids=changed,use_cache=False).last_hidden_state[0,:cut]
                    delta=(future.float()-h[:cut].float()).abs().max().item()
                    assert delta<=1e-5,('Future suffix changed prefix features',b['sample_id'],delta)
                    short=lm(input_ids=tensor[:,:cut],use_cache=False).last_hidden_state[0]
                    rel=(short.float()-h[:cut].float()).norm().item()/max(h[:cut].float().norm().item(),1e-12)
                    engineering.append(dict(sample_id=b['sample_id'],future_suffix_max_difference=delta,
                        truncated_layout_relative_l2=rel,cut_tokens=cut,
                        meaning='Causality verified; layout numerical difference measured, not live vLLM equivalence'))
                    write(ROOT/'engineering.json',dict(UTC=now(),rows=engineering,passed=True))
                    del short,future,changed
                length=torch.tensor([len(x['prompt_ids'])+min(k,len(x['generated_ids'])) for k in GRID],device='cuda')
                means=h.float().cumsum(0)[length-1]/length[:,None]
                value=dict(sample_id=b['sample_id'],source_sha256=b['source_sha256'],
                    states=h.cpu(),lengths=length.cpu(),last=h[length-1].cpu(),means=means.cpu(),
                    candidate_count=b['candidate_count'],forward_seconds=elapsed)
                tmp=file.with_suffix('.pending');torch.save(value,tmp);tmp.replace(file)
                del h,value,tensor,means,length
            forward_seconds+=elapsed
            index.append(dict(index=b['index'],sample_id=b['sample_id'],feature_file=file.name,sha256=sha(file)))
            status.update(state='extracting',completed=len(index),UTC=now(),forward_seconds=forward_seconds,wall_seconds=time.perf_counter()-start)
            if len(index)%16==0:write(folder/'status.json',status);print(json.dumps(status),flush=True)
    assert len(index)==1280
    write_rows(folder/'index.jsonl',index)
    status.update(state='completed',UTC=now(),wall_seconds=time.perf_counter()-start,
        new_generated_tokens=0,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write(folder/'status.json',status)


def train():
    p=protocol();torch=setup()
    from heads import Readout, check
    import torch.nn.functional as F
    assert read(ROOT/'features/status.json')['state']=='completed'
    write(ROOT/'head-checks.json',check())
    folder=ROOT/'training';folder.mkdir(exist_ok=True)
    assert not (folder/'completion.json').exists(),'Training already completed'
    start=time.perf_counter()
    samples=rows(ROOT/'samples.jsonl');label={r['sample_id']:r['gold'] for r in rows(ROOT/'labels-observed.jsonl')}
    index=rows(ROOT/'features/index.jsonl');assert len(index)==len(samples)==1280
    sequences=[];last=[];means=[];answer=[];lengths=[]
    for b,f in zip(samples,index):
        if b['index']%64==0:guard()
        file=ROOT/'features'/f['feature_file'];assert sha(file)==f['sha256']
        x=torch.load(file,map_location='cpu',weights_only=True)
        assert x['sample_id']==b['sample_id']
        sequences.append(x['states'].to('cuda'))
        last.append(x['last']);means.append(x['means']);lengths.append(x['lengths'])
        old=PROJECT/b['answer_feature'];assert sha(old)==b['answer_feature_sha256']
        a=torch.load(old,map_location='cpu',weights_only=True)
        assert a['sample_id']==b['sample_id'];answer.append(a['states'])
    fixed={'stream_last_linear':torch.stack(last).cuda().float(),
           'stream_mean_linear':torch.stack(means).cuda().float(),
           'answer_position_linear':torch.stack(answer).cuda().float()}
    lens=torch.stack(lengths).cuda();counts=torch.tensor([s['candidate_count'] for s in samples],device='cuda')
    labels=torch.tensor([label[s['sample_id']] for s in samples],device='cuda')
    ids={s:[i for i,b in enumerate(samples) if b['split']==s] for s in ('train','cal')}
    assert len(ids['train'])==1024 and len(ids['cal'])==256
    config=p['training'];runs=[];selected={}
    permutations={}
    for seed in config['seeds']:
        generator=torch.Generator().manual_seed(seed+20261009)
        permutations[seed]=[[ids['train'][j] for j in torch.randperm(1024,generator=generator).tolist()] for _ in range(config['epochs'])]

    def predict(model,batch):
        idx=torch.tensor(batch,device='cuda')
        if model.kind=='stream_attention':
            h=torch.nn.utils.rnn.pad_sequence([sequences[i] for i in batch],batch_first=True)
        else:h=fixed[model.kind][idx]
        return model(h,lens[idx],counts[idx])

    @torch.no_grad()
    def evaluate(model):
        model.eval();zs=[]
        for j in range(0,256,config['batch_questions']):zs.append(predict(model,ids['cal'][j:j+config['batch_questions']]))
        z=torch.cat(zs);target=labels[ids['cal'],None].expand(-1,5)
        return dict(nll=F.cross_entropy(z.reshape(-1,4),target.reshape(-1)).item(),
                    correct=(z.argmax(-1)==target).sum(0).tolist(),probabilities=z.softmax(-1).cpu())

    for kind in p['heads']:
        for seed in config['seeds']:
            key=f'{kind}-s{seed}';best=None
            for lr in config['learning_rates']:
                guard();torch.manual_seed(seed);model=Readout(kind).cuda()
                optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=config['weight_decay'])
                torch.cuda.synchronize();run_start=time.perf_counter();history=[]
                for epoch,order in enumerate(permutations[seed],1):
                    assert time.perf_counter()-start<p['limits']['training_wall_seconds']
                    guard();model.train();train_loss=0
                    for j in range(0,1024,config['batch_questions']):
                        batch=order[j:j+config['batch_questions']];optimizer.zero_grad(set_to_none=True)
                        z=predict(model,batch);target=labels[batch,None].expand(-1,5)
                        loss=F.cross_entropy(z.reshape(-1,4),target.reshape(-1))
                        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'])
                        optimizer.step();train_loss+=loss.item()*len(batch)/1024
                    if epoch in config['eval_epochs']:
                        ev=evaluate(model)
                        record=dict(kind=kind,seed=seed,lr=lr,epoch=epoch,train_nll=train_loss,dev_nll=ev['nll'],correct=ev['correct'])
                        history.append(record)
                        if best is None or ev['nll']<best['dev_nll']:
                            best=dict(**record,parameters=sum(x.numel() for x in model.parameters()))
                            torch.save(dict(**best,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                sample_ids=[samples[i]['sample_id'] for i in ids['cal']],probabilities=ev['probabilities']),folder/(key+'.pt'))
                    write(folder/'status.json',dict(state='training',UTC=now(),kind=kind,seed=seed,lr=lr,epoch=epoch,completed_runs=len(runs),total_runs=24))
                torch.cuda.synchronize()
                runs.append(dict(kind=kind,seed=seed,lr=lr,seconds=time.perf_counter()-run_start,history=history))
                write_rows(folder/'runs.jsonl',runs);print(json.dumps(runs[-1]),flush=True)
                del model,optimizer
            selected[key]=best;write(folder/'selected.json',selected)
    summary=[]
    for kind in p['heads']:
        chosen=[selected[f'{kind}-s{s}'] for s in config['seeds']]
        summary.append(dict(kind=kind,parameters=chosen[0]['parameters'],
            mean_accuracy=[statistics.mean(x['correct'][j]/256 for x in chosen) for j in range(5)],
            seed_std=[statistics.stdev(x['correct'][j]/256 for x in chosen) for j in range(5)],
            mean_nll=statistics.mean(x['dev_nll'] for x in chosen)))
    write(ROOT/'analysis.json',dict(UTC=now(),scope=p['scope'],results=summary,selected=selected,
        source_bindings=read(ROOT/'source-bindings.json'),cost_scope='Offline replay and trained heads; not deployed latency or early-stop result'))
    lines=['# 复用生成状态的读取对照','', '冻结原生Qwen3.5-4B；既有训练1024题、已观察开发256题。相同标签、五个前缀、优化与选模规则。以下为三种子均值，非独立测试。','',
        '|读取器|参数数|0|128|256|512|1024|平均NLL|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for a in summary:
        cells='|'.join(f'{100*v:.2f}% ± {100*s:.2f}pp' for v,s in zip(a['mean_accuracy'],a['seed_std']))
        lines.append(f"|{a['kind']}|{a['parameters']}|{cells}|{a['mean_nll']:.4f}|")
    lines+=['','自然状态头只读取当前前缀已有激活；answer_position_linear需要额外闭合与答案提示位置。两者是不同读取接口，头部训练预算匹配。',
        'attention为固定可训练查询的因果聚合；已验证可用常量大小累积量复现批量前缀读数，但尚未证明接入vLLM后的服务耗时或跨后端数值一致。',
        '提取采用HF teacher-forced重放，不是从在线vLLM直接取状态；工程检查单列未来后缀不变性与布局数值差异。',
        '本阶段未训练停止或调度策略；下一阶段须冻结读取器后再比较简单阈值、继续收益预测及触发式检查。']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    completion=dict(state='completed',UTC=now(),runs=24,selected=12,wall_seconds=time.perf_counter()-start,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),new_generated_tokens=0)
    write(folder/'completion.json',completion);write(folder/'status.json',completion)


def sentences():
    p=protocol();torch=setup()
    from transformers import AutoTokenizer
    from heads import Readout
    from sentences import points
    assert read(ROOT/'training/completion.json')['state']=='completed'
    tok=AutoTokenizer.from_pretrained(PROJECT/p['model'],local_files_only=True)
    samples=rows(ROOT/'samples.jsonl');inputs=rows(ROOT/'inputs-no-gold.jsonl')
    gold={r['sample_id']:r['gold'] for r in rows(ROOT/'labels-observed.jsonl')}
    folder=ROOT/'sentence-evaluation';folder.mkdir(exist_ok=True)
    models={}
    for kind in ('stream_last_linear','stream_mean_linear','stream_attention'):
        saved=torch.load(ROOT/'training'/f'{kind}-s29.pt',map_location='cpu',weights_only=True)
        head=Readout(kind).cuda();head.load_state_dict(saved['state_dict']);head.eval()
        models[kind]=head
    out=[];start=time.perf_counter();skipped=0;completed=0
    with torch.inference_mode():
        for b,x in zip(samples,inputs):
            if b['split']!='cal':continue
            if completed%16==0:guard()
            assert time.perf_counter()-start<p['limits']['training_wall_seconds']
            ends,omitted=points(tok,x['generated_ids']);skipped+=omitted
            by_k={r['prefix_tokens']:dict(**r,roles=['sentence']) for r in ends}
            for k,role in [(0,'initial'),(len(x['generated_ids']),'terminal')]:
                if k in by_k:by_k[k]['roles'].append(role)
                else:by_k[k]=dict(prefix_tokens=k,observable_after_tokens=k,roles=[role])
            ordered=[by_k[k] for k in sorted(by_k)]
            f=torch.load(ROOT/'features'/f"{b['index']:04}.pt",map_location='cpu',weights_only=True)
            h=f['states'].cuda();length=torch.tensor([[b['prompt_tokens']+r['prefix_tokens'] for r in ordered]],device='cuda')
            count=torch.tensor([b['candidate_count']],device='cuda')
            representations={'stream_last_linear':h[length[0]-1][None],
                'stream_mean_linear':(h.float().cumsum(0)[length[0]-1]/length[0,:,None])[None],
                'stream_attention':h[None]}
            for kind,head in models.items():
                probs=head(representations[kind],length,count).softmax(-1)[0].cpu().tolist()
                entries=[]
                for event,ps in zip(ordered,probs):
                    pred=max(range(len(ps)),key=ps.__getitem__)
                    entries.append(dict(**event,prediction=pred,probabilities=ps,correct=pred==gold[b['sample_id']]))
                out.append(dict(sample_id=b['sample_id'],dataset=b['dataset'],component_id=b['component_id'],
                    reader=kind,seed=29,gold=gold[b['sample_id']],points=entries))
            completed+=1
            if completed%16==0:
                write(folder/'status.json',dict(state='evaluating',UTC=now(),completed=completed,total=256))
    assert completed==256 and len(out)==768
    write_rows(folder/'trajectories.jsonl',out)
    summary={}
    for kind in models:
        a=[r for r in out if r['reader']==kind]
        summary[kind]=dict(questions=len(a),initial_correct=sum(r['points'][0]['correct'] for r in a),
            terminal_correct=sum(r['points'][-1]['correct'] for r in a),
            oracle_ever_correct=sum(any(q['correct'] for q in r['points']) for r in a),
            oracle_stable_by256=sum(any(q['prefix_tokens']<=256 and all(z['correct'] for z in r['points'][i:])
                                       for i,q in enumerate(r['points'])) for r in a),
            average_observed_sentence_points=statistics.mean(sum('sentence' in q['roles'] for q in r['points']) for r in a))
    write(folder/'analysis.json',dict(UTC=now(),summary=summary,
        skipped_sentence_boundaries_inside_tokens=skipped,
        scope='Observed dev256; seed29 predeclared diagnostic; no threshold selection or online speedup',
        terminal='All actual terminal tokens for every head; intermediate reads end at an exact complete-sentence token prefix',
        causal='Sentence boundary must be observable; recorded observable_after_tokens may exceed the read prefix by whitespace lookahead'))
    done=dict(state='completed',UTC=now(),questions=256,reader_trajectories=768,wall_seconds=time.perf_counter()-start)
    write(folder/'completion.json',done);write(folder/'status.json',done)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['extract','train','sentences']);a=parser.parse_args()
    try:
        {'extract':extract,'train':train,'sentences':sentences}[a.action]()
    except BaseException:
        write(ROOT/(a.action+'-failure.json'),dict(UTC=now(),pid=os.getpid(),traceback=traceback.format_exc()))
        raise
