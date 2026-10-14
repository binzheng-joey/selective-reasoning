"""Frozen GPU2 confirmation collection. No gold reader or scoring command.

Freeze before question projection. Preserve every predetermined sample. Write
predictions to sealed storage; only technical progress is exposed in status.
"""
import argparse
import collections
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback

import arr_gpu_queue as rt
import arr_budget_baselines as budget

MAIN = budget.MAIN
OUT = MAIN/'formal-confirmation'
SOURCE = Path(__file__).resolve()
OFFICIAL = rt.ROOT.parent/'official-gate-data-v1'
OLD = rt.ROOT.parent/'gate-utility-expanded1024-riskcost-20261005-v1'
MANIFEST = rt.QUEUE/'data-audit/confirmation-manifest.jsonl'
ARC_META = budget.ARC/'test-metadata.json'
CORE = ['direct', 'full-512', 'full-1024', 'full-2048', 'simple-margin-every8-1024',
        'ours-entropy70-1024', 'ours-entropy85-1024', 'ours-entropy95-1024']
BASELINES = ['forced-terminal-prefill-1024', 'answer-consistency-k10-prefill-1024',
             'deer-qwen-adapted-prefill-1024', 'deer-qwen-adapted-prefill-2048', 'native-8192']


def rank(label):
    return hashlib.sha256(('arr-formal-20261010|'+label).encode()).hexdigest()


def rows(path):
    return budget.rows(path)


def jobs_for(samples, methods, repeat_ids):
    jobs = []
    for rep in (0, 1, 2):
        chosen = samples if rep == 0 else [s for s in samples if s['sample_id'] in set(repeat_ids)]
        for sample in chosen:
            sid = sample['sample_id']
            order = list(methods)
            random.Random(int(rank(f'order/{rep}/{sid}')[:16],16)).shuffle(order)
            for method in order:
                jobs.append(dict(case_id=f'r{rep}-{sample["execution_index"]:04}-{method}',
                                 sample_id=sid, policy=method, repeat=rep))
    return jobs


def freeze():
    if (OUT/'protocol.json').exists():
        return verify()
    bp = budget.verify()
    stability = rt.read(MAIN/'stability-check/analysis.json')
    assert rt.read(MAIN/'stability-check/completion.json')['cases'] == 64
    assert not (MAIN/'stability-check/failure.json').exists()
    assert all(x['prediction_disagreements']==x['generation_disagreements']==0 for x in stability['summaries'])
    assert stability['readout']['own_prefix']['maximum_probability_drift']==0
    assert rt.sha(MANIFEST)==rt.read(rt.QUEUE/'data-audit/audit.json')['confirmation_manifest_sha256']
    pwlq = rows(MANIFEST)
    assert len(pwlq)==len({s['component_id'] for s in pwlq})==512
    assert collections.Counter(s['dataset'] for s in pwlq)=={'proofwriter':256,'logiqa2':256}
    arc = rt.read(ARC_META)
    arc_ids = set(arc['selected_ids'])
    assert len(arc_ids)==256
    samples = copy.deepcopy(pwlq)
    samples += [dict(s, component_id=s['question_sha256'], official_split='test',
                     internal_split='arr_confirmation', stratum='ARC')
                for s in arc['records'] if s['sample_id'] in arc_ids]
    assert len(samples)==len({s['sample_id'] for s in samples})==768
    assert collections.Counter(s['candidate_count'] for s in samples if s['dataset']=='arc_challenge')=={4:255,5:1}
    # Fresh audit against every already-used input in this workstream.
    import sentence_online as base
    observed = rows(base.STREAM/'samples.jsonl')+rows(budget.OUT/'inputs-no-gold.jsonl')
    assert not ({s['sample_id'] for s in samples}&{s['sample_id'] for s in observed})
    assert not ({s['component_id'] for s in samples}&{s['component_id'] for s in observed})
    groups = {task:sorted([s for s in samples if s['dataset']==task],key=lambda s:rank('execution/'+s['sample_id']))
              for task in ('proofwriter','logiqa2','arc_challenge')}
    ordered = [dict(groups[task][i], execution_index=3*i+j)
               for i in range(256) for j,task in enumerate(groups)]
    repeat_ids = []
    # Predetermine 48 timing subjects; PW four strata contribute four each.
    for stratum in ('AttNeg','AttNoneg','RelNeg','RelNoneg'):
        repeat_ids += [s['sample_id'] for s in sorted([s for s in ordered if s['dataset']=='proofwriter' and s['stratum']==stratum],key=lambda s:rank('repeat/'+s['sample_id']))[:4]]
    for task in ('logiqa2','arc_challenge'):
        repeat_ids += [s['sample_id'] for s in sorted(groups[task],key=lambda s:rank('repeat/'+s['sample_id']))[:16]]
    assert len(set(repeat_ids))==48
    selected = [copy.deepcopy(next(p for p in bp['policies'] if p['id']==m)) for m in CORE+BASELINES]
    OUT.mkdir(parents=True,exist_ok=True)
    rt.save(OUT/'metadata.json',ordered)
    jobs = {stage:jobs_for(ordered,ms,repeat_ids) for stage,ms in [('core',CORE),('baselines',BASELINES)]}
    assert len(jobs['core'])==6912 and len(jobs['baselines'])==4320
    rt.save(OUT/'jobs.json',jobs)
    review = dict(UTC=rt.now(), proceed_with_gpu2=True, diagnostic_questions=8,
        within_gpu2_restart_prediction_changes=0, own_prefix_probability_drift=0,
        historical_fixed_prefix_changes=stability['readout']['fixed_historical_prefix'],
        boundary='Historical cross-run differences remain; use current single-GPU2 environment for all formal comparisons. Diagnostic does not establish universal determinism.',
        decisions='Ours0.70 is the preselected representative. Retain0.85/0.95 as frozen sensitivity points; retain the stronger simple comparator. No ARC-specific refit.',
        gold_access=False)
    rt.save(OUT/'development-review.json',review)
    sources = [SOURCE, Path(budget.__file__), MANIFEST, ARC_META, OUT/'metadata.json', OUT/'jobs.json',
               OUT/'development-review.json', MAIN/'stability-check/analysis.json', budget.OUT/'protocol.json',
               budget.OUT/'analysis.json', budget.PILOT/'gates.json', budget.TASKLESS, Path(bp['head']),
               OFFICIAL/'download-manifest.json', rt.QUEUE/'data-audit/audit.json',
               OLD/'source/selected_train_reader.py', OLD/'source/six_scheme_contract_cpu.py']
    sources += [rt.ROOT/'source'/n for n in ['arr_variable_choice.py','arr_repaired_final.py','arr_native_final.py',
                'arr_prior_baselines.py','arr_integrated_online.py','arr_control.py','sentence_online.py','arr_gpu_queue.py',
                'arr_main_table_prepare.py']]
    sources += [MAIN/'stability-check/inputs-no-gold.jsonl', rt.QUEUE/'data-audit/protocol.json',
                budget.ARC/'development-exclusions.json']
    raw = {x['path']:x['raw_sha256'] for x in rt.read(OFFICIAL/'download-manifest.json')
           if x['official_split']=='train'}
    arc_raw = budget.ARC/'raw/test-00000-of-00001.parquet'; raw[str(arc_raw)]=arc['source_sha256']
    model_manifest=rt.read(rt.ROOT.parent/'prefix-readout/confirmation/model-integrity.json')
    model_files={str(Path(bp['engine']['model'])/n):h for n,h in model_manifest['files'].items()}
    p = dict(UTC=rt.now(), state='frozen_before_confirmation_projection', gpu=2,GPU_UUID=rt.GPUS[2],
        engine=bp['engine'],head=bp['head'],policies=selected,core_methods=CORE,baseline_methods=BASELINES,
        questions=768,per_task=256,core_primary_cases=6144,core_repeat_cases=768,core_total_cases=6912,
        baseline_primary_cases=3840,baseline_repeat_cases=480,repeat_ids=repeat_ids,
        primary_representative='ours-entropy70-1024',
        scope='PW/LQ project-independent official-train material groups; ARC official-test subset. No pretraining-unseen claim.',
        task_adaptation='Same gate/head/thresholds across tasks. ARC allows scientific knowledge; other prompts unchanged. >4 choices use native candidate logits at same answer position.',
        reader='Qwen3.5-4B post-trained generator plus trained answer-position linear head; this is not unmodified Kev.',
        interpretation='1024/2048 are bounded reasoning; natural8192 truncation and invalid finals remain in denominator.',
        timing='Synchronized policy wall includes initial read, routing, generation, trigger, reads and cancellation. Excludes tokenize, loading, warmup, cache reset, guards and writes; not HTTP service latency.',
        metrics=['accuracy','mean_pipeline_seconds','physical_output_tokens','raw_input_and_cached_tokens','full_reads','trigger_calls','early_stop_rate','reasoning_rate','invalid_final_rate','truncation_rate'],
        repeat_protocol='All768 first runs provide primary accuracy/time. Fixed48 subjects get two further executions after the main pass, for timing/stability only; no voting or best-repeat selection.',
        statistics='Task-wise and task-equal aggregates; 10000 paired material-group bootstrap draws stratified by PW subtype/LQ/ARC; seed20261010, pointwise95% intervals; no accuracy-loss tolerance.',
        scored_access='Outputs sealed until the remaining baseline/analysis configurations are frozen. No scorer in this worker; no gold decoding, no inspecting predictions for method selection.',
        pending='Attention probe is not complete: layer selection/online deployment must use development-only data. No claim of completed closest-prior reproduction. Core collection does not depend on it.',
        execution='GPU2 exclusively. Deterministically interleaved tasks and within-question methods. Persist atomic cases; resume only missing verified cases. No automatic cross-GPU migration.',
        wall_limit_seconds=86400,
        bindings={str(f):rt.sha(f) for f in sources},raw_sources=raw,model_files=model_files)
    for name,digest in {**raw,**model_files}.items():
        assert rt.sha(name)==digest,('Raw/model changed',name)
    rt.save(OUT/'protocol.json',p)
    print(json.dumps(dict(frozen=True,questions=768,core_configurations=8,core_primary_cases=6144,core_repeat_cases=768,new_gold_decodes=0)))
    return p


def verify(model=False):
    p=rt.read(OUT/'protocol.json')
    for name,digest in p['bindings'].items():
        assert rt.sha(name)==digest,('Changed formal binding',name)
    budget.verify()
    if model:
        for name,digest in p['model_files'].items():
            assert rt.sha(name)==digest,('Changed model',name)
    return p


def render(tok, row):
    if row['dataset']=='arc_challenge':
        system='Use the supplied question and relevant scientific knowledge. Keep the original candidate order. Follow the requested output format.'
        user='Material:\n'+row['state']+'\n\nTask:\nSelect the candidate that best answers this science question.\n\nCandidates (in the given order):\n'
        user+='\n'.join(f'{i+1}. {v}' for i,v in enumerate(row['options']))
        user+='\n\nReason through the question using the supplied information and relevant scientific knowledge. Then give exactly one final candidate index from 1 to '+str(len(row['options']))+' in the exact format <FINAL>index</FINAL>. Place this final tag after your reasoning, outside any thinking block. Do not append text after the tag.'
    else:
        from six_scheme_contract_cpu import R_SYSTEM,F_prompt
        system,user=R_SYSTEM,F_prompt(row)
    return tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=user)],
                                   tokenize=True,add_generation_prompt=True,enable_thinking=True)


def project():
    p=verify()
    assert not (OUT/'projection.json').exists(),'Projection already frozen'
    sys.path.insert(0,str(OLD/'source'));sys.path.insert(0,str(budget.ARC/'deps'))
    from selected_train_reader import extract
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    metadata=rt.read(OUT/'metadata.json')
    needed=collections.defaultdict(lambda:collections.defaultdict(list))
    for m in metadata:
        if m['dataset']!='arc_challenge':needed[m['source_path']][m['source_line']].append(m)
    data={}
    for name,lines in needed.items():
        assert rt.sha(name)==p['raw_sources'][name]
        with Path(name).open() as f:
            for ln,text in enumerate(f,1):
                for m in lines.get(ln,[]):
                    q=extract(text,m);data[q['id']]=q
                if ln>=max(lines):break
    arc_meta={s['sample_id'].removeprefix('arc:'):s for s in metadata if s['dataset']=='arc_challenge'}
    arc_raw=budget.ARC/'raw/test-00000-of-00001.parquet'
    assert rt.sha(arc_raw)==p['raw_sources'][str(arc_raw)]
    raw=pq.read_table(arc_raw,columns=['id','question','choices'],filters=[('id','in',sorted(arc_meta))])
    from arr_main_table_prepare import arc_metadata
    for x in raw.to_pylist():
        m=arc_meta[x['id']];options=x['choices']['text']
        current=arc_metadata(x)
        assert current['question_sha256']==m['question_sha256']
        assert current['options_sha256']==m['options_sha256'],'ARC option hash mismatch'
        assert x['choices']['label']==m['source_labels'] and len(options)==m['candidate_count']
        data[m['sample_id']]=dict(id=m['sample_id'],dataset='arc_challenge',state=x['question'],
                                options=options,component_id=m['component_id'])
    assert len(data)==768
    tok=AutoTokenizer.from_pretrained(p['engine']['model'],local_files_only=True)
    encoded=[]
    for m in metadata:
        q=data[m['sample_id']];ids=render(tok,q)
        assert len(ids)+8192+128<p['engine']['max_model_len'],'Cannot silently truncate selected question'
        encoded.append(dict(sample_id=m['sample_id'],dataset=m['dataset'],component_id=m['component_id'],
                            stratum=m['stratum'],execution_index=m['execution_index'],candidate_count=len(q['options']),prompt_ids=ids))
    path=OUT/'inputs-no-gold.jsonl';tmp=path.with_suffix('.pending')
    tmp.write_text(''.join(json.dumps(s,ensure_ascii=False)+'\n' for s in encoded));tmp.replace(path)
    report=dict(UTC=rt.now(),questions=768,new_gold_decodes=0,protocol_sha256=rt.sha(OUT/'protocol.json'),
                inputs_sha256=rt.sha(path),max_prompt_tokens=max(len(s['prompt_ids']) for s in encoded),
                projection='PW/LQ selected-field parser skips label substrings; ARC only id/question/choices Parquet columns.',
                candidate_counts=dict(collections.Counter(s['candidate_count'] for s in encoded)))
    rt.save(OUT/'projection.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('inputs_sha256','protocol_sha256')}))


def verify_inputs():
    p=verify()
    projection=rt.read(OUT/'projection.json')
    assert projection['protocol_sha256']==rt.sha(OUT/'protocol.json')
    assert rt.sha(OUT/'inputs-no-gold.jsonl')==projection['inputs_sha256']
    return p,projection


def row_checks(row,sample,spec):
    assert row['sample_id']==sample['sample_id'] and row['policy']==spec['id']
    assert row['policy_spec']==spec
    assert math.isfinite(row['pipeline_seconds']) and row['pipeline_seconds']>0
    assert row['prediction'] is None or 0<=row['prediction']<sample['candidate_count']
    assert len(row['generated_ids'])<=spec['cap']
    if spec['kind']=='head':
        assert row['prediction'] is not None
        probs=row['probabilities'];assert len(probs)==sample['candidate_count'] and abs(sum(probs)-1)<1e-5
        assert row['prediction']==probs.index(max(probs))
        assert row['physical_output_tokens']==len(row['generated_ids'])+row['full_reads']
        assert all(0<=q['prefix_tokens']<=len(row['generated_ids']) for q in row['checks'])


class Contention(RuntimeError):pass


def guard():
    if rt.stop_requested() or (OUT/'STOP').exists():raise InterruptedError('Stop requested')
    foreign=[pid for pid in rt.inventory()[2]['pids'] if pid!=os.getpid()]
    if foreign:raise Contention('GPU2 occupied by '+str(foreign))


def existing_rows(stage,jobs,p,inputs):
    folder=OUT/'sealed'/stage
    by={j['case_id']:j for j in jobs};specs={s['id']:s for s in p['policies']}
    done=set()
    for f in folder.glob('*.json'):
        assert f.stem in by,('Unexpected existing case',f)
        r=rt.read(f);j=by[f.stem]
        assert r['protocol_sha256']==rt.sha(OUT/'protocol.json') and r['GPU_UUID']==rt.GPUS[2]
        assert r['repeat']==j['repeat'] and r['case_id']==j['case_id']
        row_checks(r,inputs[j['sample_id']],specs[j['policy']]);done.add(f.stem)
    return done


def worker(stage):
    p,projection=verify_inputs();verify(model=True)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='2'
    lock=(rt.QUEUE/'gpu2.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Worker interrupted')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Formal phase wall limit')))
    signal.alarm(p['wall_limit_seconds'])
    jobs=rt.read(OUT/'jobs.json')[stage];inputs={s['sample_id']:s for s in rows(OUT/'inputs-no-gold.jsonl')}
    done=existing_rows(stage,jobs,p,inputs);runner=None;attempt=f'{time.time_ns()}-{os.getpid()}'
    statefile=OUT/f'{stage}-status.json'
    try:
        guard();rt.save(statefile,dict(UTC=rt.now(),state='loading',gpu=2,pid=os.getpid(),completed=len(done),total=len(jobs),scored=False))
        import arr_prior_baselines as priors
        from arr_variable_choice import Engine
        priors.parse_answer=budget.parse_prior_answer
        runner=Engine(p);runner.tr=rt.read(budget.TASKLESS)
        runner.trigger_weight=runner.torch.tensor(runner.tr['hidden_weight'],dtype=runner.torch.float32,device='cuda')
        gates=rt.read(budget.PILOT/'gates.json');specs={s['id']:s for s in p['policies']}
        # Existing observed warm-up only, same entry in every attempt.
        warm=rows(MAIN/'stability-check/inputs-no-gold.jsonl')[-3]
        runner.query(runner.read_ids(warm['prompt_ids'],[]),warm['candidate_count'])
        rt.save(OUT/'attempts'/f'{attempt}-environment.json',dict(UTC=rt.now(),gpu=2,pid=os.getpid(),
                torch=runner.torch.__version__,cuda=runner.torch.version.cuda,stage=stage,
                completed_before=len(done),protocol_sha256=rt.sha(OUT/'protocol.json'),
                model_verified=True,warmup_sample=warm['sample_id']))
        for job in jobs:
            if job['case_id'] in done:continue
            guard();s=inputs[job['sample_id']];q=specs[job['policy']]
            rt.save(statefile,dict(UTC=rt.now(),state='running',stage=stage,gpu=2,pid=os.getpid(),
                    completed=len(done),total=len(jobs),repeat=job['repeat'],dataset=s['dataset'],
                    policy=q['id'],case_id=job['case_id'],scored=False))
            if q['kind']=='head':r=runner.integrated(s,q,gates)
            elif q['kind']=='native_final':r=runner.native_final(s,q)
            else:r=runner.prior(s,q,gates)
            record=dict(**r,repeat=job['repeat'],case_id=job['case_id'],attempt=attempt,UTC=rt.now(),
                        GPU_UUID=rt.GPUS[2],protocol_sha256=rt.sha(OUT/'protocol.json'),
                        inputs_sha256=projection['inputs_sha256'])
            try:guard()
            except Contention:
                rt.save(OUT/'excluded'/stage/f'{attempt}-{job["case_id"]}.json',record)
                raise
            row_checks(r,s,q)
            rt.save(OUT/'sealed'/stage/(job['case_id']+'.json'),record);done.add(job['case_id'])
        assert len(done)==len(jobs)
        result=dict(UTC=rt.now(),state='completed_sealed',gpu=2,cases=len(done),primary_cases=sum(j['repeat']==0 for j in jobs),
                    repeat_cases=sum(j['repeat']!=0 for j in jobs),gold_access=False,scored=False)
        rt.save(OUT/f'{stage}-completion.json',result);rt.save(statefile,result)
    except (Contention,InterruptedError) as exc:
        result=dict(UTC=rt.now(),state='interrupted',reason=str(exc),stage=stage,completed=len(done),total=len(jobs),attempt=attempt)
        rt.save(OUT/'attempts'/f'{attempt}-interrupted.json',result);rt.save(statefile,result)
        raise SystemExit(75)
    except BaseException:
        result=dict(UTC=rt.now(),state='failed',stage=stage,completed=len(done),total=len(jobs),attempt=attempt,error=traceback.format_exc())
        rt.save(OUT/'attempts'/f'{attempt}-failure.json',result);rt.save(statefile,result)
        raise
    finally:
        if runner:runner.close()


def launch(stage,resume=False):
    p,_=verify_inputs()
    with (OUT/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/f'{stage}-completion.json').exists():raise RuntimeError('Stage already complete')
        old=OUT/f'{stage}-launcher.json'
        if old.exists():
            assert resume,'Existing attempt: use explicit resume after inspection'
            assert rt.read(OUT/f'{stage}-status.json')['state'] in ('interrupted','failed'),'Active or ambiguous worker'
            rt.save(OUT/'attempts'/f'launcher-before-resume-{time.time_ns()}.json',rt.read(old))
        for _ in range(2):
            assert rt.is_idle(rt.inventory()[2],dict(idle_memory_mib=1024,idle_utilization_percent=5)),'GPU2 occupied'
            time.sleep(2)
        cmd=[str(rt.PYTHON),'-B',str(SOURCE),'--worker','--stage',stage]
        with (OUT/f'{stage}-worker.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=rt.PROJECT,env=rt.env_for(2),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        rt.save(old,dict(UTC=rt.now(),pid=child.pid,gpu=2,stage=stage,command=cmd,protocol_sha256=rt.sha(OUT/'protocol.json')))
        print(json.dumps(dict(started=True,gpu=2,pid=child.pid,stage=stage,cases=len(rt.read(OUT/'jobs.json')[stage]),scored=False)))


if __name__=='__main__':
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True)
    for k in ('freeze','project','check','launch','worker'):g.add_argument('--'+k,action='store_true')
    ap.add_argument('--stage',choices=('core','baselines'),default='core');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.freeze:freeze()
    elif a.project:project()
    elif a.check:verify_inputs();print('Formal protocol and projected inputs verified; no labels accessed')
    elif a.launch:launch(a.stage,a.resume)
    else:worker(a.stage)
