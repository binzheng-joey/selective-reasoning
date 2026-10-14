"""Same-GPU vLLM generation and answer requests, latest-prefix scheduling.

No gold access. Each policy generates its own natural trajectory. Two live
requests share one engine; async here means the generator remains scheduled
while the answer request runs, not simultaneous CUDA kernels.
"""
import argparse
import bisect
import collections
import fcntl
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

PROJECT=Path(__file__).resolve().parents[4]
ROOT=PROJECT/'output/kev-adaptive-reasoning/decision-improvements'
STREAM=ROOT.parent/'stream-readout'
OUT=ROOT/'sentence-online'
sys.path.insert(0,str(STREAM/'source'))
from common import now,read,rows,sha,write,MODEL,GPUS
sys.path.insert(0,str(ROOT.parent/'sentence-async-readers-v1/source'))
sys.path.insert(0,str(ROOT.parent/'sentence-async-readers-v1/sparse-checks/source'))
from sentence_v2 import StreamDetector

POLICIES=['terminal','every8','learned_gap4','learned_tick4','learned_tick8']


def certainty(p):
    return 1+sum(v*math.log(v) for v in p if v>0)/math.log(len(p))


def guard(gpu):
    assert os.environ['CUDA_VISIBLE_DEVICES']==str(gpu)
    assert subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==GPUS[gpu]
    assert not (OUT/'STOP').exists()
    text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    other=[s for s in text.splitlines() if s.split(',')[0].strip()==GPUS[gpu] and int(s.split(',')[1])!=os.getpid()]
    assert not other,('Another GPU owner appeared',other)


class Engine:
    def __init__(self,protocol):
        import torch
        from vllm import LLM,SamplingParams
        self.torch=torch;self.SP=SamplingParams
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
        start=time.perf_counter();self.llm=LLM(**protocol['engine'])
        self.engine=self.llm.llm_engine;self.tok=self.llm.get_tokenizer()
        assert type(self.engine.engine_core).__name__=='InprocClient'
        self.runner=self.engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
        self.weight=torch.load(protocol['head'],map_location='cpu',weights_only=True)['state_dict']['linear.weight'].cuda().float()
        tr=read(OUT/'trigger.json');self.tr=tr
        self.trigger_weight=torch.tensor(tr['hidden_weight'],dtype=torch.float32,device='cuda')
        self.registry={};self.hidden={};self.captured={};self.hook_calls=0;self.hook_host_seconds=0.
        self.serial=0;self.keep_natural=False
        self.original=self.runner.model.compute_logits
        self.runner.model.compute_logits=self.capture
        torch.cuda.synchronize();self.startup_seconds=time.perf_counter()-start

    def capture(self,h):
        logits=self.original(h)
        start=time.perf_counter()
        batch=self.runner.input_batch
        assert h.ndim==2 and h.shape[0]==len(batch.req_ids) and h.shape[1]==2560
        qloc=self.runner.query_start_loc.cpu.numpy()
        for i,rid in enumerate(batch.req_ids):
            info=self.registry.get(rid)
            if info is None:continue
            pos=int(batch.num_computed_tokens_cpu[i])+int(qloc[i+1]-qloc[i])-1
            if info['kind']=='read' and pos==info['length']-1:
                assert rid not in self.captured
                p=(h[i].float()@self.weight[:info['count']].T).softmax(-1)
                self.captured[rid]=dict(probabilities=p.cpu().tolist(),native_argmax=int(logits[i].argmax()),position=pos)
            elif info['kind']=='generate' and self.keep_natural:
                k=pos-info['length']+1
                if k>=0:
                    self.hidden[k]=h[i].detach().clone()
                    for old in list(self.hidden):
                        if old<k-64:del self.hidden[old]
        self.hook_calls+=1;self.hook_host_seconds+=time.perf_counter()-start
        return logits

    def add(self,ids,cap,kind,count):
        self.serial+=1;key=f'online-{self.serial}'
        params=self.SP(temperature=0.,top_p=1.,top_k=-1,max_tokens=cap,seed=20261009,
            detokenize=False,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.)
        internal=self.engine.add_request(key,dict(prompt_token_ids=ids),params)
        self.registry[internal]=dict(kind=kind,length=len(ids),count=count)
        return key,internal

    def read_ids(self,prompt,generated):
        end=self.tok.convert_tokens_to_ids('</think>')
        suffix=self.tok.encode('\n\n<FINAL>' if end in generated else '\n</think>\n\n<FINAL>',add_special_tokens=False)
        return prompt+generated+suffix

    def query(self,ids,count):
        assert not self.engine.has_unfinished_requests()
        key,rid=self.add(ids,1,'read',count);self.torch.cuda.synchronize();start=time.perf_counter()
        out=None
        while self.engine.has_unfinished_requests():
            for o in self.engine.step():
                if o.request_id==key:out=o
        self.torch.cuda.synchronize();q=self.captured.pop(rid)
        assert list(out.outputs[0].token_ids)==[q['native_argmax']]
        q.update(seconds=time.perf_counter()-start,cached_tokens=out.num_cached_tokens or 0,input_tokens=len(ids))
        self.registry.clear();return q

    def pipeline(self,sample,policy,cap=1024,probe=False):
        torch=self.torch;prompt=sample['prompt_ids'];count=sample['candidate_count']
        assert not self.engine.has_unfinished_requests()
        assert self.llm.reset_prefix_cache(),'Between-policy cache reset failed'
        self.registry.clear();self.captured.clear();self.hidden.clear()
        self.keep_natural=policy.startswith('learned')
        detector=StreamDetector();decoded_lengths=[0];decoded_counts=[0]
        prefix=[];prev_text='';ordinal=0;last_read_ordinal=0;pending=None;inflight=None
        requests=[];checks=[];triggers=[];merged=0;natural=False;stopped=False
        detector_seconds=0.;trigger_seconds=0.;read_at=None;decision_at=None;abort_seconds=0.
        hook_start=self.hook_host_seconds;query_tokens=0
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
        genkey,genrid=self.add(prompt,cap,'generate',count);generation_finished=False
        loops=0
        while True:
            loops+=1;assert loops<cap*8+200 and time.perf_counter()-begin<240
            outputs=self.engine.step() if self.engine.has_unfinished_requests() else []
            for o in sorted(outputs,key=lambda o:o.request_id!=genkey):
                if o.request_id==genkey:
                    new=list(o.outputs[0].token_ids);assert new[:len(prefix)]==prefix
                    # Decode steps emit one token; prefill emits at most one too.
                    for k in range(len(prefix)+1,len(new)+1):
                        text=self.tok.decode(new[:k],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                        if text.endswith('\ufffd'):continue
                        assert text.startswith(prev_text)
                        decoded_lengths.append(len(text));decoded_counts.append(k)
                        t=time.perf_counter();ends=detector.update(text)
                        for end in ends:
                            at=bisect.bisect_left(decoded_lengths,end)
                            pk=decoded_counts[at]
                            if text[end:decoded_lengths[at]].strip():continue
                            ordinal+=1
                            ev=dict(prefix_tokens=pk,observable_tokens=k,ordinal=ordinal)
                            if pending is not None:merged+=1
                            pending=ev
                        detector_seconds+=time.perf_counter()-t;prev_text=text
                    prefix=new;generation_finished=o.finished
                    if o.finished:natural=o.outputs[0].finish_reason!='length'
                elif inflight is not None and o.request_id==inflight['key']:
                    if not o.finished:continue
                    q=self.captured.pop(inflight['rid'])
                    assert list(o.outputs[0].token_ids)==[q['native_argmax']]
                    q.update(**{k:v for k,v in inflight.items() if k not in ['key','rid']},
                        cached_tokens=o.num_cached_tokens or 0,completed_at_tokens=len(prefix),
                        response_seconds=time.perf_counter()-inflight['start'])
                    checks.append(q);query_tokens+=1
                    self.registry.pop(inflight['rid']);inflight=None
                    if q['terminal'] or (not probe and certainty(q['probabilities'])>=.85):
                        stopped=not q['terminal'];read_at=q['prefix_tokens'];decision_at=len(prefix)
                        torch.cuda.synchronize();ts=time.perf_counter()
                        self.engine.abort_request([genkey]);torch.cuda.synchronize()
                        abort_seconds=time.perf_counter()-ts
                        break
            if stopped or (checks and checks[-1]['terminal']):break
            if probe:
                pending=(dict(prefix_tokens=64,observable_tokens=len(prefix),ordinal=1)
                    if len(prefix)>=64 and not checks and inflight is None else None)
            if inflight is None and pending is not None and not generation_finished:
                ev=pending;pk=ev['prefix_tokens'];ord_=ev['ordinal']
                if policy=='terminal':pending=None;perform=False
                elif policy=='every8':perform=ord_-last_read_ordinal>=8;pending=None
                elif probe:perform=True;pending=None
                else:
                    eligible=(ord_-last_read_ordinal>=4 if policy=='learned_gap4' else ord_%int(policy[-1])==0)
                    if not eligible:pending=None;perform=False
                    elif pk not in self.hidden:perform=False
                    else:
                        ts=time.perf_counter();tr=self.tr
                        value=float(self.hidden[pk].float()@self.trigger_weight)+tr['position_weight']*pk/1024+tr['task_weight']*float(sample['dataset']=='logiqa2')+tr['bias']
                        score=1/(1+math.exp(-max(-80,min(80,value))))
                        trigger_seconds+=time.perf_counter()-ts
                        triggers.append(dict(**ev,available_tokens=len(prefix),score=score))
                        perform=score>=.25;pending=None
                if perform:
                    ids=self.read_ids(prompt,prefix[:pk]);key,rid=self.add(ids,1,'read',count)
                    inflight=dict(key=key,rid=rid,prefix_tokens=pk,ordinal=ord_,terminal=False,
                        input_tokens=len(ids),submitted_at_tokens=len(prefix),start=time.perf_counter())
                    last_read_ordinal=ord_
            if generation_finished and inflight is None:
                ids=self.read_ids(prompt,prefix);key,rid=self.add(ids,1,'read',count)
                inflight=dict(key=key,rid=rid,prefix_tokens=len(prefix),ordinal=ordinal,terminal=True,
                    input_tokens=len(ids),submitted_at_tokens=len(prefix),start=time.perf_counter())
            assert self.engine.has_unfinished_requests(),('No active request',policy,len(prefix))
        if self.engine.has_unfinished_requests():
            self.engine.abort_request([genkey])
        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
        assert not self.engine.has_unfinished_requests()
        assert checks and len(prefix)<=cap
        p=checks[-1]['probabilities'];self.keep_natural=False
        result=dict(sample_id=sample['sample_id'],dataset=sample['dataset'],policy=policy,
            prediction=max(range(count),key=p.__getitem__),probabilities=p,generated_ids=prefix,
            pipeline_seconds=elapsed,generation_tokens=len(prefix),physical_output_tokens=len(prefix)+query_tokens,
            full_reads=len(checks),trigger_calls=len(triggers),read_prefix_tokens=read_at,
            decision_at_tokens=decision_at,extra_tokens_since_read_prefix=len(prefix)-read_at,
            tokens_after_stop_decision=len(prefix)-decision_at,early_stop=stopped,natural_terminal=natural,
            merged_pending_events=merged,sentence_opportunities=ordinal,checks=checks,triggers=triggers,
            generator_prompt_tokens=len(prompt),reader_prompt_tokens=sum(q['input_tokens'] for q in checks),
            reader_cached_tokens=sum(q['cached_tokens'] for q in checks),
            hook_host_seconds=self.hook_host_seconds-hook_start,trigger_host_seconds=trigger_seconds,
            detector_host_seconds=detector_seconds,abort_seconds=abort_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
        self.registry.clear();self.hidden.clear();return result

    def close(self):
        self.runner.model.compute_logits=self.original
        self.engine.engine_core.shutdown()
        if self.torch.distributed.is_initialized():self.torch.distributed.destroy_process_group()


def main(gpu):
    dest=OUT/f'gpu{gpu}';dest.mkdir(exist_ok=True)
    lock=(dest/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (dest/'completion.json').exists()
    p=read(OUT/'protocol.json')
    bindings=dict(p['bindings'])
    if (OUT/'execution-amendments.json').exists():
        amendment=read(OUT/'execution-amendments.json')
        assert amendment['original_protocol_sha256']==sha(OUT/'protocol.json')
        bindings.update(amendment['active_bindings'])
    for f,h in bindings.items():assert sha(f)==h,f
    guard(gpu)
    signal.signal(signal.SIGALRM,lambda a,b:(_ for _ in ()).throw(TimeoutError('Online worker wall limit')))
    signal.alarm(14400)
    write(dest/'status.json',dict(UTC=now(),state='loading',pid=os.getpid(),GPU_UUID=GPUS[gpu]))
    runner=Engine(p)
    try:
        inputs={r['sample_id']:r for r in rows(STREAM/'inputs-no-gold.jsonl')}
        metadata={r['sample_id']:r for r in rows(STREAM/'samples.jsonl')}
        def sample(sid):return dict(**metadata[sid],prompt_ids=inputs[sid]['prompt_ids'])
        engineering=[]
        for sid in p['engineering_ids']:
            guard(gpu);s=sample(sid)
            write(dest/'status.json',dict(UTC=now(),state='engineering',phase='baseline',pid=os.getpid(),sample_id=sid))
            base=runner.pipeline(s,'terminal',cap=128)
            write(dest/'status.json',dict(UTC=now(),state='engineering',phase='concurrent_read',pid=os.getpid(),sample_id=sid))
            active=runner.pipeline(s,'probe',cap=128,probe=True)
            write(dest/'engineering-latest.json',dict(sample_id=sid,baseline=base,concurrent=active))
            first=active['checks'][0]
            assert first['prefix_tokens']==64 and not first['terminal']
            assert runner.llm.reset_prefix_cache()
            cold=runner.query(runner.read_ids(s['prompt_ids'],active['generated_ids'][:64]),s['candidate_count'])
            delta=max(abs(a-b) for a,b in zip(first['probabilities'],cold['probabilities']))
            row=dict(sample_id=sid,concurrent_prefix_probability_max_difference=delta,
                read_answer_equal=first['probabilities'].index(max(first['probabilities']))==cold['probabilities'].index(max(cold['probabilities'])),
                generation_identical=base['generated_ids']==active['generated_ids'],
                active_request_cache_tokens=first['cached_tokens'],cold_cache_tokens=cold['cached_tokens'],baseline=base,concurrent=active)
            engineering.append(row);write(dest/'engineering.json',engineering)
            assert cold['cached_tokens']==0 and delta<=.001 and row['read_answer_equal'],'Concurrent/cold reader validation failed'
            assert row['generation_identical'],'Concurrent numerical generation drift: analyze before timing expansion'
        write(dest/'settings.json',dict(UTC=now(),startup_seconds=runner.startup_seconds,torch=runner.torch.__version__,vllm=__import__('vllm').__version__,GPU_UUID=GPUS[gpu]))
        selected=p['online_ids'][gpu::2];total=len(selected)*len(POLICIES)*p['repeats'];completed=0
        (dest/'rows').mkdir(exist_ok=True)
        for rep in range(p['repeats']):
            for j,sid in enumerate(selected):
                policies=list(POLICIES);random.Random(20261009+rep*1000+j).shuffle(policies)
                for policy in policies:
                    guard(gpu);case=f'r{rep}-{j:03}-{policy}';path=dest/'rows'/(case+'.json')
                    if path.exists():completed+=1;continue
                    write(dest/'status.json',dict(UTC=now(),state='running',pid=os.getpid(),completed=completed,total=total,sample_id=sid,policy=policy,repeat=rep))
                    row=runner.pipeline(sample(sid),policy)
                    write(path,dict(**row,UTC=now(),case_id=case,repeat=rep,GPU_UUID=GPUS[gpu],protocol_sha256=sha(OUT/'protocol.json')))
                    completed+=1
        done=dict(UTC=now(),state='completed',cases=completed,questions=len(selected),policies=len(POLICIES),repeats=p['repeats'],gold_access=False)
        write(dest/'completion.json',done);write(dest/'status.json',done)
    finally:runner.close()


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--gpu',type=int,choices=[0,1],required=True);args=a.parse_args()
    try:main(args.gpu)
    except BaseException:
        write(OUT/f'gpu{args.gpu}'/'status.json',dict(UTC=now(),state='failed',pid=os.getpid(),error=traceback.format_exc()))
        raise
