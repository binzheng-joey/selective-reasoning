"""Isolated review supplements. Frozen parents are read, never modified.

GPU collection reads no labels. Development-only selection precedes subsequent
comparison; the observed768 extension is explicitly not fresh confirmation.
"""
import argparse
import copy
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

PROJECT = Path('/project')
SOURCES = PROJECT / 'output/kev-adaptive-reasoning/decision-improvements/source'
sys.path.insert(0, str(SOURCES))
import arr_formal_run as formal
from arr_variable_choice import Engine
from arr_deer_counting_repair import validate_deer

rt = formal.rt
OUT = Path(__file__).resolve().parents[1]
SOURCE = Path(__file__).resolve()


def configs(family):
    parent = formal.verify()
    ref = copy.deepcopy(next(q for q in parent['policies'] if q['id'] == 'ours-entropy70-1024'))
    if family == 'controls':
        return [dict(ref, id='factorial-' + g + '-' + s, gate=g,
                     schedule=s) for g in ('benefit_raw', 'margin')
                for s in ('learned_tick4', 'every8')]
    if family == 'development':
        result = []
        for threshold in (.70, .85, .95):
            result.append(dict(ref, id=f'dev-learned-q{int(threshold*100)}', stop_threshold=threshold))
            for gap in (4, 8, 16):
                result.append(dict(ref, id=f'dev-margin-gap{gap}-q{int(threshold*100)}',
                                   gate='margin', schedule=f'every{gap}', stop_threshold=threshold))
        deer = copy.deepcopy(next(q for q in parent['policies'] if q['kind'] == 'deer' and q['cap'] == 1024))
        for threshold in (.80, .85, .90, .95):
            result.append(dict(deer, id=f'dev-deer-q{int(threshold*100)}', trial_threshold=threshold))
        result.append(copy.deepcopy(next(q for q in parent['policies'] if q['id'] == 'forced-terminal-prefill-1024')))
        return result
    if family == 'calibrated':
        selection = rt.read(OUT / 'development/selection.json')
        return selection['evaluation_policies']
    if family == 'serial':
        return [dict(ref, id='execution-concurrent', execution='concurrent'),
                dict(ref, id='execution-serial', execution='serial_abort_resume')]
    raise ValueError(family)


def validate_head(row, sample, policy):
    # Inherited checker only restricts scheduling names, so every16 uses a local
    # policy validation extension and the unchanged execution control flow.
    formal.row_checks(row, sample, policy)
    assert row['physical_output_tokens'] == len(row['generated_ids']) + row['full_reads']
    assert row['tokens_after_stop_decision'] == 0
    assert all(0 <= c['cached_tokens'] <= c['input_tokens'] for c in row['checks'])


def prepare(family):
    parent, projection = formal.verify_inputs()
    folder = OUT / family
    path = folder / 'protocol.json'
    if path.exists():
        p = rt.read(path)
        for name, digest in p['bindings'].items():
            assert rt.sha(name) == digest, ('Changed supplement binding', name)
        return p
    policies = configs(family)
    source_inputs = formal.budget.OUT / 'inputs-no-gold.jsonl' if family == 'development' else formal.OUT / 'inputs-no-gold.jsonl'
    samples = formal.rows(source_inputs)
    if family == 'serial':
        samples = [s for s in samples if s['sample_id'] in set(parent['repeat_ids'])]
    if family == 'development':
        assert len(samples) == 48
        assert not ({s['sample_id'] for s in samples} & {s['sample_id'] for s in formal.rows(formal.OUT / 'inputs-no-gold.jsonl')})
    folder.mkdir(parents=True, exist_ok=True)
    ids = {s['sample_id'] for s in samples}
    repeats = set(parent['repeat_ids']) & ids
    jobs = []
    for repeat in ((0,) if family == 'development' else (0, 1, 2)):
        for i, sample in enumerate(samples):
            if repeat and sample['sample_id'] not in repeats:
                continue
            order = copy.deepcopy(policies)
            random.Random(formal.rank(f'review-supplement/{family}/{repeat}/{sample["sample_id"]}')).shuffle(order)
            jobs += [dict(case_id=f'r{repeat}-{i:04}-{q["id"]}', sample_id=sample['sample_id'], policy=q['id'], repeat=repeat) for q in order]
    rt.save(folder / 'jobs.json', jobs)
    deps = [SOURCE, source_inputs, formal.OUT / 'protocol.json', formal.budget.PILOT / 'gates.json',
            formal.budget.TASKLESS, folder / 'jobs.json']
    deps += [SOURCES / n for n in ('arr_integrated_online.py', 'arr_prior_baselines.py', 'arr_control.py',
                'arr_variable_choice.py', 'arr_repaired_final.py', 'arr_native_final.py', 'sentence_online.py', 'arr_deer_counting_repair.py')]
    if family == 'calibrated':
        deps.append(OUT / 'development/selection.json')
    if family == 'serial':
        deps.append(OUT / 'source/serial_adapter.py')
    p = dict(UTC=rt.now(), family=family, policies=policies, questions=len(samples), total=len(jobs),
        input_file=str(source_inputs), sample_ids=[s['sample_id'] for s in samples], repeat_ids=sorted(repeats),
        engine=parent['engine'], head=parent['head'], parent_protocol_sha256=rt.sha(formal.OUT / 'protocol.json'),
        scope='Development only' if family == 'development' else 'Observed768 review extension; not independent new confirmation',
        selection='Only development48 labels may select settings; no evaluation label access in workers',
        fairness='All compared methods per question on one physical GPU, deterministic interleaving; no cross-card latency splicing',
        cost='Synchronized online policy time, all outputs and repeated inputs. Original cross-request cache disabled.',
        serial_boundary='Serial implementation pauses by aborting and restoring exact prefix IDs; counts resumed prefill. Not a KV-preserving pause ablation.' if family == 'serial' else None,
        validation_extension='every16 uses the identical everyN scheduler and a process-local policy validator, never edits frozen source',
        bindings={str(f): rt.sha(f) for f in deps})
    rt.save(path, p)
    rt.save(folder / 'status.json', dict(UTC=rt.now(), state='prepared', completed=0, total=len(jobs)))
    return p


def install_gap16():
    import arr_integrated_online as original
    original_validate = original.validate_policy
    def validate(policy):
        if policy['schedule'] == 'every16':
            original_validate(dict(policy, schedule='every8'))
        else:
            original_validate(policy)
    original.validate_policy = validate


def worker(family, gpu):
    p = prepare(family)
    formal.verify(model=True)
    folder = OUT / family
    assert os.environ['CUDA_VISIBLE_DEVICES'] == str(gpu)
    with (rt.QUEUE / f'gpu{gpu}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('Stopped')))
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('24h worker limit')))
        signal.alarm(86400)
        checksum = rt.sha(folder / 'protocol.json')
        jobs = rt.read(folder / 'jobs.json')
        specs = {q['id']: q for q in p['policies']}
        samples = {s['sample_id']: s for s in formal.rows(p['input_file'])}
        done = set()
        runner = None
        def status(state, **extra):
            value = dict(UTC=rt.now(), state=state, pid=os.getpid(), gpu=gpu, GPU_UUID=rt.GPUS[gpu],
                completed=len(done), total=len(jobs), gold_access=False, **extra)
            rt.save(folder / 'status.json', value)
            return value
        def guard():
            if rt.stop_requested() or (OUT / 'STOP').exists():
                raise InterruptedError('Stop requested')
            other = [pid for pid in rt.inventory()[gpu]['pids'] if pid != os.getpid()]
            if other:
                raise formal.Contention(f'GPU{gpu} foreign PIDs {other}')
        try:
            for job in jobs:
                path = folder / 'rows' / (job['case_id'] + '.json')
                if path.exists():
                    row = rt.read(path)
                    assert row['protocol_sha256'] == checksum and row['GPU_UUID'] == rt.GPUS[gpu]
                    done.add(job['case_id'])
            guard()
            status('loading')
            install_gap16()
            if family == 'serial':
                from serial_adapter import Engine as Runner
            else:
                Runner = Engine
            runner = Runner(p)
            runner.tr = rt.read(formal.budget.TASKLESS)
            runner.trigger_weight = runner.torch.tensor(runner.tr['hidden_weight'], dtype=runner.torch.float32, device='cuda')
            gates = rt.read(formal.budget.PILOT / 'gates.json')
            def run(s, q):
                row = runner.integrated(s, q, gates) if q['kind'] == 'head' else runner.prior(s, q, gates)
                if q['kind'] == 'head':
                    validate_head(row, s, q)
                elif q['kind'] == 'deer':
                    validate_deer(row, s, q, runner.tok)
                else:
                    formal.row_checks(row, s, q)
                return row
            # Warm only previously observed development questions, never score.
            warm = formal.rows(formal.budget.OUT / 'inputs-no-gold.jsonl')[0]
            status('engineering')
            if family == 'controls':
                warm_policies = [dict(p['policies'][-1], mode='always', id='warmup')]
            elif family == 'development':
                warm_policies = [dict(next(q for q in p['policies'] if q.get('schedule') == 'every16'), mode='always', id='warm-gap16'), next(q for q in p['policies'] if q['kind'] == 'deer')]
            else:
                warm_policies = p['policies'][:2]
            for q in warm_policies:
                guard()
                row = run(warm, q)
                guard()
                rt.save(folder / 'engineering' / (q['id'] + '.json'), dict(scope='Warmup excluded from results', row=row))
            rt.save(folder / 'environment.json', dict(UTC=rt.now(), python=sys.version, torch=runner.torch.__version__,
                cuda=runner.torch.version.cuda, gpu=gpu, GPU_UUID=rt.GPUS[gpu], protocol_sha256=checksum))
            for job in jobs:
                if job['case_id'] in done:
                    continue
                s, q = samples[job['sample_id']], specs[job['policy']]
                guard()
                status('running', case_id=job['case_id'], dataset=s['dataset'], policy=q['id'])
                row = run(s, q)
                record = dict(**row, UTC=rt.now(), case_id=job['case_id'], repeat=job['repeat'], GPU_UUID=rt.GPUS[gpu],
                    protocol_sha256=checksum)
                try:
                    guard()
                except BaseException:
                    rt.save(folder / 'excluded' / (job['case_id'] + '.json'), record)
                    raise
                rt.save(folder / 'rows' / (job['case_id'] + '.json'), record)
                done.add(job['case_id'])
            result = status('completed_unscored')
            rt.save(folder / 'completion.json', result)
        except BaseException:
            status('interrupted_or_failed', error=traceback.format_exc())
            raise
        finally:
            if runner:
                runner.close()


def launch(family, gpu, wait=False):
    p = prepare(family)
    folder = OUT / family
    with (folder / 'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (folder / 'launcher.json').exists(), 'Existing worker; inspect before resume'
        deadline = time.monotonic() + 24*3600
        while True:
            if (OUT / 'STOP').exists() or rt.stop_requested():
                raise InterruptedError('Stop requested')
            card = rt.inventory()[gpu]
            if not card['pids'] and card['memory_mib'] < 1024 and card['utilization'] <= 5:
                time.sleep(2)
                again = rt.inventory()[gpu]
                if not again['pids'] and again['memory_mib'] < 1024 and again['utilization'] <= 5:
                    break
            if not wait:
                raise RuntimeError(f'GPU{gpu} busy: {card}')
            rt.save(folder / 'status.json', dict(UTC=rt.now(), state='waiting_for_idle_GPU', gpu=gpu,
                completed=0, total=p['total'], card=card))
            if time.monotonic() >= deadline:
                raise TimeoutError('Idle GPU wait expired after24h')
            time.sleep(30)
        env = rt.env_for(gpu)
        cache = folder / f'runtime-cache-gpu{gpu}'
        env.update(VLLM_CACHE_ROOT=str(cache / 'vllm'), TORCHINDUCTOR_CACHE_DIR=str(cache / 'inductor'), TRITON_CACHE_DIR=str(cache / 'triton'))
        command = [str(rt.PYTHON), '-B', str(SOURCE), '--worker', '--family', family, '--gpu', str(gpu)]
        with (folder / 'worker.log').open('a') as log:
            child = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(UTC=rt.now(), pid=child.pid, gpu=gpu, command=command, total=p['total'])
        rt.save(folder / 'launcher.json', value)
        print(json.dumps(value), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ('prepare', 'launch', 'worker'):
        action.add_argument('--' + name, action='store_true')
    parser.add_argument('--family', required=True, choices=('controls', 'development', 'calibrated', 'serial'))
    parser.add_argument('--gpu', type=int, choices=(0, 1, 2))
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.family), ensure_ascii=False))
    elif args.worker:
        worker(args.family, args.gpu)
    else:
        launch(args.family, args.gpu, args.wait)
