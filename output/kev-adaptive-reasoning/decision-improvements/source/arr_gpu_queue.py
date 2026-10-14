"""Bounded, exclusive A100 recovery queue. No VRAM reservation or foreign signals.

Reuse frozen runtime/online implementations; retain interrupted raw measurements.
Only completed, uncontended phases are published for the original analyzer.
Later scientific stages remain dependent on actual runtime and method selection.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[2]
ORIGINAL = ROOT / 'sentence-runtime'
OUT = ORIGINAL / 'recovery'
QUEUE = ROOT / 'arr-sprint'
PYTHON = ROOT.parent / 'vllm-qwen35-a100-isolated-20261002-v1/.venv/bin/python'
GPUS = {0: 'device-C',
        1: 'device-B',
        2: 'device-A'}
SHARDS = (0, 1)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def verify():
    p = read(QUEUE / 'queue-protocol.json')
    for name, digest in p['bindings'].items():
        assert sha(name) == digest, ('Changed frozen input', name)
    original = read(ORIGINAL / 'protocol.json')
    for name, digest in original['bindings'].items():
        assert sha(name) == digest, ('Changed original input', name)
    assert sha(OUT / 'protocol.json') == sha(ORIGINAL / 'protocol.json')
    return p, original


def inventory():
    def query(args):
        return subprocess.check_output(['nvidia-smi'] + args,
            text=True, stderr=subprocess.STDOUT, timeout=10)
    cards = {}
    for line in query(['--query-gpu=index,uuid,name,memory.used,utilization.gpu',
                       '--format=csv,noheader,nounits']).splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) != 5:
            raise RuntimeError('Unexpected nvidia-smi inventory')
        index = int(fields[0])
        if index in GPUS:
            assert fields[1] == GPUS[index], ('GPU identity changed', index)
            assert 'A100' in fields[2]
            cards[index] = dict(uuid=fields[1], name=fields[2],
                memory_mib=float(fields[3]), utilization=float(fields[4]), pids=[])
    assert set(cards) == set(GPUS), 'Authorized A100 device is missing'
    for line in query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader']).splitlines():
        parts = [v.strip() for v in line.split(',')]
        if len(parts) == 2 and parts[0] in GPUS.values():
            for index in GPUS:
                if parts[0] == GPUS[index]:
                    cards[index]['pids'].append(int(parts[1]))
    return cards


def is_idle(card, p):
    return (not card['pids'] and card['memory_mib'] <= p['idle_memory_mib']
            and card['utilization'] <= p['idle_utilization_percent'])


def stop_requested():
    return any(p.exists() for p in [QUEUE/'STOP', OUT/'STOP', ORIGINAL/'STOP'])


def env_for(gpu):
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env.pop('PYTHONHOME', None)
    cache = OUT / ('gpu' + str(gpu)) / 'runtime-cache'
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
        PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        VLLM_ENABLE_V1_MULTIPROCESSING='0', OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        VLLM_CACHE_ROOT=str(cache/'vllm'), TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),
        TRITON_CACHE_DIR=str(cache/'triton'))
    return env


def worker(gpu, condition, attempt, device):
    verify()
    import sentence_runtime as base
    attempt = Path(attempt).resolve()
    assert attempt.is_relative_to(OUT / 'attempts')
    base.OUT = attempt
    phase = attempt / ('gpu' + str(gpu)) / condition
    original_save = base.save
    assert device in GPUS
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(device)
    # Logical shards retain sample membership/order; physical assignment is pinned.
    base.online.GPUS[gpu] = GPUS[device]

    def guard(g, allow_self=True):
        assert g == gpu
        assert not stop_requested(), 'Requested queue stop'
        cards = inventory()
        others = [pid for pid in cards[device]['pids'] if not allow_self or pid != os.getpid()]
        assert not others, ('Another GPU owner appeared', device, others)

    def checked_save(path, value):
        # A competing process appearing during a timed request invalidates it.
        # Checks occur outside the measured request interval.
        if Path(path).parent.name == 'rows':
            guard(gpu)
            value.update(logical_shard=gpu, physical_gpu=device)
        original_save(path, value)

    base.guard = guard
    base.save = checked_save
    signal.signal(signal.SIGTERM, lambda a, b: (_ for _ in ()).throw(
        RuntimeError('Worker stopped by its queue controller')))
    try:
        base.worker(gpu, condition)
    except BaseException as exc:
        save(phase/'failure.json', dict(UTC=now(), state='failed',
            error=str(exc), traceback=traceback.format_exc()))
        raise


def prepare():
    QUEUE.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    assert not (QUEUE/'queue-protocol.json').exists(), 'Inspect existing queue; do not duplicate'
    for filename in ['protocol.json', 'analysis-protocol.json']:
        assert not (OUT/filename).exists()
        (OUT/filename).write_bytes((ORIGINAL/filename).read_bytes())
    p = read(ORIGINAL/'protocol.json')
    inputs = [Path(__file__).resolve(), ORIGINAL/'protocol.json', ORIGINAL/'analysis-protocol.json',
        ROOT/'source/sentence_runtime.py', ROOT/'source/sentence_runtime_analyze.py',
        ROOT/'source/sentence_online.py']
    excluded = {str(f): sha(f) for f in ORIGINAL.glob('gpu*/**/rows/*.json')}
    save(OUT/'recovery-amendment.json', dict(UTC=now(), user_authorized=True,
        original_protocol_sha256=sha(ORIGINAL/'protocol.json'),
        preserved_excluded_measurements=excluded,
        reason='Original phases failed on GPU contention before timed-prefix validation; no partial timings reused',
        unchanged=['samples','policies','repeats','cap','condition_orders','engine_settings','reader','trigger'],
        changes=['new output directory','idle queue','post-request occupancy check',
                 'retain failed attempts; retry resource interruptions only'],
        selection='Whole phases accepted only after uncontended completion and cold validation; no accuracy selection',
        downstream='Runtime numerical/cost review required before releasing integrated/confirmation jobs'))
    save(QUEUE/'queue-protocol.json', dict(UTC=now(), GPUs=GPUS,
        poll_seconds=15, required_idle_observations=2, idle_memory_mib=1024,
        idle_utilization_percent=5, maximum_attempts_per_phase=4,
        deadline_UTC='2026-10-12T12:00:00+00:00',
        phase_wall_limit_seconds=p['phase_wall_limit_seconds'],
        bindings={str(f):sha(f) for f in inputs},
        policy='No memory holder; no signals to foreign processes; fixed question/device allocation; per-device exclusive lock',
        failure_policy='Retry resource interruptions; stop and report technical failures; never open independent labels'))
    save(QUEUE/'jobs.json', dict(UTC=now(), jobs=[
        dict(id='runtime_recovery', state='ready', conditions=8, timed_cases=256,
             output=str(OUT), auto_launch=True),
        dict(id='integrated_development', state='waiting_for_runtime_review_and_cpu_selection',
             auto_launch=False, requires=['runtime_recovery','combined_replay','baseline_implementation']),
        dict(id='formal_comparison', state='waiting_for_frozen_method_and_baselines',
             auto_launch=False, requires=['integrated_development']),
        dict(id='independent_confirmation', state='waiting_for_data_audit_and_method_freeze',
             auto_launch=False, requires=['independent_data_audit','integrated_development'])]))
    print(json.dumps(dict(state='prepared', excluded_old_rows=len(excluded), protocol=str(QUEUE/'queue-protocol.json'))))


def terminate_owned(entry, reason):
    proc = entry['proc']
    if proc.poll() is not None:
        return
    save(entry['attempt']/'interrupt.json', dict(UTC=now(), reason=reason, worker_pid=proc.pid))
    proc.terminate()
    try:
        proc.wait(timeout=12)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=12)


def supervise():
    lock = (QUEUE/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    p, protocol = verify()
    deadline = datetime.fromisoformat(p['deadline_UTC'])
    active = {}
    idle_counts = {g:0 for g in GPUS}
    assignment_path = QUEUE/'device-assignment.json'
    assignments = {int(k):v for k,v in read(assignment_path)['shards'].items()} if assignment_path.exists() else {}
    assert set(assignments).issubset(SHARDS)
    assert len(set(assignments.values())) == len(assignments)
    assert all(v in GPUS for v in assignments.values())
    device_locks = {}
    started = now()
    try:
        while True:
            if stop_requested() or datetime.now(timezone.utc) >= deadline:
                for entry in active.values():
                    terminate_owned(entry, 'queue_stop_or_deadline')
                save(QUEUE/'status.json', dict(UTC=now(), state='stopped_or_expired', pid=os.getpid()))
                return
            try:
                cards = inventory()
            except Exception as exc:
                if active:
                    raise RuntimeError('GPU monitoring failed during execution: '+str(exc))
                save(QUEUE/'status.json', dict(UTC=now(), state='waiting_for_gpu_visibility',
                    pid=os.getpid(), error=str(exc), next_poll_seconds=p['poll_seconds']))
                time.sleep(p['poll_seconds'])
                continue
            # Observe every active worker, and terminate only our own child if contested.
            for gpu, entry in list(active.items()):
                proc = entry['proc']
                others = [pid for pid in cards[entry['device']]['pids'] if pid != proc.pid]
                if others and proc.poll() is None:
                    terminate_owned(entry, dict(kind='resource_contention', other_pids=others))
                code = proc.poll()
                if code is None:
                    continue
                entry['log'].close()
                phase = entry['attempt']/('gpu'+str(gpu))/entry['condition']
                interrupted = entry['attempt']/'interrupt.json'
                if code == 0 and (phase/'completion.json').exists() and not interrupted.exists():
                    target = OUT/('gpu'+str(gpu))/entry['condition']
                    assert not target.exists(), 'Never replace accepted raw measurements'
                    phase.rename(target)
                    save(entry['attempt']/'accepted.json', dict(UTC=now(), published=str(target)))
                else:
                    failure = read(phase/'failure.json') if (phase/'failure.json').exists() else {}
                    resource = (interrupted.exists() and 'resource_contention' in str(read(interrupted)))
                    resource = resource or 'Another GPU owner appeared' in failure.get('error','')
                    if not resource:
                        raise RuntimeError('Technical worker failure: '+str(phase)+' '+str(failure.get('error',code)))
                    save(entry['attempt']/'excluded.json', dict(UTC=now(), reason='resource_contention', exit_code=code))
                device_locks[gpu].close()
                del device_locks[gpu]
                del active[gpu]
                idle_counts[entry['device']] = 0
            for device in GPUS:
                idle_counts[device] = idle_counts[device]+1 if is_idle(cards[device],p) else 0
            completed = 0
            for gpu in SHARDS:
                dest = OUT/('gpu'+str(gpu))
                dest.mkdir(exist_ok=True)
                order = protocol['condition_orders'][str(gpu)]
                done = [c for c in order if (dest/c/'completion.json').exists()]
                completed += len(done)
                if len(done) == len(order):
                    if not (dest/'completion.json').exists():
                        save(dest/'completion.json', dict(UTC=now(), state='completed', conditions=len(order), gold_access=False))
                    continue
                if gpu in active:
                    continue
                devices = ([assignments[gpu]] if gpu in assignments else
                    [d for d in GPUS if d not in assignments.values()])
                eligible = [d for d in devices if idle_counts[d] >= p['required_idle_observations']]
                if not eligible:
                    continue
                device = eligible[0]
                condition = next(c for c in order if c not in done)
                attempts = OUT/'attempts'/('gpu'+str(gpu))/condition
                attempts.mkdir(parents=True,exist_ok=True)
                index = len([q for q in attempts.iterdir() if q.is_dir()])+1
                assert index <= p['maximum_attempts_per_phase'], ('Resource retry limit', gpu, condition)
                verify()
                device_lock = (QUEUE/('gpu'+str(device)+'.lock')).open('a')
                fcntl.flock(device_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                # Recheck immediately before mutation/launch; this is not a reservation.
                if not is_idle(inventory()[device],p):
                    device_lock.close(); idle_counts[device]=0; continue
                if gpu not in assignments:
                    assignments[gpu] = device
                    save(assignment_path,dict(UTC=now(),shards=assignments,
                        policy='Each logical question shard is pinned to its first available physical A100 for every condition and repeat'))
                attempt = attempts/('{:02d}'.format(index))
                attempt.mkdir()
                (attempt/'protocol.json').write_bytes((OUT/'protocol.json').read_bytes())
                command = [str(PYTHON),'-B',str(Path(__file__).resolve()),'--worker',
                    '--gpu',str(gpu),'--physical-gpu',str(device),'--condition',condition,'--attempt',str(attempt)]
                log = (attempt/'worker.log').open('ab')
                proc = subprocess.Popen(command, cwd=PROJECT, env=env_for(device),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                save(attempt/'launcher.json', dict(UTC=now(), pid=proc.pid, GPU_UUID=GPUS[device],
                    logical_shard=gpu,physical_gpu=device,command=command))
                active[gpu] = dict(proc=proc, log=log, attempt=attempt, condition=condition, device=device)
                device_locks[gpu] = device_lock
                idle_counts[device] = 0
            if completed == 8 and not active:
                import sentence_runtime_analyze as analysis
                analysis.OUT = OUT
                analysis.main(False)
                assert (OUT/'completion.json').exists()
                # The inherited analyzer calls the logical shard "GPU"; record
                # physical indices explicitly in its numerical diagnostics.
                numerical=read(OUT/'numerical-checks.json')
                for row in numerical:
                    shard=row['GPU'];row.update(logical_shard=shard,GPU=assignments[shard])
                save(OUT/'numerical-checks.json',numerical)
                save(QUEUE/'status.json', dict(UTC=now(), state='runtime_complete_review_required',
                    pid=os.getpid(), output=str(OUT/'report.md'), independent_labels_accessed=False,
                    next='Review execution costs and numerical differences before releasing downstream protocol'))
                jobs = read(QUEUE/'jobs.json')
                jobs['jobs'][0]['state'] = 'completed'
                jobs['UTC'] = now()
                save(QUEUE/'jobs.json',jobs)
                return
            save(QUEUE/'status.json', dict(UTC=now(), state='running' if active else 'waiting_for_idle_gpu',
                pid=os.getpid(), started_UTC=started, cards=cards, completed_phases=completed, total_phases=8,
                device_assignments=assignments,
                workers={str(g):dict(pid=e['proc'].pid,physical_gpu=e['device'],condition=e['condition'],attempt=str(e['attempt'])) for g,e in active.items()},
                poll_seconds=p['poll_seconds'], deadline_UTC=p['deadline_UTC'], memory_reservation=False))
            time.sleep(p['poll_seconds'])
    except BaseException:
        for entry in active.values():
            terminate_owned(entry, 'controller_failure')
        save(QUEUE/'status.json',dict(UTC=now(),state='failed',pid=os.getpid(),error=traceback.format_exc()))
        raise
    finally:
        for entry in active.values():
            entry['log'].close()
        for device_lock in device_locks.values():
            device_lock.close()


def launch():
    verify()
    if (QUEUE/'launcher.json').exists():
        old = read(QUEUE/'launcher.json')
        try:
            os.kill(old['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError('Existing launcher PID is alive; inspect rather than duplicate')
    inventory()  # Establish real host visibility before claiming an active monitor.
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--supervise']
    with (QUEUE/'queue.log').open('ab') as log:
        proc=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(UTC=now(),pid=proc.pid,command=command)
    save(QUEUE/'launcher.json',record)
    print(json.dumps(record))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    for option in ['prepare','launch','supervise','worker','check']:
        modes.add_argument('--'+option,action='store_true')
    parser.add_argument('--gpu',type=int,choices=[0,1])
    parser.add_argument('--physical-gpu',type=int,choices=[0,1,2])
    parser.add_argument('--condition',choices=['compiled-on','compiled-off','eager-on','eager-off'])
    parser.add_argument('--attempt')
    args=parser.parse_args()
    if args.prepare: prepare()
    elif args.launch: launch()
    elif args.supervise: supervise()
    elif args.worker: worker(args.gpu,args.condition,args.attempt,args.physical_gpu)
    else:
        verify()
        print(json.dumps(dict(state='bindings_valid',GPUs=GPUS)))
