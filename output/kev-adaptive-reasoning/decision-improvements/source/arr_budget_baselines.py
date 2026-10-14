"""Paired, two-A100 development matrix for frozen-budget and baseline selection.

No independent confirmation material or labels are read. Existing bound pilots
remain unchanged. Each question and every method on it stay on one physical GPU.
"""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import sys
import time
import traceback

import arr_gpu_queue as rt

ROOT = rt.ROOT
MAIN = ROOT / 'arr-sprint/main-table'
OUT = MAIN / 'budget-baselines'
ARC = MAIN / 'arc'
PILOT = ROOT / 'arr-sprint/integrated'
SOURCE = Path(__file__).resolve()
TASKLESS = MAIN / 'task-agnostic-trigger/trigger.json'


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def parse_prior_answer(text, count):
    # Preserve exactly the established <=4 interface, extend only the new path.
    from arr_control import parse_answer
    from arr_native_final import parse_candidate_index
    return parse_answer(text, count) if count <= 4 else parse_candidate_index(text, count)


def policies(old):
    by = {p['id']: p for p in old['core_policies']}
    result = [copy.deepcopy(by['direct'])]
    for cap in (512, 1024, 2048):
        p = copy.deepcopy(by['full'])
        p.update(id=f'full-{cap}', cap=cap)
        result.append(p)
    p = copy.deepcopy(by['simple-margin-every8'])
    p['id'] = 'simple-margin-every8-1024'
    result.append(p)
    # The learned trigger remains inside its calibrated 1024-token domain.
    for threshold in (.70, .85, .95):
        p = copy.deepcopy(by['benefit-tick4'])
        p.update(id=f'ours-entropy{int(threshold*100):02d}-1024',
                 stop_threshold=threshold, trigger_variant='taskless')
        result.append(p)
    for cap in (1024, 2048):
        for source in old['prior_policies']:
            p = copy.deepcopy(source)
            p.update(id=f'{source["id"]}-prefill-{cap}', cap=cap,
                     answer_interface='assistant_FINAL_prefix_completion')
            result.append(p)
    result.append(dict(id='native-8192', kind='native_final', mode='always', cap=8192))
    assert len(result) == 15 and len({p['id'] for p in result}) == 15
    return result


def verify():
    p = rt.read(OUT / 'protocol.json')
    for name, digest in p['bindings'].items():
        assert rt.sha(name) == digest, ('Changed budget input', name)
    for name, digest in rt.read(PILOT / 'protocol.json')['bindings'].items():
        assert rt.sha(name) == digest, ('Changed pilot input', name)
    return p


def prepare():
    if (OUT / 'protocol.json').exists():
        return verify()
    import sentence_online as base
    sys.path.insert(0, str(ARC / 'deps'))
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    old = rt.read(PILOT / 'protocol.json')
    ins = {q['sample_id']: q for q in rows(base.STREAM / 'inputs-no-gold.jsonl')}
    meta = {q['sample_id']: q for q in rows(base.STREAM / 'samples.jsonl')}
    samples = [dict(sample_id=sid, dataset=meta[sid]['dataset'],
                    component_id=meta[sid]['component_id'],
                    candidate_count=meta[sid]['candidate_count'], prompt_ids=ins[sid]['prompt_ids'])
               for sid in old['sample_ids']]
    metadata = rt.read(ARC / 'validation-metadata.json')
    excluded = set(rt.read(ARC / 'development-exclusions.json')['splits']['validation']['excluded_ids'])
    pool = [x for x in metadata['records'] if x['sample_id'] not in excluded]
    selected = []
    for count, quota in [(3, 1), (4, 14), (5, 1)]:
        selected.extend([x for x in pool if x['candidate_count'] == count][:quota])
    assert len(selected) == 16
    raw = ARC / 'raw/validation-00000-of-00001.parquet'
    assert rt.sha(raw) == metadata['source_sha256']
    source = {str(x['id']): x for x in pq.read_table(raw, columns=['id', 'question', 'choices']).to_pylist()}
    tok = AutoTokenizer.from_pretrained(old['engine']['model'], local_files_only=True)
    system = 'Use the supplied question and relevant scientific knowledge. Keep the original candidate order. Follow the requested output format.'
    for s in selected:
        x = source[s['sample_id'].removeprefix('arc:')]
        count = s['candidate_count']
        user = 'Material:\n' + x['question'] + '\n\nTask:\nSelect the candidate that best answers this science question.\n\nCandidates (in the given order):\n'
        user += '\n'.join(f'{i+1}. {v}' for i, v in enumerate(x['choices']['text']))
        user += '\n\nReason through the question using the supplied information and relevant scientific knowledge. Then give exactly one final candidate index from 1 to ' + str(count) + ' in the exact format <FINAL>index</FINAL>. Place this final tag after your reasoning, outside any thinking block. Do not append text after the tag.'
        ids = tok.apply_chat_template([dict(role='system', content=system), dict(role='user', content=user)],
                                     tokenize=True, add_generation_prompt=True, enable_thinking=True)
        samples.append(dict(**s, component_id=s['question_sha256'], prompt_ids=ids))
    assert len(samples) == 48 and len({s['sample_id'] for s in samples}) == 48
    # Interleave tasks, then assign a deterministic within-task balanced shard.
    grouped = {task: [s for s in samples if s['dataset'] == task]
               for task in ('proofwriter', 'logiqa2', 'arc_challenge')}
    assert all(len(v) == 16 for v in grouped.values())
    ordered = []
    for j in range(16):
        for task in grouped:
            s = grouped[task][j]
            s['gpu'] = 1 + ((j // 4) % 2 if task == 'proofwriter' else j % 2)
            ordered.append(s)
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = OUT / 'inputs-no-gold.jsonl'
    inputs.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in ordered))
    files = [SOURCE, inputs, PILOT/'protocol.json', PILOT/'gates.json', TASKLESS,
             ARC/'validation-metadata.json', ARC/'development-exclusions.json', raw,
             ROOT/'source/arr_variable_choice.py', ROOT/'source/arr_repaired_final.py',
             ROOT/'source/arr_native_final.py', ROOT/'source/arr_prior_baselines.py',
             ROOT/'source/arr_integrated_online.py', ROOT/'source/arr_control.py',
             ROOT/'source/sentence_online.py', ROOT/'source/arr_gpu_queue.py']
    p = dict(UTC=rt.now(), scope='Observed development only; no confirmation inputs or labels.',
             engine=old['engine'], engine_choice=old['engine_choice'], head=old['head'],
             policies=policies(old), sample_ids=[s['sample_id'] for s in ordered],
             assignments={str(g): [s['sample_id'] for s in ordered if s['gpu'] == g] for g in (1, 2)},
             repeats=1, cases=720, cases_per_gpu=360, wall_limit_seconds=21600,
             selection='Same 32 observed PW/LQ items plus 16 validation ARC IDs selected by metadata rank and option count, never accuracy. Exclude ARC/test overlap.',
             baseline_parser='For <=4 identical established parser; for >4 strict candidate-index parser. All invalid answers count as wrong.',
             learned_budget='Learned scheduler stays at cap1024; no out-of-domain positional extrapolation. Uniform budget controls at512/1024/2048.',
             native_reference='Natural generation cap8192; record truncation, never assume all answers completed.',
             timing='Same question/all methods on same physical GPU, deterministic shuffled order; GPU-synchronized pipeline excludes tokenize/load/cache reset; counts auxiliary generations.',
             analysis='After full completion only; task-wise and equally weighted aggregates, no automatic method selection or confirmation launch.',
             bindings={str(f): rt.sha(f) for f in files})
    rt.save(OUT / 'protocol.json', p)
    print(json.dumps(dict(prepared=True, questions=48, methods=15, conditions=720, per_gpu=360)))
    return p


def guard(gpu, dest):
    assert not rt.stop_requested() and not (OUT/'STOP').exists() and not (dest/'STOP').exists(), 'Stop requested'
    foreign = [pid for pid in rt.inventory()[gpu]['pids'] if pid != os.getpid()]
    if foreign:
        rt.save(dest/'excluded.json', dict(UTC=rt.now(), reason='resource_contention', pids=foreign))
        raise RuntimeError(f'GPU{gpu} contention: {foreign}')


def launch(gpu):
    p = verify()
    dest = OUT / f'gpu{gpu}'
    dest.mkdir(exist_ok=True)
    with (dest/'launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (dest/'launcher.json').exists(), 'Existing attempt; inspect before relaunch'
        for _ in range(2):
            assert rt.is_idle(rt.inventory()[gpu], dict(idle_memory_mib=1024, idle_utilization_percent=5)), f'GPU{gpu} occupied'
            time.sleep(2)
        cmd = [str(rt.PYTHON), '-B', str(SOURCE), '--worker', '--gpu', str(gpu)]
        with (dest/'worker.log').open('a') as log:
            child = subprocess.Popen(cmd, cwd=rt.PROJECT, env=rt.env_for(gpu), stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        rt.save(dest/'launcher.json', dict(UTC=rt.now(), pid=child.pid, gpu=gpu, command=cmd,
                                         conditions=p['cases_per_gpu']))
        print(json.dumps(dict(gpu=gpu, pid=child.pid, conditions=p['cases_per_gpu'])))


def worker(gpu):
    p = verify()
    dest = OUT / f'gpu{gpu}'
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(gpu)
    lock = (rt.QUEUE/f'gpu{gpu}.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(RuntimeError('Interrupted')))
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Development wall limit')))
    signal.alarm(p['wall_limit_seconds'])
    runner = None
    try:
        guard(gpu, dest)
        rt.save(dest/'status.json', dict(UTC=rt.now(), state='loading', gpu=gpu, pid=os.getpid()))
        import arr_prior_baselines as priors
        from arr_variable_choice import Engine
        # Local worker process only; no frozen source file is modified.
        priors.parse_answer = parse_prior_answer
        runner = Engine(p)
        runner.tr = rt.read(TASKLESS)
        runner.trigger_weight = runner.torch.tensor(runner.tr['hidden_weight'], dtype=runner.torch.float32, device='cuda')
        gates = rt.read(PILOT/'gates.json')
        samples = [s for s in rows(OUT/'inputs-no-gold.jsonl') if s['gpu'] == gpu]
        assert [s['sample_id'] for s in samples] == p['assignments'][str(gpu)]
        # New path only: actual five-option final generation. Previous head and
        # scheduler cold-equivalence checks are already complete and not repeated.
        wide = next(s for s in rows(OUT/'inputs-no-gold.jsonl') if s['candidate_count'] == 5)
        runner.reset_trial(); runner.prior_requests = []
        answer = runner.final_answer(wide['prompt_ids'], [], dict(answer_cap=64))
        pred = parse_prior_answer(answer['text'], 5)
        assert pred is not None and 0 <= pred < 5, 'Five-option generated-final interface failed'
        rt.save(dest/'engineering.json', dict(UTC=rt.now(), sample_id=wide['sample_id'],
                prediction=pred, answer=answer, gold_access=False))
        done = 0
        for j, sample in enumerate(samples):
            order = list(p['policies'])
            random.Random(20261010 + 10000*gpu + j).shuffle(order)
            for policy in order:
                guard(gpu, dest)
                rt.save(dest/'status.json', dict(UTC=rt.now(), state='running', gpu=gpu, pid=os.getpid(),
                        completed=done, total=p['cases_per_gpu'], sample_id=sample['sample_id'], policy=policy['id']))
                if policy['kind'] == 'head':
                    row = runner.integrated(sample, policy, gates)
                elif policy['kind'] == 'native_final':
                    row = runner.native_final(sample, policy)
                else:
                    row = runner.prior(sample, policy, gates)
                guard(gpu, dest)
                rt.save(dest/'rows'/f'{j:03}-{policy["id"]}.json', dict(**row, repeat=0, UTC=rt.now(),
                        GPU_UUID=rt.GPUS[gpu], protocol_sha256=rt.sha(OUT/'protocol.json')))
                done += 1
        result = dict(UTC=rt.now(), state='completed', gpu=gpu, cases=done, gold_access=False)
        rt.save(dest/'completion.json', result)
        rt.save(dest/'status.json', result)
    except BaseException:
        result = dict(UTC=rt.now(), state='failed', gpu=gpu, error=traceback.format_exc())
        rt.save(dest/'failure.json', result); rt.save(dest/'status.json', result)
        raise
    finally:
        if runner:
            runner.close()


def arc_labels(ids):
    # Validation only; return only preselected development labels.
    sys.path.insert(0, str(ARC/'deps'))
    import pyarrow.parquet as pq
    raw = ARC/'raw/validation-00000-of-00001.parquet'
    assert rt.sha(raw) == rt.read(ARC/'validation-metadata.json')['source_sha256']
    selected = {sid.removeprefix('arc:') for sid in ids if sid.startswith('arc:')}
    labels = {}
    # Predicate restriction keeps new/unused validation labels out of Python rows.
    data = pq.read_table(raw, columns=['id', 'choices', 'answerKey'], filters=[('id', 'in', sorted(selected))])
    for x in data.to_pylist():
        assert x['id'] in selected
        labels['arc:'+x['id']] = x['choices']['label'].index(x['answerKey'])
    assert len(labels) == len(selected)
    return labels


def summary(records, samples, labels, methods):
    out = []
    ids = [s['sample_id'] for s in samples]
    repeats = {r['repeat'] for r in records}
    keyed = {(r['sample_id'], r['policy'], r['repeat']): r for r in records}
    assert len(keyed) == len(records)
    assert set(keyed) == {(sid, m, rep) for sid in ids for m in methods for rep in repeats}
    for sid in ids:
        assert len({keyed[sid,m,rep]['GPU_UUID'] for m in methods for rep in repeats}) == 1
    for task in ['all', 'proofwriter', 'logiqa2', 'arc_challenge']:
        selected = [s['sample_id'] for s in samples if task == 'all' or s['dataset'] == task]
        if not selected:
            continue
        for m in methods:
            runs = [keyed[sid,m,rep] for sid in selected for rep in repeats]
            first = [keyed[sid,m,0] for sid in selected]
            value = dict(dataset=task, method=m, n=len(selected),
                         correct=sum(r['prediction'] == labels[r['sample_id']] for r in first),
                         invalid=sum(r['prediction'] is None for r in first),
                         reasoning_rate=statistics.mean(float(r['reasoning_invoked']) for r in first),
                         early_stop_rate=statistics.mean(float(r['early_stop']) for r in first))
            for key in ['pipeline_seconds', 'physical_output_tokens', 'full_reads', 'trigger_calls']:
                value[key] = statistics.mean(r[key] for r in runs)
            value['truncated'] = sum(r.get('budget_truncated', False) for r in first)
            out.append(value)
    return out


def write_report(dest, data, note):
    rt.save(dest/'analysis.json', dict(UTC=rt.now(), summaries=data, scope=note,
                                     independent_test=False))
    lines = ['# 开发比较', '', note, '', '|任务|方法|正确/题数|秒/题|输出token/题|读取/题|无效终答|',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in data:
        lines.append(f'|{r["dataset"]}|{r["method"]}|{r["correct"]}/{r["n"]}|{r["pipeline_seconds"]:.3f}|{r["physical_output_tokens"]:.1f}|{r["full_reads"]:.3f}|{r["invalid"]}|')
    (dest/'report.md').write_text('\n'.join(lines)+'\n')


def analyze_arc():
    dest = ARC/'development'
    assert (dest/'completion.json').exists() and not (dest/'failure.json').exists()
    p = rt.read(dest/'protocol.json')
    samples = rows(dest/'inputs-no-gold.jsonl')
    records = [rt.read(f) for f in sorted((dest/'rows').glob('*.json'))]
    assert len(records) == 64
    assert all(r['protocol_sha256'] == rt.sha(dest/'protocol.json') for r in records)
    data = summary(records, samples, arc_labels(p['sample_ids']), [q['id'] for q in p['policies']])
    write_report(dest, data, '8道ARC官方validation题；两次运行。接口开发结果，不能用作独立确认。')
    print(json.dumps([r for r in data if r['dataset'] == 'all'], ensure_ascii=False))


def analyze():
    p = verify()
    for gpu in (1, 2):
        assert (OUT/f'gpu{gpu}/completion.json').exists()
        assert not (OUT/f'gpu{gpu}/excluded.json').exists()
    samples = rows(OUT/'inputs-no-gold.jsonl')
    records = [rt.read(f) for f in sorted(OUT.glob('gpu*/rows/*.json'))]
    assert len(records) == p['cases']
    assert all(r['protocol_sha256'] == rt.sha(OUT/'protocol.json') for r in records)
    import sentence_online as base
    labels = {r['sample_id']: r['gold'] for r in rows(base.STREAM/'labels-observed.jsonl')}
    labels.update(arc_labels(p['sample_ids']))
    data = summary(records, samples, labels, [q['id'] for q in p['policies']])
    write_report(OUT, data, '48道开发题，每任务16道；各题方法同卡交错。固定预算与策略档位比较，不自动选择方法，不读取确认集。')
    rt.save(OUT/'completion.json', dict(UTC=rt.now(), state='development_completed', cases=720,
                                      confirmation_started=False))
    print('Completed three-task development analysis')


def watch_analysis():
    verify()
    with (OUT/'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic()+7*3600
        while time.monotonic() < deadline:
            if rt.stop_requested() or (OUT/'STOP').exists():
                rt.save(OUT/'analysis-status.json', dict(UTC=rt.now(), state='stopped'))
                return
            if any((OUT/f'gpu{gpu}/failure.json').exists() for gpu in (1, 2)):
                rt.save(OUT/'analysis-status.json', dict(UTC=rt.now(), state='worker_failed'))
                return
            if all((OUT/f'gpu{gpu}/completion.json').exists() for gpu in (1, 2)):
                try:
                    analyze()
                    rt.save(OUT/'analysis-status.json', dict(UTC=rt.now(), state='completed'))
                except BaseException:
                    rt.save(OUT/'analysis-status.json', dict(UTC=rt.now(), state='failed', error=traceback.format_exc()))
                    raise
                return
            rt.save(OUT/'analysis-status.json', dict(UTC=rt.now(), state='waiting_for_both_shards'))
            time.sleep(15)
        rt.save(OUT/'analysis-status.json', dict(UTC=rt.now(), state='wait_expired'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    for k in ('prepare', 'check', 'launch', 'worker', 'analyze', 'analyze-arc', 'watch-analysis'):
        g.add_argument('--'+k, action='store_true')
    ap.add_argument('--gpu', type=int, choices=(1, 2))
    a = ap.parse_args()
    if a.prepare:
        prepare()
    elif a.check:
        verify(); print('Budget bindings verified')
    elif a.analyze:
        analyze()
    elif a.analyze_arc:
        analyze_arc()
    elif a.watch_analysis:
        watch_analysis()
    else:
        assert a.gpu is not None
        launch(a.gpu) if a.launch else worker(a.gpu)
