"""Bounded idle queues plus development-only baseline selection.

No foreign signals, no VRAM holding, no evaluation-label selection. Child workers
abort if another GPU owner appears; no automatic retry after a technical failure.
"""
import argparse
import copy
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
import run as campaign

rt, OUT, SOURCE = campaign.rt, campaign.OUT, Path(__file__).resolve()


def select():
    folder = OUT / 'development'
    p = campaign.prepare('development')
    assert (folder / 'completion.json').exists()
    if (folder / 'selection.json').exists():
        return rt.read(folder / 'selection.json')
    samples = campaign.formal.rows(p['input_file'])
    rows = [rt.read(f) for f in sorted((folder / 'rows').glob('*.json'))]
    assert len(rows) == p['total']
    assert all(r['protocol_sha256'] == rt.sha(folder / 'protocol.json') for r in rows)
    import sentence_online as base
    from arr_budget_baselines import arc_labels, summary
    chosen_ids = {s['sample_id'] for s in samples}
    labels = {r['sample_id']: r['gold'] for r in campaign.formal.rows(base.STREAM / 'labels-observed.jsonl') if r['sample_id'] in chosen_ids}
    labels.update(arc_labels(chosen_ids))
    assert set(labels) == chosen_ids
    data = dict(UTC=rt.now(), scope='Development48 only',
                summaries=summary(rows, samples, labels, [q['id'] for q in p['policies']]))
    rt.save(folder / 'analysis.json', data)
    summaries = {r['method']: r for r in data['summaries'] if r['dataset'] == 'all'}
    by_id = {q['id']: q for q in p['policies']}
    # Retain the original learned controller's three preselected operating points.
    chosen = [copy.deepcopy(by_id[f'dev-learned-q{q}']) for q in (70, 85, 95)]
    for threshold in (70, 85, 95):
        candidates = [summaries[f'dev-margin-gap{gap}-q{threshold}'] for gap in (4, 8, 16)]
        best = min(candidates, key=lambda r: (-r['correct'], r['pipeline_seconds'], r['method']))
        chosen.append(copy.deepcopy(by_id[best['method']]))
    deer = [summaries[f'dev-deer-q{q}'] for q in (80, 85, 90, 95)]
    effective = [r for r in deer if r['early_stop_rate'] > 0]
    if effective:
        fastest = min(effective, key=lambda r: (r['pipeline_seconds'], -r['correct'], r['method']))
        best = min(effective, key=lambda r: (-r['correct'], r['pipeline_seconds'], r['method']))
        deer_ids = list(dict.fromkeys([fastest['method'], best['method'], 'dev-deer-q95']))
    else:
        deer_ids = ['dev-deer-q80', 'dev-deer-q95']
    chosen += [copy.deepcopy(by_id[m]) for m in deer_ids]
    chosen += [copy.deepcopy(by_id['forced-terminal-prefill-1024'])]
    # All choices are frozen before the observed768 extension is collected.
    result = dict(UTC=rt.now(), development_questions=48, evaluation_questions=768,
        development_protocol_sha256=rt.sha(folder / 'protocol.json'),
        selector_source_sha256=rt.sha(SOURCE), evaluation_policies=chosen,
        rule='Keep learned thresholds70/85/95 unchanged. At each threshold select simple gap4/8/16 by maximum development correct then minimum latency. DEER retain fastest and highest-accuracy effective development points plus default95.',
        baseline_status='effective_DEER_development_interval' if effective else 'no_DEER_early_exit_in_prespecified_range; do not claim functional early-stop comparator',
        scope='Development-only selection, evaluation is an observed768 extension, not independent confirmation',
        selected_development_summaries=[summaries[q['id']] for q in chosen])
    rt.save(folder / 'selection.json', result)
    return result


def queue(family, gpu):
    folder = OUT / family
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'queue-owner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rt.save(folder / 'queue-owner.json', dict(UTC=rt.now(), pid=os.getpid(), gpu=gpu,
            family=family, wait_limit_hours=24, interval_seconds=30))
        try:
            campaign.launch(family, gpu, wait=True)
            if family != 'development':
                return
            # Development worker publishes completion before freeing engine;
            # launch() still independently checks two actual idle inventories.
            deadline = time.monotonic() + 24*3600
            while time.monotonic() < deadline:
                if (OUT / 'STOP').exists() or rt.stop_requested():
                    raise InterruptedError('Stop requested')
                state = rt.read(folder / 'status.json')['state']
                if state == 'interrupted_or_failed':
                    raise RuntimeError('Development failed; no baseline promotion')
                if (folder / 'completion.json').exists():
                    result = select()
                    campaign.prepare('calibrated')
                    rt.save(OUT / 'calibrated/dependency.json', dict(UTC=rt.now(),
                        development_complete=True, selected_methods=len(result['evaluation_policies']),
                        baseline_status=result['baseline_status']))
                    campaign.launch('calibrated', gpu, wait=True)
                    return
                time.sleep(30)
            raise TimeoutError('Development follow-up wait expired')
        except BaseException:
            rt.save(folder / 'queue-failure.json', dict(UTC=rt.now(), error=traceback.format_exc()))
            raise


def start(families):
    with (OUT / 'queue-launch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = []
        for family, gpu in [('controls', 0), ('development', 1), ('serial', 2)]:
            if family not in families:
                continue
            campaign.prepare(family)
            folder = OUT / family
            assert not (folder / 'queue-launcher.json').exists(), 'Queue already exists'
            command = [str(rt.PYTHON), '-B', str(SOURCE), '--queue', '--family', family, '--gpu', str(gpu)]
            with (folder / 'queue.log').open('a') as log:
                child = subprocess.Popen(command, cwd=campaign.PROJECT, env=rt.env_for(gpu),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            value = dict(UTC=rt.now(), pid=child.pid, command=command, gpu=gpu, family=family)
            rt.save(folder / 'queue-launcher.json', value)
            result.append(value)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--start', action='store_true')
    g.add_argument('--queue', action='store_true')
    g.add_argument('--select', action='store_true')
    p.add_argument('--family', choices=('controls', 'development', 'serial'))
    p.add_argument('--gpu', type=int, choices=(0, 1, 2))
    p.add_argument('--families', nargs='+', default=['controls', 'development'])
    a = p.parse_args()
    if a.start:
        start(a.families)
    elif a.queue:
        queue(a.family, a.gpu)
    else:
        print(json.dumps(select(), ensure_ascii=False))
