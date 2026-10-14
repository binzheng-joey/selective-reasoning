"""Post-completion analysis of prespecified supplements; reuse audited labels.

No new source gold is decoded. Accuracy uses first executions; timing repeats
are summarized separately and never inflate the question count.
"""
import argparse
import copy
import itertools
import time
import traceback
from pathlib import Path
import run as campaign
import arr_formal_analysis as common
from arr_essential_analysis import paired_comparisons

rt, OUT = campaign.rt, campaign.OUT
SOURCE = Path(__file__).resolve()


def analyze(family):
    p = campaign.prepare(family)
    folder = OUT / family
    assert (folder / 'completion.json').exists()
    rows = [rt.read(f) for f in sorted((folder / 'rows').glob('*.json'))]
    jobs = rt.read(folder / 'jobs.json')
    checksum = rt.sha(folder / 'protocol.json')
    assert len(rows) == p['total'] and {r['case_id'] for r in rows} == {j['case_id'] for j in jobs}
    assert all(r['protocol_sha256'] == checksum for r in rows)
    methods = [q['id'] for q in p['policies']]
    subjects = {s['sample_id']: s for s in campaign.formal.rows(p['input_file']) if s['sample_id'] in p['sample_ids']}
    full_meta = {s['sample_id']: s for s in rt.read(campaign.formal.OUT / 'metadata.json')}
    meta = {sid: dict(full_meta[sid], candidate_count=s['candidate_count']) for sid, s in subjects.items()}
    previous = campaign.formal.MAIN / 'completed-groups-analysis-20261011-v1'
    audit = rt.read(previous / 'label-audit.json')
    assert audit['state'] == 'passed' and rt.sha(previous / 'labels.json') == audit['labels_sha256']
    assert audit['inputs_sha256'] == rt.sha(campaign.formal.OUT / 'inputs-no-gold.jsonl')
    labels = {sid: label for sid, label in rt.read(previous / 'labels.json').items() if sid in subjects}
    assert set(labels) == set(meta)
    primary = [r for r in rows if r['repeat'] == 0]
    result = common.summarize(primary, meta, labels, methods)
    if family == 'controls':
        contrasts = list(itertools.combinations(methods, 2))
    else:
        contrasts = [(m, methods[0]) for m in methods[1:]]
    result['paired'] = paired_comparisons(primary, meta, labels, contrasts=contrasts)
    repeated_ids = set(p['repeat_ids'])
    repeated_meta = {sid: m for sid, m in meta.items() if sid in repeated_ids}
    repeated = []
    for sid in repeated_meta:
        for method in methods:
            runs = [r for r in rows if r['sample_id'] == sid and r['policy'] == method]
            assert sorted(r['repeat'] for r in runs) == [0,1,2]
            row = copy.deepcopy(next(r for r in runs if r['repeat'] == 0))
            for key in ('pipeline_seconds', 'physical_output_tokens', 'generation_tokens', 'full_reads', 'trigger_calls',
                        'request_input_tokens', 'request_cached_tokens', 'reader_prompt_tokens', 'reader_cached_tokens',
                        'generator_prompt_tokens', 'generator_cached_tokens'):
                if all(key in r for r in runs):
                    row[key] = sum(r[key] for r in runs)/3
            repeated.append(row)
    repeated_pairs = paired_comparisons(repeated, repeated_meta, labels, contrasts=contrasts)
    result.update(UTC=rt.now(), scope=p['scope'], protocol_sha256=checksum,
        labels_sha256=audit['labels_sha256'], analyzer_sha256=rt.sha(SOURCE),
        repeated_timing48=repeated_pairs, repeated_timing_subjects=len(repeated_meta),
        accuracy='First execution only. Timing48 averages repetitions per subject; no answer voting.',
        interval_note='Pointwise stratified material-group bootstrap; not equivalence or simultaneous frontier confidence')
    rt.save(folder / 'analysis.json', result)
    lines = ['# 本轮补充实验结果', '', p['scope'], '',
        '|任务|方法|正确/题数|准确率|秒/题|实际输出token/题|读取/题|提前停止比例|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['summaries']:
        lines.append(f'|{r["dataset"]}|{r["method"]}|{r["correct"]}/{r["n"]}|{100*r["accuracy"]:.2f}%|{r["pipeline_seconds"]:.3f}|{r["physical_output_tokens"]:.1f}|{r["full_reads"]:.3f}|{100*r["early_stop_rate"]:.1f}%|')
    lines += ['', '逐题配对区间、分任务统计和固定48题重复计时见 analysis.json。不同GPU的结果不拼接为配对加速比。']
    (folder / 'report.md').write_text('\n'.join(lines)+'\n')
    rt.save(folder / 'analysis-completion.json', dict(UTC=rt.now(), questions=len(meta),
        primary_records=len(primary), repeated_subjects=len(repeated_meta), family=family))


def watch():
    pending = {'controls', 'serial', 'calibrated'}
    deadline = time.monotonic()+50*3600
    while pending and time.monotonic() < deadline:
        if (OUT/'STOP').exists() or rt.stop_requested():
            return
        for family in list(pending):
            folder = OUT / family
            if (folder / 'analysis-completion.json').exists():
                pending.remove(family)
            elif (folder / 'completion.json').exists():
                try:
                    analyze(family)
                except BaseException:
                    rt.save(folder / 'analysis-failure.json', dict(UTC=rt.now(), error=traceback.format_exc()))
                pending.remove(family)
        rt.save(OUT / 'analysis-watch-status.json', dict(UTC=rt.now(), pending=sorted(pending),
            state='waiting' if pending else 'finished'))
        if pending:
            time.sleep(30)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--watch',action='store_true')
    p.add_argument('--family',choices=('controls','serial','calibrated'))
    a=p.parse_args()
    watch() if a.watch else analyze(a.family)
