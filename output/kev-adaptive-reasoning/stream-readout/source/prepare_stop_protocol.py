"""Prepare label-free material-group folds and an executable-stage dependency plan.

This does not select a reader or launch stopping/online experiments.
"""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'stopping-plan'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    samples = [json.loads(s) for s in (ROOT / 'samples.jsonl').read_text().splitlines()]
    assert len(samples) == 1280
    train = [r for r in samples if r['split'] == 'train']
    dev = [r for r in samples if r['split'] == 'cal']
    assert len(train) == 1024 and len(dev) == 256
    train_groups = {r['component_id'] for r in train}
    dev_groups = {r['component_id'] for r in dev}
    assert not train_groups & dev_groups
    # Existing split currently has one question per material component.
    assert len(train_groups) == len(train) and len(dev_groups) == len(dev)
    assignment = {}
    for dataset in sorted({r['dataset'] for r in train}):
        group_ids = [r['component_id'] for r in train if r['dataset'] == dataset]
        group_ids.sort(key=lambda g: hashlib.sha256(('stop-oof-20261009:' + g).encode()).hexdigest())
        for i, group in enumerate(group_ids):
            assignment[group] = i % 4
    folds = [dict(sample_id=r['sample_id'], component_id=r['component_id'], dataset=r['dataset'],
                  split='train', held_out_fold=assignment[r['component_id']]) for r in train]
    counts = {str(i): dict(collections.Counter(r['dataset'] for r in folds if r['held_out_fold'] == i)) for i in range(4)}
    assert all(sum(v.values()) == 256 for v in counts.values())
    OUT.mkdir(exist_ok=True)
    (OUT / 'oof-folds.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in folds))
    protocol = dict(
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        state='prepared_not_launched_requires_frozen_reader_and_service_integration',
        scope='Existing train1024 and observed-dev256 only; post-hoc gate512 excluded from fitting and selection',
        source_sha256={str(ROOT / 'samples.jsonl'): sha(ROOT / 'samples.jsonl'),
                       str(Path(__file__)): sha(Path(__file__)),
                       str(OUT / 'oof-folds.jsonl'): sha(OUT / 'oof-folds.jsonl')},
        fold_sizes=counts,
        dependencies=['Complete four-reader frozen backbone comparison and causal sentence diagnostics.',
                      'Freeze reader architecture and training recipe using observed-dev results, recording task-level metrics.',
                      'Cross-fit only the selected reader on train1024; each training trajectory read by a head fitted without that material group.',
                      'Generate training sentence-prefix probabilities from those out-of-fold heads.',
                      'Train stopping estimates on out-of-fold train trajectories; select operating curves on observed-dev256.',
                      'Verify deployed-backend hidden states and cached readout equivalence before actual latency experiments.'],
        labels=dict(current_correct='1[argmax reader(prefix_t) == gold]',
                    full_budget_correct='1[argmax same reader(actual terminal at fixed cap) == gold]',
                    continuation_gain='full_budget_correct - current_correct, in {-1,0,1}',
                    lost_correct_risk='1[current wrong AND fixed-budget terminal correct]',
                    avoidable_degradation='1[current correct AND fixed-budget terminal wrong]',
                    meaning='Continuation target initially means the frozen full-budget policy, not max correctness over future checkpoints.'),
        stop_features=['current candidate probabilities, margin, normalized entropy',
                       'candidate count', 'current generated token count',
                       'answer agreement and confidence change with previous performed read',
                       'elapsed tokens and sentence boundaries since previous performed read'],
        scheduler_features=['latest completed read features', 'tokens/sentences elapsed since that read',
                            'reader pending/busy state'],
        forbidden_at_inference=['gold', 'future tokens or reader states', 'probabilities from checks not actually performed',
                                'true full-budget length before EOS', 'teacher-forced timing presented as service timing'],
        stop_arms=['confidence threshold curve',
                   'estimated current correctness AND estimated continuation value, with operating thresholds swept on observed-dev'],
        schedules=['every >=4 newly completed sentences', 'every >=8 newly completed sentences',
                   'adaptive 4/8/16 newly completed sentences based on previous read only; policy fixed before confirmation'],
        controls=['same initial route + terminal-only read', 'same route + checks but no mid-generation stop'],
        inference_contract=dict(generator='Same frozen native model, prompt, greedy sampling, backend and cap in all arms',
                                initial_route='Freeze before within-route early-stop comparisons; report initial-only and combined systems separately',
                                asynchronous='At most one pending read; coalesce to newest fully observable sentence prefix',
                                prefix='Complete cumulative prefix up to that sentence, never just an isolated latest sentence',
                                finish='Use every actually generated terminal token consistently in all arms',
                                current_cap=1024,
                                cap_scope='Initial controlled study uses existing1024 cap, not a claim about uncapped or8192 reasoning'),
        cost_contract=['Wall time from question arrival to final answer and physical generation stop, both recorded',
                       'Physical generated tokens including asynchronous overshoot',
                       'Number and accumulated duration of actual reader calls',
                       'Generation, readout, scheduling, transfer/cache costs reported separately',
                       'Do not add overlapped GPU durations and call the sum end-to-end wall time',
                       'Include hidden-state extraction/export and per-token accumulator updates',
                       'Count two-GPU consumption if used; same hardware layout for controls',
                       'Paired randomized/interleaved repeated online runs and material-group uncertainty'],
        evaluation=['Accuracy-resource curves, no fixed noninferiority tolerance chosen',
                    'Per-task accuracy and cost as well as aggregate',
                    'Calibration and lost-correct/gained-correct counts relative to terminal-only',
                    'Retrospective earliest stable-correct point is an oracle diagnostic only'],
        independent_data='Existing observed-dev is development evidence. Future revised method requires a separately frozen confirmation set; do not take over the parent confirmation run.',
        no_new_training_started=True)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 读取器完成后的早停与检查调度实验', '',
             '状态：依赖已列出、训练材料组四折已固定；尚未选择读取器，也未启动停止器训练或在线实验。', '',
             '## 数据与执行顺序', '',
             '1. 完成当前四种读取器、三种子对照和开发256题的句末轨迹。先核对各任务的效果、校准和读取成本，再固定读取器。',
             '2. 原训练1024题按材料组分为四折，每折256题（PW/LQ各128）。为所选读取器训练四个折外版本；每题的停止器训练特征必须来自没见过该材料组标签的读取头。',
             '3. 在折外训练轨迹上拟合停止收益/风险，在原开发256题上展示曲线。开发集已经参与读取器选择，因此这里只是开发证据。',
             '4. 验证同一在线生成后端的隐藏状态接口和数值一致性后，再比较真实在线耗时。现有teacher-forced特征不作为在线读取耗时。', '',
             '新门控512题只做已完成验证的事后诊断，不进入此处拟合或选模；原确认实验继续独立运行。', '',
             '## 停止目标', '',
             '先固定“继续”的含义为运行到共同预算或自然结束。监督标签为当前正确性、终点正确性及二者差值；不以“未来某处曾经正确”替代继续运行的实际结果。',
             '当前答错而终点答对是提前停止损失；当前答对而终点答错是停止可能避免的退化。二者分开记录。',
             '主比较保留置信度阈值曲线，并加入当前正确性估计与继续收益估计。预测继续收益低不能单独当作答案可靠；“放弃无效推理”与“可靠提前作答”分别统计。', '',
             '## 检查调度和控制', '',
             '比较每新增至少4句、8句和根据上次读数调整4/8/16句间隔。调度只能使用上次实际完成检查的结果与此后已出现的token/句末数；不能偷用未执行检查的置信度。',
             '生成连续进行，最多一项读取等待，合并为最近可观察的完整句末累积前缀。到终点时所有方案读取相同的实际尾部。',
             '固定初始分流，设置末尾只读一次、检查但不中途停止两种控制，再组合两类停止机制与三类调度。', '',
             '## 资源核算与推进条件', '',
             '首先沿用1024-token共同上限做受控比较；这不是8192-token或自然完整推理的结论。',
             '计入隐藏状态导出、每token聚合、调度、缓存、传输和停止后多生成的token；同时报告答案返回时间及生成实际停止时间。',
             '在线实验采用相同后端、硬件布局和初始分流，按题配对、交错顺序并重复测量。绘制准确率—成本曲线，不预设可接受降幅。',
             '读取器结果未出前不启动依赖选型的下一轮；若自然状态读取显著失效，应先修复读取质量，避免把错误读数交给停止器。']
    (OUT / 'plan.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(output=str(OUT), state=protocol['state'], fold_sizes=counts)))


if __name__ == '__main__':
    main()
