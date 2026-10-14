"""Fit a candidate trigger without task identity using existing training data.

Reuses the training-only PCA/scaler and fits the same weighted logistic target.
Writes a separate candidate; does not replace the trigger of a running pilot.
No ARC labels or independent confirmation inputs are accessed.
"""
from collections import Counter
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from threadpoolctl import threadpool_limits

import sentence_policies as sp
import arr_gpu_queue as io

OUT = io.ROOT / 'arr-sprint/main-table/task-agnostic-trigger'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only preparation'
    if (OUT / 'completion.json').exists():
        print('Candidate already fitted; not overwriting')
        return
    original_save, original_status = sp.save, sp.status
    sp.save = lambda name, obj: io.save(OUT / name, obj)
    sp.status = lambda phase, **kw: io.save(OUT / 'status.json', dict(UTC=io.now(), state='running', phase=phase, **kw))
    source = io.ROOT / 'sentence-policies/trigger-models.joblib'
    bindings = [Path(__file__).resolve(), source, sp.COLLECT / 'feature-index.json',
                sp.COLLECT / 'work.json', io.ROOT / 'source/sentence_policies.py']
    io.save(OUT / 'protocol.json', dict(UTC=io.now(), source_bindings={str(p): io.sha(p) for p in bindings},
        target='OOF answer read correct and normalized entropy certainty >= .85',
        features='Existing training-only PCA64 natural hidden state and position/1024; no task identity',
        C=.01, deployment='Separate unselected candidate; no live artifact replacement',
        training='Existing 1024 training questions', development='Existing observed 256 questions',
        ARC_labels_used=False, independent_confirmation_access=False))
    try:
        d = sp.load_data()
        old = joblib.load(source)
        with threadpool_limits(limits=2):
            reduced = old['pca'].transform(old['scaler'].transform(d['hidden']))
            x = np.column_stack([reduced, d['position']])
            certainty = np.asarray([sp.cert(p[:r['candidate_count']], 'entropy')
                                    for p, r in zip(d['probabilities'], d['meta'])])
            target = ((certainty >= .85) & d['correct'].astype(bool)).astype(int)
            mask = d['train_mask']
            model = sp.e.fit_lr(x[mask], target[mask], C=.01, weight=d['weights'][mask])
            projection = model.coef_[0, :64] @ old['pca'].components_
            weight = projection / old['scaler'].scale_
            bias = float(model.intercept_[0] - projection @ old['pca'].mean_ - weight @ old['scaler'].mean_)
            folded = d['hidden'] @ weight + d['position'] * model.coef_[0, -1] + bias
            error = float(np.max(np.abs(model.decision_function(x) - folded)))
            assert error < 1e-7, 'Affine fold mismatch'
            new_prob = model.predict_proba(x)[:, 1]
            old_x = np.column_stack([reduced, d['position'], d['task']])
            old_prob = old['models']['0.85']['model'].predict_proba(old_x)[:, 1]
        summaries = []
        for split, selected in [('train', mask), ('development', ~mask)]:
            ids = [r['sample_id'] for r, keep in zip(d['meta'], selected) if keep]
            counts = Counter(ids)
            weights = np.asarray([1. / counts[s] for s in ids])
            for name, prob in [('existing_task_feature', old_prob), ('without_task_feature', new_prob)]:
                y, score = target[selected], prob[selected]
                summaries.append(dict(split=split, model=name, questions=len(counts), prefixes=len(ids),
                    question_weighted_auroc=float(roc_auc_score(y, score, sample_weight=weights)),
                    question_weighted_log_loss=float(log_loss(y, score, sample_weight=weights, labels=[0, 1])),
                    event_trigger_rate_at_025=float(np.average(score >= .25, weights=weights))))
        joblib.dump(dict(model=model, scaler=old['scaler'], pca=old['pca']), OUT / 'model.joblib')
        io.save(OUT / 'trigger.json', dict(hidden_weight=weight.tolist(), bias=bias,
            position_weight=float(model.coef_[0, -1]), task_weight=0., uses_task_identity=False,
            entropy_threshold=.85, candidate_only=True, GPU_validated=False))
        io.save(OUT / 'analysis.json', dict(UTC=io.now(), folded_max_logit_difference=error,
            summaries=summaries, scope='Prefix target diagnostic; not end-to-end accuracy, latency or an ARC result',
            selected_for_main_table=False, independent_confirmation_access=False))
        io.save(OUT / 'completion.json', dict(UTC=io.now(), state='completed', GPU=False,
            train_questions=1024, development_questions=256, independent_confirmation_access=False))
        io.save(OUT / 'status.json', dict(UTC=io.now(), state='completed_candidate_requires_online_validation'))
        print('Fitted task-agnostic candidate; affine max logit difference', error)
    finally:
        sp.save, sp.status = original_save, original_status


if __name__ == '__main__':
    main()
