"""Explicit variable-choice readout candidate, separate from frozen pilots.

The trained head supports up to four candidates. For more candidates, use the
native LM logits of all digit tokens at the same answer position. This fixed,
label-free fallback is shared by every head-based policy and explicitly logged;
it is not a trained fifth output or evidence of a five-class trained readout.
"""
from arr_repaired_final import Engine as RepairedEngine


class Engine(RepairedEngine):
    def __init__(self, protocol):
        super().__init__(protocol)
        ids = [self.tok.encode(str(i), add_special_tokens=False) for i in range(1, 10)]
        assert all(len(x) == 1 for x in ids), 'Candidate indices must be single native tokens'
        self.candidate_ids = [x[0] for x in ids]

    def capture(self, h):
        # Reuse the exact frozen hook, including initial-state capture and the
        # generator trigger states. Restore any temporarily capped registry count.
        wide = []
        for i, rid in enumerate(self.runner.input_batch.req_ids):
            info = self.registry.get(rid)
            if info and info['kind'] == 'read' and info['count'] > self.weight.shape[0]:
                assert info['count'] <= len(self.candidate_ids)
                wide.append((i, rid, info['count']))
                info['count'] = self.weight.shape[0]
        try:
            logits = super().capture(h)
            for i, rid, count in wide:
                if rid in self.captured:
                    scores = logits[i, self.candidate_ids[:count]].float()
                    self.captured[rid].update(probabilities=scores.softmax(-1).cpu().tolist(),
                        reader_kind='native_candidate_logits_fallback', candidate_count=count)
            for rid in self.captured:
                self.captured[rid].setdefault('reader_kind', 'trained_answer_position_linear')
            return logits
        finally:
            for _, rid, count in wide:
                self.registry[rid]['count'] = count
