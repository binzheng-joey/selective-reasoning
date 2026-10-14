"""Integrated routing and concurrent answer-head reading on one vLLM engine.

Reuses the frozen, hash-bound Engine backend without changing its source.
All wall-time fields measure real online work only when run on a GPU. CPU fake
backend tests check control flow and never enter experiment result directories.
"""
import bisect
import math
import time

import sentence_online as base
from arr_control import head_stop, route


class Engine(base.Engine):
    def __init__(self, protocol):
        self.capture_initial = False
        self.initial_hidden = None
        self.trial_keys = []
        super().__init__(protocol)

    def add(self, ids, cap, kind, count):
        key, rid = super().add(ids, cap, kind, count)
        self.trial_keys.append(key)
        return key, rid

    def capture(self, h):
        logits = super().capture(h)
        if self.capture_initial:
            # Only one initial read is active. Keep the exact answer-cue state.
            for i, rid in enumerate(self.runner.input_batch.req_ids):
                if rid in self.captured:
                    self.initial_hidden = h[i].detach().float().cpu().tolist()
        return logits

    def reset_trial(self):
        assert not self.engine.has_unfinished_requests()
        assert self.llm.reset_prefix_cache(), 'Between-method reset failed'
        self.registry.clear(); self.captured.clear(); self.hidden.clear()
        self.initial_hidden = None; self.keep_natural = False
        self.trial_keys = []

    def integrated(self, sample, policy, gates):
        """policy.mode is direct/always/gated; always incurs no initial read.

        A gated route retains its initial prompt cache into generation. The
        stopwatch spans initial read, gate, generation, checks, and cancellation.
        Every method resets cache once before the stopwatch, never in between.
        """
        validate_policy(policy)
        self.reset_trial()
        torch = self.torch
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        initial, decision, gate_seconds = None, None, 0.
        try:
            if policy['mode'] != 'always':
                self.capture_initial = policy.get('gate') in ('benefit_raw', 'initial_error')
                initial = self.query(self.read_ids(sample['prompt_ids'], []), sample['candidate_count'])
                self.capture_initial = False
                t = time.perf_counter()
                decision = (dict(reason=False, score=None, threshold=None)
                    if policy['mode'] == 'direct' else route(gates, policy['gate'],
                    policy['fraction'], initial['probabilities'], len(sample['prompt_ids']), self.initial_hidden))
                gate_seconds = time.perf_counter()-t
            if decision is not None and not decision['reason']:
                p = initial['probabilities']
                row = dict(prediction=max(range(len(p)), key=p.__getitem__), probabilities=p,
                    generation_tokens=0, physical_output_tokens=1, full_reads=1,
                    trigger_calls=0, generated_ids=[], checks=[], triggers=[],
                    read_prefix_tokens=0, decision_at_tokens=0, extra_tokens_since_read_prefix=0,
                    tokens_after_stop_decision=0, early_stop=False, prefix_answer_accepted=False, natural_terminal=False,
                    sentence_opportunities=0, merged_pending_events=0, initial_reads=1,
                    reader_prompt_tokens=initial['input_tokens'], reader_cached_tokens=initial['cached_tokens'],
                    generator_prompt_tokens=0, generator_cached_tokens=0, reasoning_invoked=False)
            else:
                row = self.reason(sample, policy)
                row['reasoning_invoked'] = True
                row['initial_reads'] = int(initial is not None)
                if initial is not None:
                    row['physical_output_tokens'] += 1
                    row['full_reads'] += 1
                    row['reader_prompt_tokens'] += initial['input_tokens']
                    row['reader_cached_tokens'] += initial['cached_tokens']
            torch.cuda.synchronize()
            row.update(sample_id=sample['sample_id'], dataset=sample['dataset'], policy=policy['id'],
                policy_spec=policy, initial=initial, routing=decision, gate_seconds=gate_seconds,
                pipeline_seconds=time.perf_counter()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved())
            assert not self.engine.has_unfinished_requests()
            return row
        finally:
            # Even a failed read cannot leak a generation request into the next method.
            if self.engine.has_unfinished_requests():
                self.engine.abort_request(self.trial_keys)
            self.capture_initial = False; self.keep_natural = False
            self.initial_hidden = None
            self.registry.clear(); self.hidden.clear(); self.captured.clear()

    def reason(self, sample, policy):
        torch = self.torch
        prompt, count, cap = sample['prompt_ids'], sample['candidate_count'], policy['cap']
        schedule = policy['schedule']
        self.keep_natural = schedule.startswith('learned')
        detector = base.StreamDetector()
        lengths, counts, prefix, previous = [0], [0], [], ''
        ordinal = last_read = merged = 0
        pending = inflight = None
        checks, triggers = [], []
        generation_finished = natural = stopped = prefix_accepted = False
        detector_seconds = trigger_seconds = abort_seconds = 0.
        hook_start = self.hook_host_seconds
        begin = time.perf_counter(); read_at = decision_at = None; gen_cached = 0
        genkey, genrid = self.add(prompt, cap, 'generate', count)
        try:
            for _ in range(cap*8+1000):
                if time.perf_counter()-begin > policy.get('wall_limit_seconds', 300):
                    raise TimeoutError('Per-question online limit')
                outputs = self.engine.step() if self.engine.has_unfinished_requests() else []
                for o in sorted(outputs, key=lambda o: o.request_id != genkey):
                    if o.request_id == genkey:
                        new = list(o.outputs[0].token_ids)
                        assert new[:len(prefix)] == prefix and len(new) <= cap
                        # Terminal-only controls do not pay for sentence decoding.
                        if schedule != 'terminal':
                            ts = time.perf_counter()
                            for k in range(len(prefix)+1, len(new)+1):
                                text = self.tok.decode(new[:k], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                                if text.endswith('\ufffd'): continue
                                assert text.startswith(previous)
                                lengths.append(len(text)); counts.append(k)
                                for end in detector.update(text):
                                    at = bisect.bisect_left(lengths, end)
                                    if text[end:lengths[at]].strip(): continue
                                    ordinal += 1
                                    if pending is not None: merged += 1
                                    pending = dict(prefix_tokens=counts[at], observable_tokens=k, ordinal=ordinal)
                                previous = text
                            detector_seconds += time.perf_counter()-ts
                        prefix = new; generation_finished = o.finished
                        gen_cached = o.num_cached_tokens or 0
                        if o.finished: natural = o.outputs[0].finish_reason != 'length'
                    elif inflight is not None and o.request_id == inflight['key'] and o.finished:
                        q = self.captured.pop(inflight['rid'])
                        assert list(o.outputs[0].token_ids) == [q['native_argmax']]
                        q.update({k:v for k,v in inflight.items() if k not in ('key','rid','start')})
                        q.update(cached_tokens=o.num_cached_tokens or 0, completed_at_tokens=len(prefix),
                                 response_seconds=time.perf_counter()-inflight['start'])
                        checks.append(q); self.registry.pop(inflight['rid']); inflight = None
                        accepted = q['terminal'] or head_stop(q['probabilities'], policy['stop_rule'], policy['stop_threshold'])
                        q['stop_accepted'] = accepted
                        if accepted:
                            prefix_accepted = not q['terminal']
                            stopped = prefix_accepted and not generation_finished
                            read_at, decision_at = q['prefix_tokens'], len(prefix)
                            ts = time.perf_counter()
                            if not generation_finished: self.engine.abort_request([genkey])
                            torch.cuda.synchronize(); abort_seconds += time.perf_counter()-ts
                            break
                if read_at is not None: break
                if inflight is None and pending is not None and not generation_finished:
                    ev = pending; pk, ord_ = ev['prefix_tokens'], ev['ordinal']
                    perform = False
                    if schedule.startswith('every'):
                        perform = ord_-last_read >= int(schedule[5:]); pending = None
                    else:
                        eligible = (ord_-last_read >= 4 if schedule == 'learned_gap4' else ord_%4 == 0)
                        if not eligible: pending = None
                        elif pk in self.hidden:
                            ts = time.perf_counter(); tr = self.tr
                            value = float(self.hidden[pk].float()@self.trigger_weight)
                            value += tr['position_weight']*pk/1024 + tr['task_weight']*float(sample['dataset']=='logiqa2')+tr['bias']
                            score = 1/(1+math.exp(-max(-80,min(80,value))))
                            trigger_seconds += time.perf_counter()-ts
                            triggers.append(dict(**ev, score=score, available_tokens=len(prefix)))
                            perform = score >= policy['trigger_threshold']; pending = None
                    if perform:
                        ids = self.read_ids(prompt, prefix[:pk]); key, rid = self.add(ids, 1, 'read', count)
                        inflight = dict(key=key, rid=rid, prefix_tokens=pk, ordinal=ord_, terminal=False,
                            input_tokens=len(ids), submitted_at_tokens=len(prefix), start=time.perf_counter())
                        last_read = ord_
                if generation_finished and inflight is None:
                    ids = self.read_ids(prompt, prefix); key, rid = self.add(ids, 1, 'read', count)
                    inflight = dict(key=key, rid=rid, prefix_tokens=len(prefix), ordinal=ordinal, terminal=True,
                        input_tokens=len(ids), submitted_at_tokens=len(prefix), start=time.perf_counter())
                assert self.engine.has_unfinished_requests(), 'Controller has no active request'
            else:
                raise RuntimeError('Online iteration limit')
        finally:
            if self.engine.has_unfinished_requests():
                keys = [genkey] + ([inflight['key']] if inflight else [])
                self.engine.abort_request(keys)
        assert checks and read_at is not None and not self.engine.has_unfinished_requests()
        p = checks[-1]['probabilities']
        return dict(prediction=max(range(count), key=p.__getitem__), probabilities=p,
            generation_tokens=len(prefix), physical_output_tokens=len(prefix)+len(checks),
            full_reads=len(checks), trigger_calls=len(triggers), generated_ids=prefix,
            read_prefix_tokens=read_at, decision_at_tokens=decision_at,
            extra_tokens_since_read_prefix=len(prefix)-read_at,
            tokens_after_stop_decision=len(prefix)-decision_at, early_stop=stopped, prefix_answer_accepted=prefix_accepted,
            natural_terminal=natural, sentence_opportunities=ordinal, merged_pending_events=merged,
            checks=checks, triggers=triggers, generator_prompt_tokens=len(prompt), generator_cached_tokens=gen_cached,
            reader_prompt_tokens=sum(q['input_tokens'] for q in checks),
            reader_cached_tokens=sum(q['cached_tokens'] for q in checks),
            detector_host_seconds=detector_seconds, trigger_host_seconds=trigger_seconds,
            hook_host_seconds=self.hook_host_seconds-hook_start, abort_seconds=abort_seconds)


def validate_policy(p):
    assert p['mode'] in ('direct', 'always', 'gated')
    assert p['schedule'] in ('terminal','every1','every4','every8','learned_gap4','learned_tick4')
    assert p['stop_rule'] in ('never','entropy','confidence')
    assert 0 <= p['stop_threshold'] <= 1 and 1 <= p['cap'] <= 8192
    if p['schedule'].startswith('learned'):
        assert p['cap'] == 1024, 'Existing trigger was calibrated with position/1024'
        assert 0 <= p['trigger_threshold'] <= 1
    if p['mode'] == 'gated':
        assert p['gate'] in ('margin','confidence','entropy','benefit_raw','initial_error')
        assert 0 <= p['fraction'] <= 1
