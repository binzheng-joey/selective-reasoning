"""Author-mechanism adaptations to this project's fixed-choice task interface.

AC uses causal sentence checkpoints and generated answers, not the answer head.
DEER uses the released Qwen3 transition/trial-confidence/rollback mechanism.
Backend and task adaptations are recorded; neither is claimed to reproduce the
authors' published numbers. DEER-Pro branch-parallel decoding is not implemented.
"""
import bisect
import time

import sentence_online as base
from arr_control import Agreement, deer_confidence, parse_answer, route
from arr_integrated_online import Engine as IntegratedEngine


class Engine(IntegratedEngine):
    def request(self, ids, cap, stop=None, logprobs=False, include_stop=False):
        """Synchronous trial/chunk request; retained prefix cache, counted cost."""
        assert not self.engine.has_unfinished_requests()
        self.serial += 1
        key = 'prior-' + str(self.serial)
        self.trial_keys.append(key)
        params = self.SP(temperature=0., top_p=1., top_k=-1, max_tokens=cap,
            seed=20261009, detokenize=True, stop=stop,
            include_stop_str_in_output=include_stop, logprobs=1 if logprobs else None,
            presence_penalty=0., frequency_penalty=0., repetition_penalty=1.)
        self.engine.add_request(key, dict(prompt_token_ids=ids), params)
        final = None; begin = time.perf_counter()
        while self.engine.has_unfinished_requests():
            if time.perf_counter()-begin > 300:
                self.engine.abort_request([key]); raise TimeoutError('Prior request exceeded limit')
            for out in self.engine.step():
                if out.request_id == key and out.finished: final = out
        assert final is not None
        self.torch.cuda.synchronize()
        out = final.outputs[0]
        probs = [float(lp[t].logprob) for t,lp in zip(out.token_ids,out.logprobs)] if logprobs else []
        if logprobs: assert len(probs) == len(out.token_ids)
        result = dict(token_ids=list(out.token_ids), text=out.text,
            finish_reason=out.finish_reason, stop_reason=out.stop_reason,
            logprobs=probs, input_tokens=len(ids), cached_tokens=final.num_cached_tokens or 0,
            seconds=time.perf_counter()-begin)
        self.prior_requests.append(result)
        return result

    def cue(self, generated, trial=False):
        # Preserve the existing prompt's one-based candidate numbering.
        if trial:
            return '\n**Final Answer**\nThe final candidate index is \\boxed'
        end = self.tok.convert_tokens_to_ids('</think>')
        return ('' if end in generated else '\n</think>')+'\n\nGive only the final candidate index in the format <FINAL>index</FINAL>.'

    def final_answer(self, prompt, generated, policy):
        ids = prompt+generated+self.tok.encode(self.cue(generated), add_special_tokens=False)
        return self.request(ids, policy.get('answer_cap',64))

    def prior(self, sample, policy, gates):
        assert policy['kind'] in ('answer_consistency','deer','forced_terminal')
        assert policy['mode'] in ('always','gated')
        self.reset_trial(); self.prior_requests = []
        self.torch.cuda.synchronize(); self.torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter(); initial = decision = None
        try:
            if policy['mode'] == 'gated':
                self.capture_initial = policy['gate'] in ('benefit_raw','initial_error')
                initial = self.query(self.read_ids(sample['prompt_ids'], []),sample['candidate_count'])
                self.capture_initial = False
                decision = route(gates,policy['gate'],policy['fraction'],initial['probabilities'],
                                 len(sample['prompt_ids']),self.initial_hidden)
            if decision and not decision['reason']:
                p = initial['probabilities']
                row = dict(prediction=max(range(len(p)),key=p.__getitem__),reasoning_invoked=False,
                    early_stop=False,generation_tokens=0,generated_ids=[],full_reads=1,
                    physical_output_tokens=1,trial_reads=0,checks=[],trigger_calls=0)
            else:
                if policy['kind'] == 'deer': row = self.deer(sample,policy)
                elif policy['kind'] == 'answer_consistency': row = self.consistency(sample,policy)
                else:
                    generation = self.request(sample['prompt_ids'],policy['cap'])
                    answer = self.final_answer(sample['prompt_ids'],generation['token_ids'],policy)
                    row = dict(prediction=parse_answer(answer['text'],sample['candidate_count']),
                        early_stop=False,generation_tokens=len(generation['token_ids']),
                        generated_ids=generation['token_ids'],trial_reads=0,checks=[],full_reads=1)
                row.update(reasoning_invoked=True,trigger_calls=0)
                row['full_reads'] += int(initial is not None)
                row['physical_output_tokens'] = sum(len(r['token_ids']) for r in self.prior_requests)+int(initial is not None)
            self.torch.cuda.synchronize()
            row.update(sample_id=sample['sample_id'],dataset=sample['dataset'],policy=policy['id'],
                policy_spec=policy,initial=initial,routing=decision,pipeline_seconds=time.perf_counter()-begin,
                initial_reads=int(initial is not None),requests=self.prior_requests,
                request_input_tokens=sum(r['input_tokens'] for r in self.prior_requests)+(initial['input_tokens'] if initial else 0),
                request_cached_tokens=sum(r['cached_tokens'] for r in self.prior_requests)+(initial['cached_tokens'] if initial else 0),
                peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=self.torch.cuda.max_memory_reserved())
            return row
        finally:
            if self.engine.has_unfinished_requests(): self.engine.abort_request(self.trial_keys)
            self.registry.clear(); self.captured.clear(); self.hidden.clear()
            self.capture_initial=False; self.initial_hidden=None; self.keep_natural=False

    def consistency(self, sample, policy):
        """Pause on causal sentence events; branch, reject/accept, resume exact IDs.

        Chunk re-prefill and all trial generations are charged. No offline future
        trajectory is consulted. Separate serial execution from our asynchronous
        scheduler in reporting; this is an online adaptation of offline AC code.
        """
        prompt,count,cap=sample['prompt_ids'],sample['candidate_count'],policy['cap']
        agreement=Agreement(policy.get('agreement_count',10))
        prefix=[];checks=[];detector=base.StreamDetector();lengths=[0];counts=[0]
        previous='';ordinal=last_read=0;done=stopped=False
        while len(prefix)<cap and not done:
            # Stream until the next eligible complete sentence, then pause.
            self.serial+=1;key='ac-reason-'+str(self.serial);self.trial_keys.append(key)
            params=self.SP(temperature=0.,top_p=1.,top_k=-1,max_tokens=cap-len(prefix),
                seed=20261009,detokenize=False,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.)
            ids=prompt+prefix;self.engine.add_request(key,dict(prompt_token_ids=ids),params)
            base_prefix=list(prefix);selected=None;chunk=None;ts=time.perf_counter()
            while self.engine.has_unfinished_requests():
                if time.perf_counter()-ts>300:raise TimeoutError('AC generation exceeded limit')
                for out in self.engine.step():
                    if out.request_id != key: continue
                    chunk=out;new=base_prefix+list(out.outputs[0].token_ids)
                    for k in range(len(prefix)+1,len(new)+1):
                        text=self.tok.decode(new[:k],skip_special_tokens=False,clean_up_tokenization_spaces=False)
                        if text.endswith('\ufffd'):continue
                        assert text.startswith(previous)
                        lengths.append(len(text));counts.append(k)
                        for end in detector.update(text):
                            at=bisect.bisect_left(lengths,end)
                            if text[end:lengths[at]].strip():continue
                            ordinal+=1
                            if ordinal-last_read>=policy.get('sentence_gap',1):selected=counts[at]
                        previous=text
                    prefix=new;done=out.finished
                if selected is not None or done:
                    if not done:self.engine.abort_request([key])
                    break
            assert chunk is not None
            self.prior_requests.append(dict(token_ids=list(chunk.outputs[0].token_ids),
                text=None,finish_reason=chunk.outputs[0].finish_reason,kind='ac_reason',
                input_tokens=len(ids),cached_tokens=chunk.num_cached_tokens or 0,seconds=time.perf_counter()-ts))
            if selected is not None and not done:
                answer=self.final_answer(prompt,prefix[:selected],policy)
                prediction=parse_answer(answer['text'],count)
                stopped=agreement.update(prediction)
                checks.append(dict(prefix_tokens=selected,observable_tokens=len(prefix),ordinal=ordinal,
                    prediction=prediction,agreement_run=agreement.run,accepted=stopped))
                last_read=ordinal
                if stopped:break
        if not stopped:
            answer=self.final_answer(prompt,prefix,policy)
            prediction=parse_answer(answer['text'],count)
        return dict(prediction=prediction,early_stop=stopped,generation_tokens=len(prefix),
            generated_ids=prefix,checks=checks,trial_reads=len(checks),
            full_reads=len(checks)+int(not stopped),sentence_opportunities=ordinal,
            read_prefix_tokens=checks[-1]['prefix_tokens'] if stopped else len(prefix),
            extra_tokens_since_read_prefix=len(prefix)-checks[-1]['prefix_tokens'] if stopped else 0)

    def deer(self,sample,policy):
        prompt,count,cap=sample['prompt_ids'],sample['candidate_count'],policy['cap']
        marker=policy.get('transition','Wait');history='';checks=[];stopped=False
        natural_output=0
        # Match Qwen3 release's string-based continuation/rollback. Physical
        # outputs and re-encoded continuation length are reported separately.
        while True:
            encoded=self.tok.encode(history,add_special_tokens=False)
            remaining=cap-len(encoded)
            if remaining<=0:break
            enable=len(checks)<policy.get('max_trials',10)
            chunk=self.request(prompt+encoded,remaining,
                stop=[marker,'</think>'] if enable else ['</think>'])
            natural_output+=len(chunk['token_ids']);history+=chunk['text']
            ended=chunk['stop_reason']=='</think>' or chunk['finish_reason']=='length'
            if ended or chunk['stop_reason']!=marker:break
            encoded=self.tok.encode(history,add_special_tokens=False)
            cue=self.tok.encode(self.cue(encoded,trial=True),add_special_tokens=False)
            trial=self.request(prompt+encoded+cue,policy.get('trial_cap',20),
                               stop=['</think>'],logprobs=True,include_stop=True)
            end_id=self.tok.convert_tokens_to_ids('</think>')
            confidence=deer_confidence(trial['logprobs'],bool(trial['token_ids']) and trial['token_ids'][-1]==end_id)
            stopped=confidence>policy.get('trial_threshold',.95)
            checks.append(dict(prefix_tokens=len(encoded),confidence=confidence,accepted=stopped,
                               trial_tokens=len(trial['token_ids'])))
            if stopped:break
            # Rejected trial tokens are discarded; restore the natural marker.
            history+=marker
        encoded=self.tok.encode(history,add_special_tokens=False)
        answer=self.final_answer(prompt,encoded,policy)
        return dict(prediction=parse_answer(answer['text'],count),early_stop=stopped,
            generation_tokens=natural_output,generated_ids=encoded,continuation_tokens=len(encoded),
            checks=checks,trial_reads=len(checks),full_reads=len(checks)+1,
            adaptation='Released Qwen3 mechanism; fixed-choice cue; greedy trials; shared reasoning cap plus counted answer/trial budgets')
