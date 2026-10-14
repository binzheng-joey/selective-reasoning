"""Ordinary generated-final baseline, separate from forced terminal answering.

Reuses the development engine without changing its frozen files. No extra
answer cue or decision head is used. Missing final answers remain invalid and
must be counted as errors. A forced-terminal reference is reported separately.
This adapter is CPU-tested, not yet validated on the live GPU backend.
"""
import re
import time

from arr_prior_baselines import Engine as PriorEngine


def parse_candidate_index(text, count):
    """Support official five-option items without changing the frozen pilot parser."""
    candidates = re.findall(r'<FINAL>\s*([1-9][0-9]*)\s*</FINAL>', text)
    candidates += re.findall(r'\\boxed\s*\{\s*([1-9][0-9]*)\s*\}', text)
    if not candidates:
        match = re.fullmatch(r'\s*([1-9][0-9]*)[.\s]*', text)
        candidates = [match.group(1)] if match else []
    if len(set(candidates)) != 1:
        return None
    index = int(candidates[0]) - 1
    return index if 0 <= index < count else None


def generated_final(tokenizer, token_ids, candidate_count):
    """Current prompts start in thinking mode; only parse after its last close."""
    end = tokenizer.convert_tokens_to_ids('</think>')
    positions = [i for i, token in enumerate(token_ids) if token == end]
    if not positions:
        return None, 'thinking_not_closed'
    text = tokenizer.decode(token_ids[positions[-1] + 1:], skip_special_tokens=False,
                            clean_up_tokenization_spaces=False).rstrip()
    # Remove a trailing chat EOS without treating reasoning numbers as answers.
    for suffix in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(suffix):
            text = text[:-len(suffix)].rstrip()
    answer = parse_candidate_index(text, candidate_count)
    return answer, 'valid' if answer is not None else 'missing_or_invalid_final'


class Engine(PriorEngine):
    def native_final(self, sample, policy):
        if policy['kind'] != 'native_final' or policy['mode'] != 'always':
            raise ValueError('Ordinary baseline always generates; no head or routing')
        self.reset_trial()
        self.prior_requests = []
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        try:
            result = self.request(sample['prompt_ids'], policy['cap'])
            prediction, status = generated_final(self.tok, result['token_ids'], sample['candidate_count'])
            self.torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            return dict(sample_id=sample['sample_id'], dataset=sample['dataset'],
                policy=policy['id'], policy_spec=policy, prediction=prediction,
                final_status=status, finish_reason=result['finish_reason'],
                budget_truncated=result['finish_reason'] == 'length',
                reasoning_invoked=True, early_stop=False, initial_reads=0,
                full_reads=0, trial_reads=0, trigger_calls=0, initial=None, routing=None,
                generation_tokens=len(result['token_ids']), physical_output_tokens=len(result['token_ids']),
                generated_ids=result['token_ids'], requests=self.prior_requests,
                request_input_tokens=result['input_tokens'], request_cached_tokens=result['cached_tokens'],
                pipeline_seconds=seconds, peak_allocated_bytes=self.torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=self.torch.cuda.max_memory_reserved(),
                answer_mechanism='Natural generated final; no forced cue or answer head')
        finally:
            if self.engine.has_unfinished_requests():
                self.engine.abort_request(self.trial_keys)
            self.registry.clear()
            self.captured.clear()
            self.hidden.clear()
            self.capture_initial = False
            self.initial_hidden = None
            self.keep_natural = False
