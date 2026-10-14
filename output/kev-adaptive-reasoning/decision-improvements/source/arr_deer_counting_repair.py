"""Validation-only DEER repair; frozen inference and saved outputs stay intact.

DEER budgets each request from its re-encoded text continuation. That length is
not the count of emitted tokens. Replay the original text/rollback mechanism to
verify request budgets, then independently reconcile all physical token costs.
"""
import math

from arr_control import deer_confidence


def require(ok, message):
    if not ok:
        raise ValueError('DEER accounting: ' + message)


def validate_deer(row, sample, spec, tokenizer):
    require(spec['kind'] == 'deer' and spec['mode'] == 'always', 'unsupported policy')
    require(row['sample_id'] == sample['sample_id'] and row['policy'] == spec['id'], 'identity')
    require(row['policy_spec'] == spec, 'policy changed')
    require(math.isfinite(row['pipeline_seconds']) and row['pipeline_seconds'] > 0, 'timing')
    require(row['prediction'] is None or type(row['prediction']) is int and
            0 <= row['prediction'] < sample['candidate_count'], 'answer index')
    require(row['initial'] is None and row['initial_reads'] == 0 and
            row['reasoning_invoked'] and row['trigger_calls'] == 0, 'unexpected routing')
    requests = row['requests']
    require(len(requests) >= 2, 'missing natural/final requests')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    end_id = tokenizer.convert_tokens_to_ids('</think>')
    prompt_length = len(sample['prompt_ids'])
    marker = spec.get('transition', 'Wait')
    history = ''; index = 0; natural = 0; checks = []; stopped = False

    def request(limit, input_length, trial=False):
        nonlocal index
        require(index < len(requests), 'missing request')
        result = requests[index]; index += 1
        emitted = len(result['token_ids'])
        require(limit > 0 and emitted <= limit, 'request exceeds generation budget')
        require(result['input_tokens'] == input_length, 'request prefix length')
        require(0 <= result['cached_tokens'] <= input_length, 'cache accounting')
        require(math.isfinite(result['seconds']) and result['seconds'] >= 0, 'request timing')
        require(result['finish_reason'] in ('stop', 'length'), 'unfinished request')
        if result['finish_reason'] == 'length':
            require(emitted == limit, 'length-finished request did not exhaust budget')
        require(len(result['logprobs']) == (emitted if trial else 0), 'logprob accounting')
        return result

    while True:
        encoded = encode(history)
        remaining = spec['cap'] - len(encoded)
        if remaining <= 0:
            break
        chunk = request(remaining, prompt_length + len(encoded))
        natural += len(chunk['token_ids']); history += chunk['text']
        ended = chunk['stop_reason'] == '</think>' or chunk['finish_reason'] == 'length'
        if ended or chunk['stop_reason'] != marker:
            break
        require(len(checks) < spec.get('max_trials', 10), 'extra transition trial')
        encoded = encode(history)
        cue = encode('\n**Final Answer**\nThe final candidate index is \\boxed')
        trial = request(spec.get('trial_cap', 20), prompt_length + len(encoded) + len(cue), True)
        confidence = deer_confidence(trial['logprobs'],
                                     bool(trial['token_ids']) and trial['token_ids'][-1] == end_id)
        stopped = confidence > spec.get('trial_threshold', .95)
        check = dict(prefix_tokens=len(encoded), confidence=confidence,
                     accepted=stopped, trial_tokens=len(trial['token_ids']))
        checks.append(check)
        if stopped:
            break
        history += marker

    encoded = encode(history)
    require(index == len(requests) - 1, 'unexpected/missing final request')
    cue = ('' if end_id in encoded else '\n</think>') + '\n\n<FINAL>'
    answer = request(spec.get('answer_cap', 64), prompt_length + len(encoded) + len(encode(cue)))
    require(answer['answer_cue'] == cue and
            answer['answer_interface'] == 'assistant_FINAL_prefix_completion', 'final interface')
    require(row['generated_ids'] == encoded and row['continuation_tokens'] == len(encoded),
            're-encoded continuation mismatch')
    require(row['generation_tokens'] == natural, 'natural token total')
    require(row['checks'] == checks and row['trial_reads'] == len(checks) and
            row['full_reads'] == len(checks) + 1 and row['early_stop'] == stopped, 'trial decisions')
    require(row['physical_output_tokens'] == sum(len(r['token_ids']) for r in requests), 'physical total')
    require(row['request_input_tokens'] == sum(r['input_tokens'] for r in requests), 'input total')
    require(row['request_cached_tokens'] == sum(r['cached_tokens'] for r in requests), 'cached total')
    return dict(natural_tokens=natural, continuation_tokens=len(encoded), cap=spec['cap'],
                natural_minus_cap=natural-spec['cap'], continuation_minus_cap=len(encoded)-spec['cap'])
