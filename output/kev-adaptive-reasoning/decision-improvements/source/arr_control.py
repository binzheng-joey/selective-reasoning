"""CPU-only policy arithmetic shared by offline checks and the online worker.

No label lookup, training, or torch import. Invalid trial answers reset agreement.
"""
import math
import re


def distribution(p):
    if len(p) < 2 or any(not math.isfinite(v) or v < 0 for v in p):
        raise ValueError('Invalid candidate probabilities')
    if abs(sum(p) - 1) > 1e-4:
        raise ValueError('Candidate probabilities must sum to one')
    return list(p)


def features(p, prompt_length):
    p = distribution(p)
    z = sorted(p)
    return [z[-1], z[-1]-z[-2],
            -math.fsum(v*math.log(v) for v in p if v > 0)/math.log(len(p)),
            math.log1p(prompt_length)]


def sigmoid(x):
    return 1/(1+math.exp(-x)) if x >= 0 else math.exp(x)/(1+math.exp(x))


def logistic(model, x):
    if 'constant' in model:
        return model['constant']
    if len(x) != len(model['weight']):
        raise ValueError('Gate feature dimension mismatch')
    return sigmoid(math.fsum(a*b for a, b in zip(x, model['weight']))+model['bias'])


def route(bundle, name, fraction, p, prompt_length, hidden=None):
    """Strict > matches the existing held-out threshold protocol, including ties."""
    f = features(p, prompt_length)
    if name in ('benefit_raw', 'initial_error'):
        if hidden is None:
            raise ValueError('This gate requires the initial answer-position state')
        x = f + list(hidden)
        score = (logistic(bundle['models'][0], x)-logistic(bundle['models'][1], x)
                 if name == 'benefit_raw' else logistic(bundle['models'][2], x))
    else:
        score = {'confidence': -f[0], 'margin': -f[1], 'entropy': f[2]}[name]
    key = name + '/' + str(float(fraction))
    threshold = bundle['thresholds'][key]
    if fraction in (0., 1.):
        decision = fraction == 1.
    else:
        decision = score > threshold
    return dict(reason=bool(decision), score=score, threshold=threshold,
                score_minus_threshold=None if threshold is None else score-threshold)


def head_stop(p, rule='entropy', threshold=.85):
    p = distribution(p)
    if rule == 'never':
        return False
    score = 1-features(p, 0)[2] if rule == 'entropy' else max(p)
    if rule not in ('entropy', 'confidence'):
        raise ValueError(rule)
    return score >= threshold


class Agreement:
    def __init__(self, count=10):
        if count < 1:
            raise ValueError('Agreement count must be positive')
        self.count, self.last, self.run = count, None, 0

    def update(self, answer):
        if answer is None:
            self.last, self.run = None, 0
            return False
        self.run = self.run+1 if self.last == answer else 1
        self.last = answer
        return self.run >= self.count


def parse_answer(text, count):
    """Strict final boxed/FINAL index, never a number guessed from reasoning prose.

    Existing task prompts use one-based numeric indices. Ambiguous or
    out-of-range outputs remain failures and are included in the denominator.
    """
    boxed = re.findall(r'\\boxed\s*\{\s*([1-4])\s*\}', text)
    final = re.findall(r'<FINAL>\s*([1-4])\s*</FINAL>', text)
    candidates = boxed + final
    if not candidates:
        m = re.fullmatch(r'\s*([1-4])[.\s]*', text)
        candidates = [m.group(1)] if m else []
    if not candidates or len(set(candidates)) != 1:
        return None
    index = int(candidates[0])-1
    return index if index < count else None


def deer_confidence(logprobs, ended_think):
    """Qwen3 author's avg2: omit first token; require final </think>.

    Use the sampled token's logprob explicitly, not dict insertion order.
    Include the last token as the actual source loop does. Guard the upstream
    zero-denominator edge case. This is trial-generation confidence, not head p.
    """
    if not ended_think or len(logprobs) < 2:
        return 0.
    values = logprobs[1:]
    if any(not math.isfinite(v) or v > 1e-6 for v in values):
        raise ValueError('Invalid trial token log probability')
    return math.exp(math.fsum(max(v, math.log(1e-10)) for v in values)/len(values))
