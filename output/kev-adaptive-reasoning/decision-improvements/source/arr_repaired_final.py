"""Explicit assistant answer-prefix completion for generated-answer baselines.

No learned answer head, allowed-token mask, gold access, or answer repair.
Trial confidence and AC/DEER continuation rules remain in the frozen parent.
"""
from arr_native_final import Engine as NativeEngine


class Engine(NativeEngine):
    def final_answer(self, prompt, generated, policy):
        end = self.tok.convert_tokens_to_ids('</think>')
        cue = ('' if end in generated else '\n</think>') + '\n\n<FINAL>'
        ids = prompt + generated + self.tok.encode(cue, add_special_tokens=False)
        result = self.request(ids, policy.get('answer_cap', 64), stop=['</FINAL>'], include_stop=True)
        # Include the actual assistant prefix for parsing; don't charge injected
        # prompt tokens as generated tokens, or invent a missing closing tag.
        result['raw_text'] = result['text']
        result['text'] = '<FINAL>' + result['raw_text']
        result['answer_cue'] = cue
        result['answer_interface'] = 'assistant_FINAL_prefix_completion'
        return result
