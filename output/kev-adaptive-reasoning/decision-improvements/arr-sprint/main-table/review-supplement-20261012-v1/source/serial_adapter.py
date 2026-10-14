"""Matched serial execution using abort/exact-token-prefix resumption.

Frozen asynchronous scheduling is inherited. Serial pauses are realized by
abort/resubmit, and their repeated-prefill work is charged. This is an execution
implementation comparison, not an ideal KV-preserving pause experiment.
"""
import inspect
import textwrap
import time

from arr_integrated_online import Engine as Original
from arr_variable_choice import Engine as VariableEngine


def build_serial_reason():
    source = textwrap.dedent(inspect.getsource(Original.reason))
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, ('Frozen control flow changed', old)
        source = source.replace(old, new)
    replace("genkey, genrid = self.add(prompt, cap, 'generate', count)",
            "genkey, genrid = self.add(prompt, cap, 'generate', count)\n"
            "    gen_base = []; serial_paused = False; resumed = 0\n"
            "    generator_inputs = len(prompt); generator_cached_total = 0\n"
            "    self.registry[genrid]['prefix_offset'] = 0")
    replace("new = list(o.outputs[0].token_ids)", "new = gen_base + list(o.outputs[0].token_ids)")
    replace("if not generation_finished: self.engine.abort_request([genkey])",
            "if not generation_finished and not serial_paused: self.engine.abort_request([genkey])")
    replace("            if read_at is not None: break", """            if inflight is None and serial_paused and read_at is None:
                gen_base = list(prefix)
                genkey, genrid = self.add(prompt + gen_base, cap - len(prefix), 'generate', count)
                self.registry[genrid]['prefix_offset'] = len(prefix)
                generator_inputs += len(prompt) + len(prefix)
                resumed += 1; serial_paused = False; gen_cached = 0
            if read_at is not None: break""")
    replace("                if perform:\n                    ids = self.read_ids", """                if perform:
                    ts = time.perf_counter()
                    self.engine.abort_request([genkey])
                    self.registry.pop(genrid, None)
                    torch.cuda.synchronize(); abort_seconds += time.perf_counter()-ts
                    generator_cached_total += gen_cached
                    gen_cached = 0; serial_paused = True
                    ids = self.read_ids""")
    replace("generator_prompt_tokens=len(prompt), generator_cached_tokens=gen_cached,",
            "generator_prompt_tokens=generator_inputs, generator_cached_tokens=generator_cached_total+gen_cached,\n"
            "        serial_resume_requests=resumed, execution='serial_abort_resume',")
    namespace = dict(Original.reason.__globals__)
    exec(compile(source, __file__ + ':frozen-reason-serial-adapter', 'exec'), namespace)
    return namespace['reason']


SERIAL_REASON = build_serial_reason()


class Engine(VariableEngine):
    _serial_mode = False

    def reason(self, sample, policy):
        if policy.get('execution') != 'serial_abort_resume':
            return super().reason(sample, policy)
        self._serial_mode = True
        try:
            return SERIAL_REASON(self, sample, policy)
        finally:
            self._serial_mode = False

    def capture(self, h):
        if not self._serial_mode:
            return super().capture(h)
        keep = self.keep_natural
        self.keep_natural = False
        try:
            logits = super().capture(h)
        finally:
            self.keep_natural = keep
        if keep:
            begin = time.perf_counter()
            batch = self.runner.input_batch
            qloc = self.runner.query_start_loc.cpu.numpy()
            for i, rid in enumerate(batch.req_ids):
                info = self.registry.get(rid)
                if not info or info['kind'] != 'generate':
                    continue
                pos = int(batch.num_computed_tokens_cpu[i]) + int(qloc[i+1]-qloc[i]) - 1
                k = pos - info['length'] + 1 + info.get('prefix_offset', 0)
                if k >= 0:
                    self.hidden[k] = h[i].detach().clone()
                    for old in list(self.hidden):
                        if old < k-64:
                            del self.hidden[old]
            self.hook_host_seconds += time.perf_counter()-begin
        return logits
