"""Freeze existing train1024/dev256 and model before feature extraction."""
import collections
import shutil
from common import ROOT, PROJECT, PR, OLD, MODEL, GRID, GPUS, read, rows, sha, now, write, write_rows, relative


def main():
    assert not (ROOT/'protocol.json').exists(), 'Frozen protocol exists'
    base=read(PR/'readout-protocol.json')
    feature_index={r['sample_id']:r for r in rows(PR/'answer-head-training/features/index.jsonl')}
    samples=[];bindings={};inputs=[]
    for n,b in enumerate(base['samples']):
        assert sha(b['source_path'])==b['source_sha256']
        pair=read(b['source_path'])
        assert pair['complete'] and not pair['failure'] and not pair['R']['failure']
        f=PR/'answer-head-training/features'/feature_index[b['sample_id']]['feature_file']
        assert sha(f)==feature_index[b['sample_id']]['feature_sha256']
        generated=pair['R']['generated_ids'][:1024]
        s=dict(index=n,sample_id=b['sample_id'],dataset=b['dataset'],split=b['split'],
            component_id=b['component_id'],candidate_count=b['candidate_count'],
            source=relative(b['source_path']),source_sha256=b['source_sha256'],
            answer_feature=relative(f),answer_feature_sha256=sha(f),
            prompt_tokens=len(pair['R']['input_ids']),generated_tokens=len(generated))
        samples.append(s)
        inputs.append(dict(index=n,sample_id=b['sample_id'],prompt_ids=pair['R']['input_ids'],generated_ids=generated))
    assert len(samples)==1280 and len({s['sample_id'] for s in samples})==1280
    groups={split:{s['component_id'] for s in samples if s['split']==split} for split in ['train','cal']}
    assert groups['train'].isdisjoint(groups['cal'])
    assert collections.Counter(s['split'] for s in samples)=={'train':1024,'cal':256}
    labels=[]
    for split in ['train','cal']:
        source=OLD/f'{split}-gold-alignment.jsonl'
        gold={r['sample_id']:r['gold'] for r in rows(source)}
        bindings[relative(source)]=sha(source)
        labels.extend(dict(sample_id=s['sample_id'],split=split,gold=gold[s['sample_id']]) for s in samples if s['split']==split)
    write_rows(ROOT/'inputs-no-gold.jsonl',inputs)
    write_rows(ROOT/'labels-observed.jsonl',labels)
    write_rows(ROOT/'samples.jsonl',samples)
    for f in [ROOT/'samples.jsonl',ROOT/'inputs-no-gold.jsonl',ROOT/'labels-observed.jsonl',PR/'readout-protocol.json']:
        bindings[relative(f)]=sha(f)
    model_files={relative(f):sha(f) for f in MODEL.iterdir() if f.is_file() and (f.suffix in ['.json','.safetensors'] or f.name=='merges.txt')}
    p=dict(UTC=now(),scope='Existing train1024 and observed dev256; no independent-test claim',
        model=relative(MODEL),revision=base['model_revision'],model_files=model_files,
        questions=dict(train=1024,observed_dev=256),allowed_GPUs=GPUS,bindings=bindings,
        features=dict(layer='Final normalized language-model hidden states',precision='BF16 frozen native HF SDPA',
            extraction='Teacher-forced replay of existing prompt and first1024 generation tokens, no new sampling',
            sequence='Prompt plus actual generated prefix; no injected answer cue for stream heads',
            causal_scope='Future states masked from all readouts; full-sequence and truncated prefix numerical check recorded',
            no_online_equivalence_claim=True),
        heads=['answer_position_linear','stream_last_linear','stream_mean_linear','stream_attention'],
        main_checkpoints=GRID,
        training=dict(seeds=[17,29,43],learning_rates=[.0003,.001],epochs=20,eval_epochs=[5,10,20],
            batch_questions=16,optimizer='AdamW',weight_decay=.01,gradient_clip=1.,
            loss='Hard-gold CE; each question equally weighted across five actual checkpoints',
            selection='Observed-dev mean NLL across all five budgets; no gold-based early-position selection',
            head_precision='FP32 parameters and logits; all four heads use the same rule',
            generation_backbone_frozen=True,main_runs=24),
        limits=dict(feature_wall_seconds=7200,training_wall_seconds=7200,resource_wait_seconds=28800,
                    estimated_feature_bytes=sum((s['prompt_tokens']+s['generated_tokens'])*2560*2 for s in samples)),
        sentence_evaluation=dict(seed=29,split='observed_dev256',
            readers=['stream_last_linear','stream_mean_linear','stream_attention'],
            selection='All causal complete-sentence boundaries aligned to existing token ends; token-internal boundaries recorded and skipped',
            terminal='All actual terminal tokens',threshold_selection=False),
        next_stage='After matched head training and sentence diagnostics: freeze reader before separate stopping/scheduling protocol.',
        protected='No new gate512 or confirmation1024 gold/input use; no mutation to other experiment directories')
    assert shutil.disk_usage(ROOT).free>p['limits']['estimated_feature_bytes']*2+2**30
    write(ROOT/'protocol.json',p)
    write(ROOT/'status.json',dict(state='prepared',UTC=now(),phase='readout-comparison',questions=1280))
    print(dict(prepared=True,questions=1280,estimated_feature_GiB=p['limits']['estimated_feature_bytes']/2**30))


if __name__=='__main__':main()
