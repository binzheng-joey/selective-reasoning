"""Reserve fresh material groups from metadata only; never open raw questions/gold."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT.parent
PROJECT=ROOT.parents[2]
OUT=ROOT/'arr-sprint/data-audit'
OFFICIAL=OUTPUT/'official-gate-data-v1'
PREVIOUS=OUTPUT/'prefix-readout/fresh-stopping'
HASHES=['material_sha256','normalized_material_sha256','sentence_set_sha256']
FORBIDDEN={'gold','answer','label','state','material','question','options','prediction','correct',
           'generated_ids','prompt_ids','prefix_ids','probabilities','hidden','token_ids'}


def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,value):
    p=OUT/name;tmp=p.with_suffix(p.suffix+'.pending')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'confirmation-manifest.jsonl').exists(),'Preserve reserved sample selection'
    sources={}
    def load(path):
        path=Path(path);data=path.read_bytes();sources[str(path)]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
        return [json.loads(line) for line in data.splitlines() if line.strip()] if path.suffix=='.jsonl' else json.loads(data)
    previous=load(PREVIOUS/'exclusions.json')
    ids=set(previous['sample_ids']);components=set(previous['canonical_components'])
    groups={tuple(x) for x in previous['source_groups']}
    hashes={k:set(previous['material_hashes'][k]) for k in HASHES}
    def absorb(x,found):
        if isinstance(x,str):
            if re.fullmatch(r'(?:pw|lq):[^\s]+',x):ids.add(x);found.add(x)
        elif isinstance(x,list):
            for v in x:absorb(v,found)
        elif isinstance(x,dict):
            sid=x.get('sample_id',x.get('id'))
            if isinstance(sid,str) and re.fullmatch(r'(?:pw|lq):[^\s]+',sid):ids.add(sid);found.add(sid)
            if isinstance(x.get('component_id'),str):components.add(x['component_id'])
            if x.get('dataset') and x.get('source_group_id') is not None:groups.add((x['dataset'],str(x['source_group_id'])))
            for k in HASHES:
                if isinstance(x.get(k),str):hashes[k].add(x[k])
            for k,v in x.items():
                if k not in FORBIDDEN:absorb(v,found)
    meta=[]
    for role in ['train','calibration','unused_dev_reserve']:
        part=load(OFFICIAL/'manifests'/(role+'.jsonl'))
        assert all(not FORBIDDEN.intersection(x) for x in part),'Non-metadata source'
        meta.extend(part)
    by_component=defaultdict(list)
    for row in meta:by_component[row['component_id']].append(row)
    names=subprocess.check_output(['rg','--files',str(OUTPUT),'-g','*manifest*.json*',
        '-g','matched-index.jsonl','-g','*consumption.json','-g','fit-and-leakage-check.json'],text=True).splitlines()
    selected=[];excluded=[]
    for name in names:
        p=Path(name)
        if any(root==p or root in p.parents for root in [OUT,ROOT/'arr-sprint',OFFICIAL,PREVIOUS]):
            excluded.append(dict(path=name,reason='source universe or current reservation'));continue
        if any(x in p.parts for x in ['cache','input-snapshots','reconstructed-now','source','tmp']):continue
        if any(x in name.lower() for x in ['sealed','official_test','officialtest']):continue
        if any(x in p.name for x in ['sha256','launch-manifest','delivery-manifest','startup-stage']):continue
        selected.append(p)
    explicit=[PREVIOUS/'calibration-manifest.jsonl',PREVIOUS/'confirmation-manifest.jsonl',
        OUTPUT/'stream-readout/samples.jsonl',ROOT/'gate-unified-endpoint/split.json']
    history=[]
    for p in sorted(set(selected+explicit)):
        found=set();absorb(load(p),found);history.append(dict(path=str(p),sample_ids=len(found)))
    before=dict(sample_ids=len(ids),components=len(components),groups=len(groups))
    rounds=0
    while True:
        sizes=(len(ids),len(components),len(groups),*(len(hashes[k]) for k in HASHES))
        for x in meta:
            if x['sample_id'] in ids or (x['dataset'],str(x['source_group_id'])) in groups or any(x[k] in hashes[k] for k in HASHES):
                components.add(x['component_id'])
        for cid in list(components):
            for x in by_component.get(cid,[]):
                ids.add(x['sample_id']);groups.add((x['dataset'],str(x['source_group_id'])))
                for k in HASHES:hashes[k].add(x[k])
        rounds+=1
        after=(len(ids),len(components),len(groups),*(len(hashes[k]) for k in HASHES))
        if after==sizes:break
        assert rounds<20
    buckets=defaultdict(list)
    rank=lambda s:hashlib.sha256(('arr-october-confirmation|20261010|'+s).encode()).hexdigest()
    def stratum(x):return x['source_group_id'].split('-OWA-')[0] if x['dataset']=='proofwriter' else 'LQ'
    for cid,members in by_component.items():
        if cid in components:continue
        train=[x for x in members if x['role']=='train' and x['official_split']=='train']
        if not train:continue
        kinds={stratum(x) for x in train}
        if len(kinds)!=1:continue
        representative=min(train,key=lambda x:rank('question/'+x['sample_id']))
        buckets[next(iter(kinds))].append((rank('family/'+cid),representative))
    targets=dict(AttNeg=64,AttNoneg=64,RelNeg=64,RelNoneg=64,LQ=256)
    pools={k:len(buckets[k]) for k in targets}
    chosen=[]
    for kind,n in targets.items():
        assert len(buckets[kind])>=n,('Insufficient untouched groups',kind,len(buckets[kind]),n)
        for order,(_,x) in enumerate(sorted(buckets[kind],key=lambda a:a[0])[:n]):
            row={k:v for k,v in x.items() if k not in FORBIDDEN}
            row.update(internal_split='arr_confirmation',selection_seed=20261010,stratum=kind,
                selection_rank=rank('family/'+x['component_id']),stratum_order=order)
            chosen.append(row)
    chosen.sort(key=lambda x:rank('execution/'+x['sample_id']))
    assert len(chosen)==len({r['component_id'] for r in chosen})==512
    assert not ({r['sample_id'] for r in chosen}&ids)
    assert not ({r['component_id'] for r in chosen}&components)
    for r in chosen:
        assert (r['dataset'],str(r['source_group_id'])) not in groups
        assert all(r[k] not in hashes[k] for k in HASHES)
    save('protocol.json',dict(UTC=now(),selection='Metadata-only fixed SHA256 ranking, one question per material component',
        targets=targets,questions=512,source_scope='Unused groups from official train, internal confirmation',
        exclusions='All prior assignments, including the entire older1024 confirmation pool, are excluded conservatively',
        gold_access=False,question_projection=False,source_sha256=sha(__file__),
        release='Project questions and run frozen policies only after final method and evaluation protocol are fixed',
        statistical_scope='512 is a compute-budget choice; no accuracy-loss tolerance or power guarantee imposed'))
    with (OUT/'confirmation-manifest.jsonl').open('x') as f:
        for row in chosen:f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n')
    save('audit.json',dict(UTC=now(),state='metadata_audit_complete',sources=sources,history_files=history,
        base_counts=before,closed_counts=dict(sample_ids=len(ids),components=len(components),groups=len(groups)),
        closure_rounds=rounds,eligible_groups_before=pools,selected=dict(Counter(r['dataset'] for r in chosen)),
        selected_strata=dict(Counter(r['stratum'] for r in chosen)),
        overlap=dict(sample_ids=0,components=0,source_groups=0,material_hashes=0),
        old_remaining256='Not reused; all1024 prior confirmation groups excluded',
        confirmation_manifest_sha256=sha(OUT/'confirmation-manifest.jsonl'),
        raw_sources_opened=0,new_gold_decodes=0,new_question_projections=0,
        limits=['External unrecorded use is unknown','Model pretraining exposure is unknown',
                'Inherited canonical and exact/normalized/sentence-set closure; no exhaustive semantic paraphrase audit']))
    (OUT/'report.md').write_text('# 新确认集材料隔离与预留\n\n'
        '已仅根据元数据预留512道新材料组题目：ProofWriter256、LogiQA2256；ProofWriter四类各64。\n\n'
        '排除了旧历史使用/分配、当前训练与开发、旧校准，以及整个旧1024确认池（包括尚未评分的256题）。'
        '在当前元数据审计范围内，题目ID、材料组、源材料和三种材料哈希均无交集。\n\n'
        '尚未读取新题正文、gold或生成任何新预测。模型/阈值冻结后再投影题目与运行，不能利用这批题调整方法。'
        '这些是官方train中的项目内确认材料，不称官方test或预训练未见数据。512题是算力预算选择，不是统计充分性保证。\n')
    print(json.dumps(dict(state='metadata_audit_complete',reserved=512,strata=targets,
        eligible_groups=pools,history_sources=len(history),new_gold_decodes=0)))


if __name__=='__main__':main()
