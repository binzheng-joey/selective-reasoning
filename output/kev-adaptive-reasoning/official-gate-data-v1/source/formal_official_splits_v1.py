"""CPU-only official splits, reuse gold-skipping parser, exact duplicate components."""
import argparse
import collections
import datetime
import heapq
import importlib.util
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True, type=Path)
ROOT = p.parse_args().root
OLD = ROOT.parent / 'gate-data-partitions-v1'
spec = importlib.util.spec_from_file_location('gold_free_helpers', OLD / 'source/gate_data_v1.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
SEED = 20260930


def rank(s):
    return g.hh('official-gate-data-v1|' + str(SEED) + '|' + s)


def collect():
    nodes, inventory = [], []
    old_roles = {n: c['role'] for c in g.read_rows(OLD / 'components.jsonl') for n in c['raw_nodes']}
    for src in json.loads((ROOT / 'download-manifest.json').read_text()):
        ds, split = src['dataset'], src['official_split']
        assert g.digest(src['path']) == src['raw_sha256']
        count = questions = 0
        with Path(src['path']).open() as f:
            for line_number, text in enumerate(f, 1):
                if not text.strip():
                    continue
                d, end = g.project(text)
                assert not text[end:].strip()
                sid = str(d['id'])
                material = d['theory'] if ds == 'proofwriter' else d['text']
                qs = d['questions'] if ds == 'proofwriter' else {'': d}
                items = []
                for qid, q in sorted(qs.items()):
                    sample = 'pw:' + sid + ':' + qid if ds == 'proofwriter' else 'lq:' + sid
                    options = ['True', 'False', 'Unknown'] if ds == 'proofwriter' else q['options']
                    items.append({'sample_id': sample, 'source_path': src['path'], 'source_line': line_number,
                                  'source_qid': qid or None, 'question_sha256': g.hh(q['question']),
                                  'options_sha256': g.hh(g.encode(options))})
                key = ds + ':' + split + ':' + str(line_number)
                node = {'key': key, 'dataset': ds, 'official_split': split, 'source_id': sid,
                        'material': material, 'items': items, 'observed': False,
                        'prior_dev_role': old_roles[ds + ':raw:' + str(line_number)] if split == 'dev' else None,
                        'payload': g.hh(g.encode([material, [(r['sample_id'], r['question_sha256'], r['options_sha256']) for r in items]]))}
                nodes.append(node)
                count += 1
                questions += len(items)
        inventory.append({'dataset': ds, 'official_split': split, 'raw_rows': count, 'questions': questions,
                          'source_sha256': src['raw_sha256'], 'source_path': src['path']})
        print('SOURCE_PARSED_GOLD_SKIPPED', ds, split, count, questions, flush=True)
    observed = []
    for file in [g.E / 'data/inputs.jsonl', g.F / 'inputs.jsonl']:
        for d in g.read_rows(file):
            sample = d['id']
            ds = 'proofwriter' if sample.startswith('pw:') else 'logiqa2'
            nodes.append({'key': 'observed:' + sample, 'dataset': ds, 'official_split': 'dev',
                          'source_id': sample.split(':')[1], 'material': d['state'], 'items': [],
                          'observed': True, 'observed_sample_id': sample})
            observed.append(sample)
    assert len(observed) == len(set(observed)) == 224
    return nodes, inventory


def main():
    protected = json.loads((OLD / 'protected-prior-sha256.json').read_text())
    protected.update({str(f): g.digest(f) for f in OLD.rglob('*') if f.is_file()})
    g.write_new(ROOT / 'prior-preserved-sha256.json', protected)
    downloads = json.loads((ROOT / 'download-manifest.json').read_text())
    protocol = {'version': 'official-gate-data-v1', 'seed': SEED, 'script_sha256': g.digest(__file__),
                'official_sources': downloads, 'selection_fields': ['dataset', 'source_ID', 'group', 'fixed_SHA_rank'],
                'gold_values_decoded': False, 'observed_samples': 224,
                'duplicate_rules': ['same dataset/source ID globally (crosssplit ID conflicts quarantined)',
                                    'normalized material', 'sorted unique sentence set', '5-word shingle Jaccard>=0.90'],
                'policy': 'Observed connected components -> exploration. Any other component spanning official splits -> quarantine entirely. Conflicting repeated source IDs -> quarantine. Remaining official train only -> train, official test only -> sealed_official_test. Official dev retains prior calibration/closed holdout; prior candidate dev train becomes unused_dev_reserve.',
                'derived_variants': 'All prefixes/trajectories/option permutations inherit parent component and role',
                'representative': 'One hash-ranked question per component; no gold/effect stratification',
                'prefix_filter': 'Exact Jaccard prefix filtering; prefix size n-ceil(0.9*n)+1 under shared total shingle order; full intersection verifies candidates',
                'independent_validation': 'Second shared order (lexicographic instead of frequency), verifies every qualifying pair stays within one component/role',
                'limitations': ['Unseen low-overlap paraphrases/entity renames may evade detection', 'LogiQA no explicit passage ID', 'Pretraining/adapter contamination unknown', 'PW license remains unconfirmed; local research authorized by user, no distribution'],
                'roles': ['train', 'calibration', 'sealed_official_test', 'sealed_internal_dev_holdout', 'unused_dev_reserve', 'exploration', 'quarantine']}
    g.write_new(ROOT / 'official-split-protocol.json', protocol)
    nodes, inventory = collect()
    parents = list(range(len(nodes)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)
    indices = [collections.defaultdict(list) for _ in range(3)]
    edges = []
    for i, n in enumerate(nodes):
        n['normal'] = g.norm(n['material'])
        n['sentences'] = g.sentence_set(n['material'])
        assert n['normal']
        for k, key in enumerate([(n['dataset'], n['source_id']), n['normal'], n['sentences']]):
            prior = indices[k][key]
            if prior:
                union(i, prior[0])
                edges.append({'left': nodes[prior[0]]['key'], 'right': n['key'], 'reason': ['source_ID', 'normalized_material', 'sentence_set'][k]})
            prior.append(i)
    ambiguous = set()
    id_audit = collections.Counter()
    for inds in indices[0].values():
        raw = [i for i in inds if not nodes[i]['observed']]
        if len({nodes[i]['official_split'] for i in raw}) > 1:
            id_audit[nodes[raw[0]]['dataset']] += 1
        if len({nodes[i]['payload'] for i in raw}) > 1:
            ambiguous.update(raw)
    sets = [g.shingles(n['material']) for n in nodes]
    freq = collections.Counter(t for a in sets for t in a)
    inverted = collections.defaultdict(list)
    candidate_checks = 0
    for i, a in enumerate(sets):
        prefix = heapq.nsmallest(len(a) - math.ceil(.9*len(a)) + 1, a, key=lambda t: (freq[t], t)) if a else []
        candidates = set(j for t in prefix for j in inverted[t])
        for j in sorted(candidates):
            b = sets[j]
            if min(len(a), len(b)) / max(len(a), len(b)) < .9:
                continue
            candidate_checks += 1
            overlap = len(a & b)
            similarity = overlap / (len(a) + len(b) - overlap)
            if similarity >= .9:
                union(i, j)
                edges.append({'left': nodes[j]['key'], 'right': nodes[i]['key'], 'reason': 'shingle_jaccard', 'similarity': similarity})
        for t in prefix:
            inverted[t].append(i)
        if i % 2500 == 0:
            print('GRAPH_PROGRESS', i, '/', len(nodes), 'candidate_checks', candidate_checks, flush=True)
    clusters = collections.defaultdict(list)
    for i in range(len(nodes)):
        clusters[find(i)].append(i)
    comps = []
    component_raw_indices = {}
    node_comp = {}
    for inds in clusters.values():
        raw = [i for i in inds if not nodes[i]['observed']]
        assert raw
        seen = sorted(nodes[i]['observed_sample_id'] for i in inds if nodes[i]['observed'])
        splits = sorted({nodes[i]['official_split'] for i in raw})
        datasets = sorted({nodes[i]['dataset'] for i in raw})
        reasons = []
        if seen:
            reasons.append('observed_exploration_or_connected_source_material')
        if len(splits) > 1:
            reasons.append('component_spans_official_splits')
        if any(i in ambiguous for i in raw):
            reasons.append('conflicting_source_ID')
        if len(datasets) > 1:
            reasons.append('component_spans_datasets')
        if seen:
            role = 'exploration'
        elif reasons:
            role = 'quarantine'
        elif splits == ['train']:
            role = 'train'
        elif splits == ['test']:
            role = 'sealed_official_test'
        else:
            assert splits == ['dev']
            previous = {nodes[i]['prior_dev_role'] for i in raw}
            assert len(previous) == 1
            role = {'calibration_dev': 'calibration', 'sealed_internal_test_dev': 'sealed_internal_dev_holdout',
                    'candidate_train_dev': 'unused_dev_reserve', 'exploration': 'exploration', 'quarantine': 'quarantine'}[next(iter(previous))]
        cid = g.hh('|'.join(sorted(nodes[i]['key'] for i in inds)))
        component_raw_indices[cid] = raw
        for i in inds:
            node_comp[i] = cid
            nodes[i]['role'] = role
        counts = collections.Counter((nodes[i]['dataset'], nodes[i]['official_split']) for i in raw)
        comps.append({'component_id': cid, 'role': role, 'official_splits': splits, 'datasets': datasets,
                      'reasons': reasons, 'observed_sample_ids': seen, 'raw_nodes': [nodes[i]['key'] for i in raw],
                      'raw_rows_by_source': {ds + ':' + split: count for (ds, split), count in sorted(counts.items())}})
    manifests = {role: [] for role in protocol['roles']}
    representatives = {role: [] for role in ['train', 'calibration', 'sealed_official_test', 'sealed_internal_dev_holdout']}
    for c in sorted(comps, key=lambda x: x['component_id']):
        inds = component_raw_indices[c['component_id']]
        unique = {}
        for i in inds:
            n = nodes[i]
            for item in n['items']:
                record = dict(item, dataset=n['dataset'], official_split=n['official_split'], role=c['role'],
                              component_id=c['component_id'], source_group_id=n['source_id'],
                              material_sha256=g.hh(n['material']), normalized_material_sha256=g.hh(n['normal']), sentence_set_sha256=g.hh(n['sentences']))
                manifests[c['role']].append(record)
                unique.setdefault(item['sample_id'], record)
        if c['role'] in representatives:
            representatives[c['role']].append(min(unique.values(), key=lambda r: rank('question|' + r['sample_id'])))
    for role, rows in manifests.items():
        rows.sort(key=lambda r: (r['component_id'], r['sample_id'], r['source_line']))
        g.write_new(ROOT / 'manifests' / (role + '.jsonl'), rows, True)
    for role, rows in representatives.items():
        rows.sort(key=lambda r: rank('group|' + r['component_id']))
        g.write_new(ROOT / 'representatives' / (role + '.jsonl'), rows, True)
    g.write_new(ROOT / 'official-components.jsonl', sorted(comps, key=lambda c: c['component_id']), True)
    g.write_new(ROOT / 'official-duplicate-edges.jsonl', edges, True)
    g.write_new(ROOT / 'official-exclusions.jsonl', [c for c in comps if c['reasons']], True)
    counts = {}
    for role in protocol['roles']:
        counts[role] = {ds: {'groups': sum(c['role'] == role and ds in c['datasets'] for c in comps),
                            'question_records': sum(r['dataset'] == ds for r in manifests[role]),
                            'raw_rows': sum(c['raw_rows_by_source'].get(ds + ':' + split, 0) for c in comps if c['role'] == role for split in ['train','dev','test'])}
                        for ds in ['proofwriter', 'logiqa2']}
    summary = {'official_inventory': inventory, 'counts': counts, 'crosssplit_components': sum(len(c['official_splits']) > 1 for c in comps),
               'crosssplit_source_IDs': dict(id_audit), 'duplicate_edge_counts': dict(collections.Counter(e['reason'] for e in edges)),
               'graph_candidate_pairs': candidate_checks, 'observed_samples': 224, 'labels_generated': 0}
    g.write_new(ROOT / 'official-split-counts.json', summary)
    print('MANIFEST_COMPLETE', g.encode(summary), flush=True)
    owner = {}
    for role, rows in manifests.items():
        assert g.read_rows(ROOT / 'manifests' / (role + '.jsonl')) == rows
        for r in rows:
            for key in [('component', r['component_id']), ('source', r['dataset'], r['source_group_id']), ('sample', r['sample_id'])]:
                assert key not in owner or owner[key] == role
                owner[key] = role
            assert not g.BLOCK & set(r) and not {'state','question','options'} & set(r)
            if role == 'train':
                assert r['official_split'] == 'train'
            if role == 'calibration':
                assert r['official_split'] == 'dev'
            if role == 'sealed_official_test':
                assert r['official_split'] == 'test'
    inverse2 = collections.defaultdict(list)
    independent_checks = 0
    for i, a in enumerate(sets):
        prefix = heapq.nsmallest(len(a)-math.ceil(.9*len(a))+1, a) if a else []
        candidates = set(j for t in prefix for j in inverse2[t])
        for j in candidates:
            if node_comp[j] == node_comp[i]:
                continue
            b = sets[j]
            if min(len(a),len(b))/max(len(a),len(b)) < .9:
                continue
            independent_checks += 1
            assert len(a & b)/len(a | b) < .9, 'Independent prefix order found missing duplicate edge'
        for t in prefix:
            inverse2[t].append(i)
    print('INDEPENDENT_DUPLICATE_CHECKS', independent_checks, flush=True)
    assert sum(len(c['observed_sample_ids']) for c in comps) == 224
    for src in downloads:
        assert g.digest(src['path']) == src['raw_sha256']
    preserved = g.check_old(protected)
    hashes = {str(f.relative_to(ROOT)): g.digest(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and not f.name.endswith('.log')}
    g.write_new(ROOT / 'immutable-artifact-sha256.json', hashes)
    result = {'passed': True, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'independent_duplicate_candidate_pairs_checked': independent_checks, 'official_boundaries_preserved': True,
              'source_component_sample_role_disjoint': True, 'dev_original_hashes_unchanged': True,
              'previous_dev_holdout_not_used_for_training_or_calibration': True, 'prior_preserved': preserved,
              'source_hashes_verified': True, 'labels_generated': 0, 'model_calls': 0,
              'manifest_hashes': {role:g.digest(ROOT/'manifests'/ (role+'.jsonl')) for role in manifests}}
    g.write_new(ROOT / 'official-validation-result.json', result)
    print('OFFICIAL_VALIDATION_RESULT', g.encode(result), flush=True)


if __name__ == '__main__':
    main()
