"""CPU-only, gold-skipping group partitions; build new artifacts or verify read-only."""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

P = Path('/project')
E = P / 'experiments/kev-adaptive-reasoning'
O = P / 'output/kev-adaptive-reasoning'
F = O / 'fresh-dev24-paired-v1'
SEED = 20260930
ROLES = ['candidate_train_dev', 'calibration_dev', 'sealed_internal_test_dev']
BLOCK = {'answer', 'label', 'labels', 'QDep', 'QLen', 'strategy', 'proofs',
         'proofsWithIntermediates', 'allProofs', 'proofDetails', 'intermediates',
         'type', 'triples', 'rules'}
DECODER = json.JSONDecoder()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def hh(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def encode(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def rank(s):
    return hh('gate-data-v1|' + str(SEED) + '|' + s)


def ws(s, i):
    while i < len(s) and s[i].isspace():
        i += 1
    return i


def skip(s, i):
    i = ws(s, i)
    if s[i] == '"':
        i += 1
        while True:
            if s[i] == '\\':
                i += 2
            elif s[i] == '"':
                return i + 1
            else:
                i += 1
    if s[i] in '[{':
        level = 0
        while True:
            if s[i] == '"':
                i = skip(s, i)
                continue
            if s[i] in '[{':
                level += 1
            elif s[i] in ']}':
                level -= 1
                if not level:
                    return i + 1
            i += 1
    while i < len(s) and s[i] not in ',}]':
        i += 1
    return i


def project(s, i=0):
    i = ws(s, i)
    if s[i] == '{':
        out = {}
        i = ws(s, i + 1)
        while s[i] != '}':
            key, j = DECODER.raw_decode(s, i)
            i = ws(s, j)
            assert s[i] == ':'
            i = ws(s, i + 1)
            if key in BLOCK:
                i = skip(s, i)
            else:
                out[key], i = project(s, i)
            i = ws(s, i)
            if s[i] == ',':
                i = ws(s, i + 1)
            else:
                assert s[i] == '}'
        return out, i + 1
    if s[i] == '[':
        out = []
        i = ws(s, i + 1)
        while s[i] != ']':
            value, i = project(s, i)
            out.append(value)
            i = ws(s, i)
            if s[i] == ',':
                i = ws(s, i + 1)
            else:
                assert s[i] == ']'
        return out, i + 1
    return DECODER.raw_decode(s, i)


def norm(s):
    return ' '.join(re.findall(r'\w+', s.lower()))


def sentence_set(s):
    return '|'.join(sorted({norm(t) for t in re.split(r'[.!?]+', s) if norm(t)}))


def shingles(s):
    words = norm(s).split()
    return {tuple(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def sources():
    return [E / 'data/raw/proofwriter-owa-d5-dev.jsonl',
            E / 'data/raw/logiqa-dev.jsonl', E / 'data/inputs.jsonl',
            E / 'data/manifest.json', E / 'data/raw/proofwriter-extraction.json',
            F / 'inputs.jsonl', F / 'sample-group-manifest.json', F / 'protocol.json']


def rules():
    return {'version': 'gate-data-v1', 'seed': SEED,
            'available_official_splits': ['dev'], 'official_train_available': False,
            'official_test_available': False, 'official_final_test_status': 'blocked_missing_local_source',
            'roles': ROLES, 'official_split_field_immutable': 'dev',
            'assignment': 'Within dataset, connected components sorted by SHA256 gate-data-v1|seed|dataset|component; floor(0.60*N) train, floor(0.20*N) calibration, remainder sealed internal dev test.',
            'stratification': ['dataset'], 'forbidden_selection_fields': sorted(BLOCK),
            'group_edges': ['same dataset/source ID', 'normalized material equality',
                            'sorted unique sentence-set equality', '5-word shingle Jaccard>=0.90'],
            'exploration': '200 prior + fresh24; all connected raw groups touching any observed input excluded transitively',
            'ambiguity': 'Repeated raw source ID with unequal gold-free input payload quarantines its entire component',
            'derived_data': 'Every trajectory/prefix/option permutation inherits parent component and role; no row resplitting',
            'primary_unit': 'One hash-ranked representative question per component; all other questions remain same role',
            'limitations': ['No independent official train/test currently available',
                           'LogiQA no passage ID; material-derived grouping only',
                           'Low-overlap paraphrase/entity rename may evade matching',
                           'Cannot rule out pretraining or adapter-training contamination',
                           'Duplicates across absent official splits cannot yet be checked'],
            'sources': {str(p): digest(p) for p in sources()},
            'upstream': {'proofwriter_member': 'proofwriter-dataset-V2020.12.3/OWA/depth-5/meta-dev.jsonl',
                         'proofwriter_release': 'V2020.12.3',
                         'logiqa_revision': '955e1d3df6c59d9bfb44d9913da1e1a27ec14e18',
                         'logiqa_documentation': 'https://github.com/csitfun/LogiQA2.0',
                         'logiqa_expected_member': 'logiqa/DATA/LOGIQA/dev.txt',
                         'provenance_limit': 'Local LogiQA sha and recorded revision pinned; no fresh upstream file download/byte comparison'}}


def collect():
    nodes = []
    inventory = {}
    for dataset, file in [('proofwriter', sources()[0]), ('logiqa2', sources()[1])]:
        rows = questions = 0
        for lineno, s in enumerate(file.read_text().splitlines(), 1):
            if not s.strip():
                continue
            d, end = project(s)
            assert not s[end:].strip()
            sid = str(d['id'])
            material = d['theory'] if dataset == 'proofwriter' else d['text']
            qs = d['questions'] if dataset == 'proofwriter' else {'': d}
            items = []
            for qid, q in sorted(qs.items()):
                sample = 'pw:' + sid + ':' + qid if dataset == 'proofwriter' else 'lq:' + sid
                items.append({'sample_id': sample, 'source_path': str(file), 'source_line': lineno,
                              'source_qid': qid or None, 'question_sha256': hh(q['question']),
                              'options_sha256': hh(encode(['True', 'False', 'Unknown'] if dataset == 'proofwriter' else q['options']))})
            nodes.append({'key': dataset + ':raw:' + str(lineno), 'dataset': dataset,
                          'source_id': sid, 'official_split': 'dev', 'material': material,
                          'items': items, 'observed': False,
                          'payload_sha256': hh(encode([material, [(x['sample_id'], x['question_sha256'], x['options_sha256']) for x in items]]))})
            rows += 1
            questions += len(items)
        inventory[dataset] = {'raw_rows': rows, 'raw_questions': questions, 'official_split': 'dev',
                              'source_sha256': digest(file)}
    observed = []
    for file in [E / 'data/inputs.jsonl', F / 'inputs.jsonl']:
        for s in file.read_text().splitlines():
            d = json.loads(s)
            sample = d['id']
            dataset = 'proofwriter' if sample.startswith('pw:') else 'logiqa2'
            sid = sample.split(':')[1]
            nodes.append({'key': 'observed:' + sample, 'dataset': dataset, 'source_id': sid,
                          'official_split': 'dev', 'material': d['state'], 'items': [],
                          'observed': True, 'observed_sample_id': sample})
            observed.append(sample)
    assert len(observed) == len(set(observed)) == 224
    return nodes, inventory, sorted(observed)


def partitions():
    nodes, inventory, observed = collect()
    parent = list(range(len(nodes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    indexes = [collections.defaultdict(list) for _ in range(3)]
    edge_rows = []
    for i, n in enumerate(nodes):
        n['normalized_sha256'] = hh(norm(n['material']))
        n['sentence_set_sha256'] = hh(sentence_set(n['material']))
        for k, key in enumerate([(n['dataset'], n['source_id']), norm(n['material']), sentence_set(n['material'])]):
            if not key:
                continue
            previous = indexes[k][key]
            if previous:
                union(i, previous[0])
                edge_rows.append({'left': nodes[previous[0]]['key'], 'right': n['key'],
                                  'reason': ['source_id', 'normalized_material', 'sentence_set'][k]})
            previous.append(i)
    ambiguous = set()
    for inds in indexes[0].values():
        raw = [i for i in inds if not nodes[i]['observed']]
        if len({nodes[i]['payload_sha256'] for i in raw}) > 1:
            ambiguous.update(raw)
    sets = [shingles(n['material']) for n in nodes]
    inverted = collections.defaultdict(list)
    checked = 0
    for i, a in enumerate(sets):
        candidates = set()
        for token in a:
            candidates.update(inverted[token])
        for j in sorted(candidates):
            b = sets[j]
            if min(len(a), len(b)) / max(len(a), len(b)) < .90:
                continue
            checked += 1
            inter = len(a & b)
            if inter / (len(a) + len(b) - inter) >= .90:
                union(i, j)
                edge_rows.append({'left': nodes[j]['key'], 'right': nodes[i]['key'],
                                  'reason': 'shingle_jaccard', 'similarity': inter / (len(a) + len(b) - inter)})
        for token in a:
            inverted[token].append(i)
    clusters = collections.defaultdict(list)
    for i in range(len(nodes)):
        clusters[find(i)].append(i)
    components = []
    for inds in clusters.values():
        raw = [i for i in inds if not nodes[i]['observed']]
        if not raw:
            raise AssertionError('Observed source not represented in current raw data')
        datasets = sorted({nodes[i]['dataset'] for i in inds})
        seen = sorted(nodes[i]['observed_sample_id'] for i in inds if nodes[i]['observed'])
        reasons = []
        if seen:
            reasons.append('observed_exploration_or_connected_duplicate')
        if any(i in ambiguous for i in inds):
            reasons.append('ambiguous_repeated_source_id')
        if any(not norm(nodes[i]['material']) or not nodes[i]['items'] for i in raw):
            reasons.append('missing_material_or_questions')
        if len(datasets) > 1:
            reasons.append('cross_dataset_duplicate_component')
        role = 'exploration' if seen else 'quarantine' if reasons else None
        component = hh('|'.join(sorted(nodes[i]['key'] for i in inds)))
        components.append({'component_id': component, 'datasets': datasets, 'role': role,
                           'exclusion_reasons': reasons, 'observed_sample_ids': seen,
                           'raw_nodes': [nodes[i]['key'] for i in raw], 'raw_count': len(raw),
                           'raw_question_count': sum(len(nodes[i]['items']) for i in raw)})
    for dataset in inventory:
        eligible = sorted([c for c in components if c['role'] is None and c['datasets'] == [dataset]],
                          key=lambda c: rank(dataset + '|' + c['component_id']))
        n = len(eligible)
        a, b = math.floor(.60 * n), math.floor(.20 * n)
        for j, c in enumerate(eligible):
            c['role'] = ROLES[0] if j < a else ROLES[1] if j < a + b else ROLES[2]
    bykey = {n['key']: n for n in nodes}
    manifests = {r: [] for r in ROLES + ['exploration', 'quarantine']}
    reps = {r: [] for r in ROLES}
    for c in sorted(components, key=lambda x: x['component_id']):
        assert c['role'] is not None
        unique = {}
        for key in c['raw_nodes']:
            n = bykey[key]
            for item in n['items']:
                r = dict(item, dataset=n['dataset'], official_split='dev', role=c['role'],
                         component_id=c['component_id'], source_group_id=n['source_id'],
                         material_sha256=hh(n['material']), normalized_material_sha256=n['normalized_sha256'],
                         sentence_set_sha256=n['sentence_set_sha256'])
                manifests[c['role']].append(r)
                unique.setdefault(item['sample_id'], r)
        if c['role'] in ROLES:
            reps[c['role']].append(min(unique.values(), key=lambda x: rank('representative|' + x['sample_id'])))
    for r in manifests:
        manifests[r].sort(key=lambda x: (x['component_id'], x['sample_id'], x['source_line']))
    for r in reps:
        reps[r].sort(key=lambda x: rank('stage|' + x['component_id']))
    stats = {role: {ds: {'components': sum(c['role'] == role and ds in c['datasets'] for c in components),
                        'raw_rows': sum(c['raw_count'] for c in components if c['role'] == role and c['datasets'] == [ds]),
                        'question_records': sum(r['dataset'] == ds for r in manifests[role])}
                   for ds in inventory} for role in manifests}
    return {'inventory': inventory, 'observed_sample_count': len(observed), 'counts': stats,
            'official_final_test': {'groups': 0, 'questions': 0, 'status': 'missing_local_official_test'},
            'shingle_pairs_checked': checked, 'edge_counts': dict(collections.Counter(e['reason'] for e in edge_rows))}, components, edge_rows, manifests, reps


def write_new(path, obj, jsonl=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        if jsonl:
            for row in obj:
                f.write(encode(row) + '\n')
        elif isinstance(obj, str):
            f.write(obj)
        else:
            f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n')


def schema():
    nullable_number = {'type': ['number', 'null']}
    outcome = {'type': 'object', 'additionalProperties': False,
               'required': ['prediction', 'correct', 'end_reason', 'actual_generated_tokens', 'failure'],
               'properties': {'prediction': {'type': ['integer', 'null']}, 'correct': {'type': ['boolean', 'null']},
                              'end_reason': {'enum': ['stop_now', 'end_marker', 'eos', 'message_boundary', 'cap', 'failure']},
                              'actual_generated_tokens': {'type': 'integer', 'minimum': 0},
                              'failure': {'type': ['string', 'null']}}}
    fields = {'record_id': {'type': 'string'}, 'sample_id': {'type': 'string'},
              'component_id': {'type': 'string'}, 'partition': {'enum': ROLES},
              'trajectory_id': {'type': 'string'}, 'gate_stage': {'enum': ['initial', 'continue']},
              'prefix_token_count': {'type': 'integer', 'minimum': 0}, 'prefix_ids_sha256': {'type': 'string'},
              'state_hashes': {'type': 'object', 'required': ['input', 'position', 'mask', 'cache', 'probe_copy_unchanged']},
              'model_policy_sha256': {'type': 'string'}, 'feature_artifact_sha256': {'type': 'string'},
              'feature_origins': {'type': 'array', 'items': {'enum': ['input', 'current_prefix', 'current_hidden', 'current_probabilities', 'observed_runtime']}},
              'action': {'type': 'object', 'required': ['stop_action', 'continue_action', 'additional_token_cap', 'policy'],
                         'properties': {'stop_action': {'const': 'classify_current_prefix'},
                                        'continue_action': {'type': 'string'},
                                        'additional_token_cap': {'type': 'integer', 'minimum': 0},
                                        'policy': {'type': 'object', 'required': ['weights_sha256', 'head_sha256', 'code_sha256', 'tokenizer_sha256', 'prompt_sha256', 'precision', 'greedy', 'temperature', 'top_p', 'top_k', 'seed', 'attention', 'fused', 'graphs', 'stop_token_ids', 'stop_marker', 'classification_rule', 'timing_boundary', 'feature_version']}}},
              'stop_outcome': outcome, 'continue_outcome': outcome,
              'four_cell': {'enum': ['both_correct', 'wrong_to_right', 'right_to_wrong', 'both_wrong', 'unavailable']},
              'correctness_delta': {'type': ['integer', 'null'], 'minimum': -1, 'maximum': 1},
              'cost': {'type': 'object', 'required': ['stop_e2e_seconds', 'continue_e2e_seconds', 'delta_seconds', 'components', 'peak_memory_bytes'],
                       'properties': {'stop_e2e_seconds': nullable_number, 'continue_e2e_seconds': nullable_number,
                                      'delta_seconds': nullable_number, 'components': {'type': 'object'},
                                      'peak_memory_bytes': {'type': ['integer', 'null']}}},
              'gold_reference_sha256': {'type': 'string'}, 'label_status': {'enum': ['completed', 'branch_failure', 'missing_counterfactual']}}
    return {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'title': 'Gate paired counterfactual label v1',
            'type': 'object', 'additionalProperties': False, 'required': list(fields), 'properties': fields}


PROTOCOL_DOC = '''# 分组与门控监督协议 v1

本阶段仅CPU数据准备；无模型调用、训练、标签生成或数据下载。现有200题和fresh24仅探索，连通材料组全部排除。
本地仅官方dev。candidate_train_dev、calibration_dev、sealed_internal_test_dev均来自dev，不能称官方train/test；official_final_test为空且阻塞。
LogiQA官方README明确区分train.txt/dev.txt/test.txt：https://github.com/csitfun/LogiQA2.0 。本地原始文件SHA与既有版本记录固定，未新下载上游文件作字节比对。

## 冻结划分

以source ID、规范化材料、句子集合和5词shingle Jaccard>=0.90建立无向图，按连通分量隔离，探索污染沿图传递。歧义重复source ID整组隔离。
仅按dataset分层，seed=20260930；完整规则见protocol.json。无gold、QDep、reasoning type、模型对错或文本质量挑选。
剩余分量以固定SHA顺序60/20/20分配；全问题继承组，代表问题按独立固定SHA选择。所有前缀/轨迹/选项置换必须继承组，不得重新按行切分。
manifest只有来源定位、ID、角色和hash，不复制题目或gold。封存内部dev test不运行模型、不向用户展示题目/标签。官方test目前0。
未来若获授权增加官方train/test：新版本保留官方边界；跨split重复整组隔离，test组不能进入train，不能将当前封存组重新分配来改善效果。
无法排除低重合改写、实体替换或模型预训练/适配训练污染。当前没有跨缺失split检查的证据。

## 初始gate

比较stop-now原生分类和固定继续动作（候选Kev-state greedy至96或合法stop）的正确性差与真实端到端成本。prompt0、copy32/96保留作混杂控制。
四格both_correct/wrong_to_right/right_to_wrong/both_wrong全部保存，不能只将direct错误标为生成。目标可为delta_correctness-lambda*delta_cost；lambda或资源约束须外部声明，再仅在calibration选择，报告Pareto。
运行特征限当前输入、prefix、hidden/probs及已发生成本。gold、未来答案、未来hidden、最终长度、未来是否完成均禁止作为部署特征。

## 继续gate

从同一prefix状态分支stop-now与continue。prefix IDs/输入/position/mask/cache哈希必须相同，评分probe复制缓存且检查原cache不变。
候选动作预声明：32->96（额外至多64），96->512（额外至多416）；合法停止沿已有固定协议。此前已触发技术stop的轨迹不重开。
固定0/32/96/512基线必须独立在线计时。一个512轨迹的中间prefix可离线复用监督材料，但不能冒充独立在线32/96延迟；分支执行时间需另行实测。
保存四格、delta_correctness、stop/continue预测、actual tokens、end/cap/failure、真实增量成本和阶段成本。失败保留且不可默默删掉；两支可比较性不足时标missing_counterfactual，不伪造delta。
greedy单轨迹标签是固定确定性策略的反事实结果，不是无偏的期望净收益估计。随机policy需独立种子重复并报告估计不确定性。
policy完整记录权重/代码/tokenizer/prompt、precision、greedy/temperature/top-p/top-k、seed、mask/position、cache/probe、stop token IDs/marker、预算及耗时边界。

## 训练与评价隔离

训练只用candidate_train_dev；阈值、校准和Pareto操作点只用calibration_dev；封存内部评估不得调阈值。官方最终结论待独立官方heldout数据。
阶段模型、head、prompt或继续policy变化必须版本化重生成相关训练/校准标签和重新校准，不沿用旧model-policy label。未来测试只在冻结后按预声明程序评分。
基线：direct/fixed0，prompt0，fixed32/96/512，maximum confidence、margin、entropy、learned初始gate、learned继续gate；copy32/96为材料控制。报告数据集分层与全部失败/cap，不只完成子集。
成本包括head/probe/cache复制/gate/sync/编码/生成/解码/log；日志策略一致；模型加载单列，真实峰值含模型heads/cache。在线策略报告平均/中位/P90/P95、token、峰值显存、WR/RW与配对差。
不锁定1pp容忍或以本24dev选择阈值后宣称结果；oracle route只作离线可选择空间，不能报成在线效果。
推导正确性、语义完成和分类正确分开。答对但复述或错误推导不得自动当可靠思维链正例。

## 下阶段提案，尚未授权执行

小试：32个train代表组（每类16）+16个calibration代表组（每类8），不触碰封存区；7条件每题独立，最多336formal+7warmup=343调用，生成<=48*(32+96+512)+48=30768 tokens。
优先验证标签schema、相同prefix分支和成本口径；任何额外分支调用须另列预算，不能假称上述7条件已包含所有真实cache分支监督成本。
以既有24题均值估计核心7条件约22分钟；全部生成触顶按0.1秒/token粗估约54分钟，输入长度/环境不同会改变实际时间。模型加载、额外branch/gate开销另计。
正式候选（需元数据/官方heldout缺口先解决）：train每类128组、calibration每类32组、独立评估每类64组，共448组，7条件3136formal+7warmup，核心生成上限286768 tokens。可复用同policy已收集小试但不能重复计为新样本。
这些规模是建议而非当前运行授权；若当前某分层不足则缩小或先报告，不能靠重复问题充数。当前不启动任何GPU、模型或训练。

## CPU复核

python3 -B source/gate_data_v1.py validate --root THIS_DIRECTORY
validate只读重算全部连通图/分配、比较每个manifest、验证所有边两端同组同role、来源/组/ID无跨role、代表与pilot继承组、保护旧文件hash。
无标签记录当前存在；schema是未来格式，未生成虚构训练样本或标签值。
'''


def protect():
    old = json.loads((F / 'preserved-existing-sha256.json').read_text())
    paths = set(old)
    paths.update(json.loads((F / 'protocol.json').read_text())['frozen_source_hashes'])
    paths.update(str(p) for p in F.rglob('*') if p.is_file())
    paths.update(str(p) for p in (E / 'data').rglob('*') if p.is_file())
    paths.add(str(E / 'README.md'))
    paths.add(str(E / 'prepare_data.py'))
    return {s: digest(s) for s in sorted(paths)}


def check_old(hashes):
    changed = [p for p, h in hashes.items() if digest(p) != h]
    assert not changed, 'Protected prior files changed'
    return {'protected_files': len(hashes), 'changed': 0}


def build(root):
    assert not (root / 'protocol.json').exists(), 'Version already frozen; never overwrite'
    frozen = rules()
    frozen['script_sha256'] = digest(__file__)
    write_new(root / 'protocol.json', frozen)
    protected = protect()
    write_new(root / 'protected-prior-sha256.json', protected)
    summary, comps, edges, manifests, reps = partitions()
    write_new(root / 'inventory-and-counts.json', summary)
    write_new(root / 'components.jsonl', comps, True)
    write_new(root / 'duplicate-edges.jsonl', edges, True)
    write_new(root / 'exclusions.jsonl', [c for c in comps if c['exclusion_reasons']], True)
    for role, rows in manifests.items():
        write_new(root / 'manifests' / (role + '.jsonl'), rows, True)
    for role, rows in reps.items():
        write_new(root / 'representatives' / (role + '.jsonl'), rows, True)
    write_new(root / 'manifests/official_final_test.jsonl', [], True)
    pilot = []
    for role, count in [(ROLES[0], 16), (ROLES[1], 8)]:
        for ds in summary['inventory']:
            pool = [r for r in reps[role] if r['dataset'] == ds]
            assert len(pool) >= count, 'Not enough disjoint representative groups for proposed pilot'
            pilot += pool[:count]
    write_new(root / 'proposed-pilot-manifest.jsonl', pilot, True)
    write_new(root / 'gate-label.schema.json', schema())
    write_new(root / 'supervision-and-evaluation-protocol.md', PROTOCOL_DOC)
    frozen_policy = json.loads((F / 'protocol.json').read_text())
    write_new(root / 'policy-reference.json', {k: frozen_policy[k] for k in ['prompt_template', 'end_marker', 'technical_end_ids', 'settings', 'stop_rules', 'timing', 'memory', 'frozen_source_hashes', 'vendor_commit']})
    write_new(root / 'label-status.json', {'records_generated': 0, 'status': 'schema_only_no_model_calls'})
    old_status = check_old(protected)
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write_new(root / 'artifact-sha256.json', hashes)
    print('BUILD_COMPLETE', encode({'summary': summary, 'protected': old_status, 'artifact_count': len(hashes)}), flush=True)


def read_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def validate(root):
    frozen = json.loads((root / 'protocol.json').read_text())
    assert frozen['script_sha256'] == digest(__file__)
    assert {k: frozen[k] for k in rules()} == rules()
    hashes = json.loads((root / 'artifact-sha256.json').read_text())
    assert all(digest(root / p) == h for p, h in hashes.items())
    summary, comps, edges, manifests, reps = partitions()
    assert json.loads((root / 'inventory-and-counts.json').read_text()) == summary
    assert read_rows(root / 'components.jsonl') == comps
    assert read_rows(root / 'duplicate-edges.jsonl') == edges
    assert read_rows(root / 'exclusions.jsonl') == [c for c in comps if c['exclusion_reasons']]
    node_component = {n: c['component_id'] for c in comps for n in c['raw_nodes']}
    for c in comps:
        for s in c['observed_sample_ids']:
            node_component['observed:' + s] = c['component_id']
    assert all(node_component[e['left']] == node_component[e['right']] for e in edges)
    owner = {}
    for role, expected in manifests.items():
        actual = read_rows(root / 'manifests' / (role + '.jsonl'))
        assert actual == expected
        for r in actual:
            for key in [('component', r['component_id']), ('source', r['dataset'], r['source_group_id']), ('sample', r['sample_id'])]:
                assert key not in owner or owner[key] == role
                owner[key] = role
            assert not (BLOCK & set(r)) and 'state' not in r and 'question' not in r and 'options' not in r
    for role, expected in reps.items():
        assert read_rows(root / 'representatives' / (role + '.jsonl')) == expected
        assert len(expected) == len({r['component_id'] for r in expected})
    assert not read_rows(root / 'manifests/official_final_test.jsonl')
    pilot = read_rows(root / 'proposed-pilot-manifest.jsonl')
    assert len(pilot) == len({r['component_id'] for r in pilot}) == 48
    assert all(r in reps[r['role']] and r['role'] != ROLES[2] for r in pilot)
    assert json.loads((root / 'label-status.json').read_text())['records_generated'] == 0
    protected = check_old(json.loads((root / 'protected-prior-sha256.json').read_text()))
    result = {'passed': True, 'source_hashes_verified': True, 'full_graph_recomputed': True,
              'all_duplicate_edges_within_components': True, 'all_role_pairs_disjoint': True,
              'representatives_and_pilot_inherit_groups': True, 'official_final_test_empty': True,
              'label_records': 0, 'artifacts_verified': len(hashes), 'protected_prior': protected,
              'counts': summary['counts']}
    print('VALIDATION_RESULT', encode(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'validate'])
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    {'build': build, 'validate': validate}[args.action](args.root)
