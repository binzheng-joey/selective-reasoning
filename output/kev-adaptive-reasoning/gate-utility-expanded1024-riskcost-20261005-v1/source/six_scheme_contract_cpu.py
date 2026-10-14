"""Prompt/parse contract for already-observed32; no imports of torch or model APIs."""
import collections,json,re
R_SYSTEM='Use only the supplied material. Do not invent missing facts. Keep the original candidate order. Follow the requested output format.'
def problem(row):return 'Material:\n'+row['state']+'\n\nTask:\n'+row['instr']+'\n\nCandidates (in the given order):\n'+'\n'.join(f'{i+1}. {v}' for i,v in enumerate(row['options']))
def B_prompt(row):return problem(row)+'\n\nProvide a derivation of the relevant facts and logical constraints. Do not explicitly state the final answer, the selected option number or letter, or identify any candidate as the final choice. Do not use a final-answer tag. End the derivation with [END_REASONING].'
def F_prompt(row):return problem(row)+'\n\nReason through the task using only the material. Then give exactly one final candidate index from 1 to '+str(len(row['options']))+' in the exact format <FINAL>index</FINAL>. Place this final tag after your reasoning, outside any thinking block. Do not append text after the tag.'
def A_prompt(row):return problem(row)+'''\n\nDecompose the reasoning task into all subquestions you consider necessary. Choose the number yourself: there is no fixed number of subquestions. Do not omit a needed subquestion to meet a preferred count. Each subquestion must have at least two explicit candidate answers and depend only on the original material and explicitly listed previous subquestions. Return one complete JSON object, outside any thinking block, with exactly this schema:
{"subquestions":[{"id":"unique string","question":"standalone question","candidates":["candidate in original local order","another candidate"],"depends_on":["id of a previous subquestion"]}]}
List every required node and dependency. Use an empty depends_on list for independent nodes. Do not give answers to the subquestions or the final problem. The dependency graph must be acyclic. Return only the JSON object after any thinking block. A truncated object is invalid; do not output a partial plan.'''
def A_merge_prompt(row,plan,results):return problem(row)+'\n\nComplete decomposition DAG:\n'+json.dumps(plan,ensure_ascii=False,sort_keys=True)+'\n\nKev subquestion judgments (unverified, in plan order):\n'+json.dumps(results,ensure_ascii=False,sort_keys=True)+'\n\nUse the original task, complete decomposition and all Kev judgments to decide the original candidate. Do not answer the subquestions again or introduce new subquestions. Explain the combination, then place exactly one final original-candidate index in <FINAL>index</FINAL> outside any thinking block; no text after the tag.'
def self_question_state(row,prompt0_text,initial_prediction):return prompt0_text+'\n\nInitial classification (not verified): option '+str(initial_prediction+1)+'.\nSelf-assessment: 判断当前任务是否还需要额外推理才能可靠选择原候选答案。'
SELF_QUESTION='当前任务是否还需要额外推理才能可靠选择原候选答案？'
SELF_CANDIDATES=['是','否','无法确定']
def visible_text(tok,ids):
 # Uniform native rendering: remove special transport/control tokens, keep all
 # ordinary generated reasoning/answer text unchanged. Raw IDs and raw decode
 # with skip_special_tokens=False must also be saved for audits.
 return tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
def after_thinking(text,thinking_enabled):
 if thinking_enabled:
  if '</think>' not in text:raise ValueError('thinking block never closed')
  return text.rsplit('</think>',1)[1].strip()
 return text.strip()
def F_parse(text,n,thinking_enabled=True):
 body=after_thinking(text,thinking_enabled);matches=list(re.finditer(r'<FINAL>([0-9]+)</FINAL>',body))
 if len(matches)!=1 or body[matches[0].end():].strip():raise ValueError('missing/multiple/nonterminal finalindex tag')
 idx=int(matches[0].group(1))
 if not 1<=idx<=n:raise ValueError('finalindex out of candidate range')
 return idx-1
def B_violation(text):
 patterns=[r'<FINAL>',r'(?im)^\s*(?:final answer|the answer|answer|correct answer|correct option)\s*[:=]',r'(?i)\b(?:the )?(?:correct|final) (?:answer|choice|option) (?:is|would be)\b',r'(?i)\b(?:therefore|thus|hence),? (?:the )?(?:answer|option|choice) (?:is|must be)\b']
 return dict(explicit_answer_violation=any(re.search(p,text) for p in patterns),matched_patterns=[p for p in patterns if re.search(p,text)],treatment='record violation; pass original visible text unedited; no stripF/manualpolish/repairgen',scope='frozen lexical detector; implicit answer leakage may remain')
def A_parse(text,thinking_enabled=True):
 body=after_thinking(text,thinking_enabled)
 if body.startswith('```json\n') and body.endswith('\n```'):body=body[8:-4]
 obj=json.loads(body)
 if not isinstance(obj,dict) or set(obj)!={'subquestions'} or not isinstance(obj['subquestions'],list) or not obj['subquestions']:raise ValueError('invalid/empty complete plan')
 nodes=obj['subquestions'];by={}
 for node in nodes:
  if not isinstance(node,dict) or set(node)!={'id','question','candidates','depends_on'}:raise ValueError('illegal node schema')
  sid=node['id'];q=node['question'];cs=node['candidates'];deps=node['depends_on']
  if not isinstance(sid,str) or not sid.strip() or sid in by or not isinstance(q,str) or not q.strip():raise ValueError('invalid/duplicate node identity')
  if not isinstance(cs,list) or len(cs)<2 or any(not isinstance(c,str) or not c.strip() for c in cs) or len(set(cs))!=len(cs):raise ValueError('invalid candidates')
  if not isinstance(deps,list) or any(not isinstance(d,str) for d in deps) or len(set(deps))!=len(deps):raise ValueError('invalid dependencies')
  by[sid]=node
 if any(sid in node['depends_on'] or any(d not in by for d in node['depends_on']) for sid,node in by.items()):raise ValueError('self/unknown dependency')
 done=set();layers=[]
 while len(done)<len(by):
  ready=[node['id'] for node in nodes if node['id'] not in done and set(node['depends_on'])<=done]
  if not ready:raise ValueError('cyclic dependency graph')
  layers.append(ready);done.update(ready)
 return dict(nodes=nodes,layers=layers,subquestion_count=len(nodes),hard_subquestion_count_cap=None)
def subquestion_input(row,node,results):
 deps=[results[s] for s in node['depends_on']]
 return dict(id=row['id']+'|'+node['id'],state=row['state']+'\n\nDeclared dependency judgments (unverified):\n'+json.dumps(deps,ensure_ascii=False),instr=node['question'],options=node['candidates'])
