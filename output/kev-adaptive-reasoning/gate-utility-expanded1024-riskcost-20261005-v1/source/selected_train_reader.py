import json,hashlib
def encode(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))
DEC=json.JSONDecoder()
def ws(s,i):
 while i<len(s) and s[i].isspace():i+=1
 return i
def skip(s,i):
 i=ws(s,i)
 if s[i]=='"':
  i+=1
  while True:
   if s[i]=='\\':i+=2
   elif s[i]=='"':return i+1
   else:i+=1
 if s[i] in '[{':
  depth=0
  while True:
   if s[i]=='"':i=skip(s,i);continue
   if s[i] in '[{':depth+=1
   elif s[i] in ']}':
    depth-=1
    if depth==0:return i+1
   i+=1
 while i<len(s) and s[i] not in ',}]':i+=1
 return i
def fields(s,start=0):
 i=ws(s,start);assert s[i]=='{';i=ws(s,i+1);out={}
 while s[i]!='}':
  k,j=DEC.raw_decode(s,i);i=ws(s,j);assert s[i]==':';a=ws(s,i+1);b=skip(s,a);out[k]=(a,b);i=ws(s,b)
  if s[i]==',':i=ws(s,i+1)
  else:assert s[i]=='}'
 return out
def extract(s,m):
 fs=fields(s)
 get=lambda k:json.loads(s[slice(*fs[k])])
 assert str(get('id'))==m['source_group_id']
 if m['dataset']=='proofwriter':
  qs=fields(s,fs['questions'][0]);qf=fields(s,qs[m['source_qid']][0]);a,b=qf['question'];q=json.loads(s[a:b]);state=get('theory');options=['True','False','Unknown']
 else:q=get('question');state=get('text');options=get('options')
 assert hashlib.sha256(q.encode()).hexdigest()==m['question_sha256']
 assert hashlib.sha256(encode(options).encode()).hexdigest()==m['options_sha256']
 assert hashlib.sha256(state.encode()).hexdigest()==m['material_sha256']
 return dict(id=m['sample_id'],dataset=m['dataset'],component_id=m['component_id'],state=state,instr=q,options=options)
