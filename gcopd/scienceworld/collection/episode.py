"""Source collection with accepted-action history and retry control."""
from pathlib import Path
import os, sys, json, time, hashlib, re, urllib.request, collections
import gcopd.scienceworld.collection.prompts as prompts
from gcopd.scienceworld.environment import rejection_reason, choices
from gcopd.scienceworld.protocol import parse_format
from gcopd.scienceworld.collection.equivalence import equivalent as _base_equivalent
P=None
CFG=None
def equivalent(a,b):
 if _base_equivalent(a,b):return True
 # Preserve section headers and indentation; only reorder consecutive list items at the same level.
 def canonical_lists(text):
  lines=str(text).strip().splitlines();out=[];i=0
  while i<len(lines):
   line=lines[i].rstrip();indent=len(line)-len(line.lstrip())
   if indent:
    group=[]
    while i<len(lines):
     x=lines[i].rstrip()
     if len(x)-len(x.lstrip())!=indent:break
     group.append(x);i+=1
    out.extend(sorted(group))
   else:out.append(line);i+=1
  return out
 return canonical_lists(a)==canonical_lists(b)

def replay_binding(step):
 action=step['action'];feedback=step['feedback']
 # Use an explicitly named numbered instance from successful feedback, not a guessed object.
 if action.startswith('pick up '):
  m=re.fullmatch(r'You move (?:the )?(.+?) to (?:the )?inventory\.',feedback.strip())
  if m and re.search(r'\b\d+\b',m.group(1)):return 'pick up '+m.group(1).lower()
 elif action.startswith('move '):
  m=re.fullmatch(r'You move (?:the )?(.+?) to (?:the )?(.+?)\.',feedback.strip())
  if m and re.search(r'\b\d+\b',m.group(1)):return 'move '+m.group(1).lower()+' to '+m.group(2).lower()
 elif action.startswith('focus on '):
  m=re.fullmatch(r'You focus on (?:the )?(.+?)\.',feedback.strip())
  if m and re.search(r'\b\d+\b',m.group(1)):return 'focus on '+m.group(1).lower()
 return action

class TechnicalError(Exception):pass

def atomic(path,data):
 path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False));tmp.replace(path)

def sample(messages,gf,rep,step,attempt):
 seed=int(hashlib.sha256(f'{gf}|{rep}|{step}|{attempt}'.encode()).hexdigest()[:8],16)%2**31
 body={'model':CFG['model'],'messages':messages,'seed':seed,'temperature':CFG['temperature'],'top_p':1.0,'top_k':20,'max_tokens':CFG['max_tokens'],'truncate_prompt_tokens':CFG['max_prompt'],'chat_template_kwargs':{'enable_thinking':False}}
 last=None
 for http_try in range(6):
  try:
   req=urllib.request.Request(CFG['endpoint']+'/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=900) as f:res=json.load(f)
   return res['choices'][0]['message']['content'],{'seed':seed,'finish_reason':res['choices'][0]['finish_reason'],'usage':res.get('usage'),'http_retries':http_try}
  except Exception as e:last=str(e);time.sleep(min(30,2**http_try))
 raise TechnicalError('HTTP unavailable: '+str(last)[:300])

def restore(env,gf,prefix):
 last=None
 for trial in range(3):
  try:
   task,var,simp=gf.split('|');env.load(task,int(var),simp);obs,info=env.reset()
   for s in prefix:
    if not equivalent(s['observation_before'],obs):raise TechnicalError('prefix observation mismatch before '+str(s['step']))
    action=replay_binding(s)
    if action.isdigit() and action in choices(s['observation_before']):
     original_options=choices(s['observation_before']);replay_options=choices(obs)
     if original_options==replay_options:
      # Equal complete option maps retain the recorded choice, even when labels repeat.
      action=s['action']
     else:
      wanted=s['semantic_action'];hits=[k for k,v in replay_options.items() if v==wanted]
      if len(hits)!=1:raise TechnicalError('ambiguous prefix rebind: changed option map')
      action=hits[0]
    obs,_,done,info=env.step(action)
    if info['score']!=s['score'] or not equivalent(s['feedback'],obs):raise TechnicalError('prefix mismatch after '+str(s['step']))
   return obs,info
  except Exception as e:last=e
 raise TechnicalError('Cannot restore current step: '+str(last))

def run_episode(env,job):
 uid,split,gf,rep=job;cp=P/'checkpoints'/f'{uid}.json';state=json.load(open(cp)) if cp.exists() else {'accepted':[],'attempts':[],'infra_errors':[]};accepted=state['accepted'];attempts=state['attempts'];obs,info=restore(env,gf,accepted);outcome='horizon';initial_score=info['score'] if not accepted else state.get('initial_score',0);state['initial_score']=initial_score
 chat=[];raw_chat=[];hist=[]
 for s in accepted:
  chat.extend([{'role':'user','content':s['user_msg']},{'role':'assistant','content':'<action>'+s['action']+'</action>'}]);raw_chat.extend([{'role':'user','content':s['user_msg']},{'role':'assistant','content':s['raw_output']}]);hist.append(prompts.tcod_format_history(s['observation_before'],s['step'],s['action']))
 if info['score']>=100:outcome='success'
 elif accepted and accepted[-1].get('done'):outcome='environment_terminal_failure'
 while len(accepted)<CFG['horizon'] and info['score']<100 and not (accepted and accepted[-1].get('done')):
  if (P/'STOP').exists():raise TechnicalError('stop requested')
  t=len(accepted);used=sum(a['step']==t+1 for a in attempts)
  pending=next((a for a in attempts if a['step']==t+1 and a['type']=='format_invalid' and parse_format(a['raw_output'],obs)[1] is None),None)
  if used>=CFG['max_step_attempts'] and pending is None:outcome='validity_retry_exhausted';break
  user=prompts.tcod_build_user_msg(env.get_task_description(),obs,env.get_possible_actions(),env.get_possible_objects(),hist)
  messages=chat+[{'role':'user','content':user}]
  assert all('<thought>' not in m['content'] and '</thought>' not in m['content'] for m in messages if m['role']=='assistant')
  if pending:
   raw=pending['raw_output'];meta={k:pending.get(k) for k in ['seed','finish_reason','usage','http_retries']};attempt=pending;attempt['previous_type']='format_invalid';attempt['type']='candidate';attempt['reason']=None
  else:
   raw,meta=sample(messages,gf,rep,t,used);attempt={'step':t+1,'attempt':used+1,'raw_output':raw,**meta,'type':'candidate','reason':None,'time':time.time()};attempts.append(attempt)
  action,error=parse_format(raw,obs)
  if error:attempt.update(type='format_invalid',reason=error)
  if error:atomic(cp,state);continue
  before=obs;before_tick=env.get_num_moves();semantic=choices(before).get(action,action);nextobs,_,done,nextinfo=env.step(action);refusal=rejection_reason(nextobs)
  if refusal:
   attempt.update(type='environment_rejection',reason=refusal,feedback=nextobs,tick_before=before_tick,tick_after=env.get_num_moves());atomic(cp,state);obs,info=restore(env,gf,accepted);continue
  attempt['type']='accepted';step={'step':t+1,'action':action,'semantic_action':semantic,'raw_output':raw,'user_msg':user,'observation_before':before,'feedback':nextobs,'score':nextinfo['score'],'tick_before':before_tick,'tick_after':env.get_num_moves(),'done':bool(done)};accepted.append(step);atomic(cp,state)
  chat.extend([{'role':'user','content':user},{'role':'assistant','content':'<action>'+action+'</action>'}]);raw_chat.extend([{'role':'user','content':user},{'role':'assistant','content':raw}]);hist.append(prompts.tcod_format_history(before,t+1,action));obs,info=nextobs,nextinfo
  if done:outcome='success' if info['score']>=100 else 'environment_terminal_failure';break
 scores=[s['score'] for s in accepted];won=bool(info['score']>=100)
 record={'uid':uid,'split':split,'gamefile':gf,'gen_seed':rep,'won':won,'final_score':info['score'],'best_score':max([initial_score]+scores),'length':len(accepted),'score_per_step':scores,'end_reason':'success' if won else outcome,'chat':chat,'raw_chat':raw_chat,'steps':accepted,'attempts':attempts,'retry_counts':dict(collections.Counter(a['type'] for a in attempts)),'sampling_protocol':CFG,'technical_errors':state['infra_errors'],'finished':time.time()}
 atomic(P/'results'/f'{uid}.json',record);cp.unlink(missing_ok=True)
 return record
