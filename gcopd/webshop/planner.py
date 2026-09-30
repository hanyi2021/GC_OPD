"""Historical WebShop training oracle: target, query, equivalent, broad and catalog search.

Planner candidates must be executed and replayed before becoming graph sources.
"""
import functools
import html
import json
from pathlib import Path
import random
import re
import sys
import uuid
from gcopd.webshop.build_graph import http

GOALS = []
ATTRS = {}


def load_dependencies(webshop_root):
    global normalize_color, fuzz, get_reward
    sys.path.insert(0, str(Path(webshop_root).resolve()))
    from web_agent_site.engine.normalize import normalize_color
    from web_agent_site.engine.goal import get_reward
    from thefuzz import fuzz


def initial(goal, truth, env):
    eid='goal_oracle_'+uuid.uuid4().hex
    try:
        state=http(env+'/reset',{'episode':eid,'goal':goal})
        assert state['instruction']==truth['instruction_text']
        ep={'goal':goal,'rep':16,'tag':'privileged_goal_oracle_v1','source_kind':'privileged_goal_oracle_v1','mode':'train','model':None,'instruction':state['instruction'],'steps':[],'env_url':env,'protocol':{'planner':'privileged_goal_oracle_v1','privileged_fields':['name','asin','goal_options','query'],'goal_shuffle_seed':233,'env_seed':42,'max_steps':15},'complete':False}
        queries=list(dict.fromkeys(' '.join(str(q).replace('[',' ').replace(']',' ').split()) for q in [truth['name'],truth.get('query','')] if q))
        qi=0;selected=set();reason='decision_limit'
        for turn in range(15):
            available=state['available_actions']['clickables'];lower={a.lower():a for a in available}
            pt=state['state']['page_type']
            if pt=='start':
                action=f'search[{queries[qi]}]';qi+=1
            elif pt=='search_results':
                if truth['asin'].lower() in lower:action=f"click[{truth['asin'].lower()}]"
                elif 'next >' in lower:action='click[next >]'
                elif qi<len(queries):action=f'search[{queries[qi]}]';qi+=1
                else:reason='target_not_found';break
            elif pt=='item_page':
                todo=[str(o).lower() for o in truth.get('goal_options',[]) if str(o).lower() not in selected]
                if todo:
                    if todo[0] not in lower:reason='required_option_unavailable';break
                    action=f'click[{todo[0]}]';selected.add(todo[0])
                else:action='click[buy now]'
            else:reason='unexpected_page';break
            before=state;state=http(env+'/step',{'episode':eid,'action':action})
            ep['steps'].append({'turn':turn,'obs':before['obs'],'obs_fmt':before['obs'],'action':action,'raw_output':f'<action>{action}</action>','state_before':before['state'],'state_after':state['state'],'feedback':state['obs'],'reward':state.get('reward',0),'done':state.get('done',False),'executed':True,'format_error':None,'env_rejection':None,'attempts':[]})
            if state.get('done'):reason='environment_terminal';break
        ep.update(complete=True,done=bool(state.get('done')),score=state.get('reward',0),won=bool(state.get('done')) and state.get('reward',0)>=1-1e-8,decision_count=len(ep['steps']),stop_reason=reason,n_format_errors=0,n_env_rejections=0)
        assert ep['steps']
        replay=http(env+'/replay',{'goal':goal,'actions':[s['action'] for s in ep['steps']]})
        assert replay['obs']==[ep['steps'][0]['obs']]+[s['feedback'] for s in ep['steps']]
        assert [s['strict_key'] for s in replay['states'][1:]]==[s['state_after']['strict_key'] for s in ep['steps']]
        assert replay['dones']==[s['done'] for s in ep['steps']]
        assert all(abs(a-b['reward'])<1e-9 for a,b in zip(replay['rewards'],ep['steps']))
        ep['replay_verified']=True
        return ep
    finally:
        http(env+'/close', {'episode':eid})


def norm(text):return ' '.join(html.unescape(str(text)).lower().split())


def query_attempt(goal,truth,query,env,index):
 eid='oracle_refine_'+uuid.uuid4().hex;state=http(env+'/reset',{'episode':eid,'goal':goal});assert state['instruction']==truth['instruction_text']
 steps=[];selected=set();extra=set();pages=0;reason='decision_limit'
 try:
  for turn in range(15):
   av={norm(x):str(x).lower() for x in state['available_actions']['clickables']};pt=state['state']['page_type']
   if pt=='start':action=f'search[{query}]';pages=1
   elif pt=='search_results':
    if truth['asin'].lower() in av:action=f"click[{truth['asin'].lower()}]"
    elif 'next >' in av and pages<(1 if index==0 else 5):action='click[next >]';pages+=1
    else:reason='query_no_target_in_top50';break
   elif pt=='item_page':
    todo=[norm(x) for x in truth.get('goal_options',[]) if norm(x) not in selected]
    if not todo:action='click[buy now]'
    elif todo[0] in av:action=f'click[{av[todo[0]]}]';selected.add(todo[0])
    elif 'customsize' in av and 'customsize' not in extra:action='click[customsize]';extra.add('customsize')
    else:reason='required_option_unavailable';break
   else:reason='unexpected_page';break
   before=state;state=http(env+'/step',{'episode':eid,'action':action})
   steps.append({'turn':turn,'obs':before['obs'],'obs_fmt':before['obs'],'action':action,'raw_output':f'<action>{action}</action>','state_before':before['state'],'state_after':state['state'],'feedback':state['obs'],'reward':state.get('reward',0),'done':state.get('done',False),'executed':True,'format_error':None,'env_rejection':None,'attempts':[]})
   if state.get('done'):reason='environment_terminal';break
  score=state.get('reward',0) if state.get('done') else 0
  ep={'goal':goal,'rep':17,'tag':'privileged_goal_oracle_query_search_v2','mode':'train','model':None,'instruction':truth['instruction_text'],'steps':steps,'env_url':env,'protocol':{'planner':'privileged_goal_oracle_query_search_v2','max_steps':15,'goal_shuffle_seed':233,'env_seed':42},'complete':True,'done':bool(state.get('done')),'score':score,'won':bool(state.get('done')) and score>=1-1e-8,'decision_count':len(steps),'stop_reason':reason,'query':query,'candidate_index':index,'replay_verified':False}
  if ep['won']:
   replay=http(env+'/replay',{'goal':goal,'actions':[s['action'] for s in steps]})
   assert replay['obs']==[steps[0]['obs']]+[s['feedback'] for s in steps]
   assert replay['dones']==[s['done'] for s in steps]
   assert [s['strict_key'] for s in replay['states'][1:]]==[s['state_after']['strict_key'] for s in steps]
   assert replay['rewards'][-1]>=1-1e-8;ep['replay_verified']=True
  return ep
 finally:
  try:http(env+'/close',{'episode':eid})
  except Exception:pass


def plan_options(obs,goals):
 groups={};header=None
 for line in obs.splitlines():
  match=re.match(r'^\s{2,}\[(?:clicked )?button\] (.*?) \[(?:clicked )?button_\]',line)
  if match and header:groups.setdefault(header,[]).append(match[1].lower())
  elif line.strip() and '[button]' not in line and '[clicked button]' not in line:header=line.strip().lower()
 target=(1<<len(goals))-1;states={0:[]}
 for name,values in groups.items():
  new=dict(states)
  for value in values:
   mask=sum(1<<j for j,goal in enumerate(goals) if fuzz.token_set_ratio(normalize_color(value),normalize_color(goal))>85)
   if not mask:continue
   for old,choices in states.items():
    key=old|mask;candidate=choices+[value]
    if key not in new or len(candidate)<len(new[key]):new[key]=candidate
  states=new
 return states.get(target)


def discover_equivalent(goal,truth,env,query):
 eid='oracle_discover_'+uuid.uuid4().hex;http(env+'/reset',{'episode':eid,'goal':goal});out=[]
 try:
  state=http(env+'/step',{'episode':eid,'action':f'search[{query}]'})
  for page in range(1,4):
   lines=state['obs'].splitlines()
   for i,line in enumerate(lines):
    m=re.match(r'^\[button\] ([A-Za-z0-9]{10}) \[button_\]$',line)
    if m and i+1<len(lines) and fuzz.token_set_ratio(truth['name'].lower(),lines[i+1].lower())>=70:out.append((m[1],page))
   if page==3 or 'next >' not in state['available_actions']['clickables']:break
   state=http(env+'/step',{'episode':eid,'action':'click[next >]'})
 finally:http(env+'/close',{'episode':eid})
 return out


def equivalent_attempt(goal,truth,env,query,asin,page,index):
 eid='oracle_equivalent_'+uuid.uuid4().hex;state=http(env+'/reset',{'episode':eid,'goal':goal});steps=[]
 def take(action):
  nonlocal state
  before=state;state=http(env+'/step',{'episode':eid,'action':action})
  steps.append({'turn':len(steps),'obs':before['obs'],'obs_fmt':before['obs'],'action':action,'raw_output':f'<action>{action}</action>','state_before':before['state'],'state_after':state['state'],'feedback':state['obs'],'reward':state.get('reward',0),'done':state.get('done',False),'executed':True,'format_error':None,'env_rejection':None,'attempts':[]})
 try:
  take(f'search[{query}]')
  for _ in range(page-1):take('click[next >]')
  if asin.lower() not in state['available_actions']['clickables']:return None
  take(f'click[{asin.lower()}]')
  plan=plan_options(state['obs'],truth.get('goal_options',[]))
  price=re.search(r'Price: \$([0-9,.]+)',state['obs'])
  if plan is None or (price and float(price[1].replace(',',''))>truth['price_upper']):return None
  if len(steps)+len(plan)+1>15:return None
  for value in plan:take(f'click[{value}]')
  take('click[buy now]')
  ep={'goal':goal,'rep':17,'tag':'privileged_goal_oracle_equivalent_product_v3','mode':'train','model':None,'instruction':truth['instruction_text'],'steps':steps,'env_url':env,'protocol':{'planner':'privileged_goal_oracle_equivalent_product_v3','max_steps':15,'goal_shuffle_seed':233,'env_seed':42},'complete':True,'done':state['done'],'score':state['reward'],'won':bool(state['done']) and state['reward']>=1-1e-8,'query':query,'candidate_asin':asin,'candidate_index':index,'replay_verified':False}
  if ep['won']:
   replay=http(env+'/replay',{'goal':goal,'actions':[s['action'] for s in steps]})
   assert replay['obs']==[steps[0]['obs']]+[s['feedback'] for s in steps]
   assert replay['dones']==[s['done'] for s in steps]
   assert [s['strict_key'] for s in replay['states'][1:]]==[s['state_after']['strict_key'] for s in steps]
   assert replay['rewards'][-1]>=1-1e-8;ep['replay_verified']=True
  return ep
 finally:http(env+'/close',{'episode':eid})


def discover_broad(goal,truth,env,query):
 eid='oracle_discover_'+uuid.uuid4().hex;http(env+'/reset',{'episode':eid,'goal':goal});out=[]
 try:
  state=http(env+'/step',{'episode':eid,'action':f'search[{query}]'})
  for page in range(1,4):
   lines=state['obs'].splitlines()
   for i,line in enumerate(lines):
    m=re.match(r'^\[button\] ([A-Za-z0-9]{10}) \[button_\]$',line)
    if m and i+1<len(lines) :out.append((m[1],page))
   if page==3 or 'next >' not in state['available_actions']['clickables']:break
   state=http(env+'/step',{'episode':eid,'action':'click[next >]'})
 finally:http(env+'/close',{'episode':eid})
 return out


def broad_attempt(goal,truth,env,query,asin,page,index):
 eid='oracle_equivalent_'+uuid.uuid4().hex;state=http(env+'/reset',{'episode':eid,'goal':goal});steps=[]
 def take(action):
  nonlocal state
  before=state;state=http(env+'/step',{'episode':eid,'action':action})
  steps.append({'turn':len(steps),'obs':before['obs'],'obs_fmt':before['obs'],'action':action,'raw_output':f'<action>{action}</action>','state_before':before['state'],'state_after':state['state'],'feedback':state['obs'],'reward':state.get('reward',0),'done':state.get('done',False),'executed':True,'format_error':None,'env_rejection':None,'attempts':[]})
 try:
  take(f'search[{query}]')
  for _ in range(page-1):take('click[next >]')
  if asin.lower() not in state['available_actions']['clickables']:return None
  take(f'click[{asin.lower()}]')
  plan=plan_options(state['obs'],truth.get('goal_options',[]))
  price=re.search(r'Price: \$([0-9,.]+)',state['obs'])
  if plan is None or (price and float(price[1].replace(',',''))>truth['price_upper']):return None
  if len(steps)+len(plan)+1>15:return None
  for value in plan:take(f'click[{value}]')
  take('click[buy now]')
  ep={'goal':goal,'rep':17,'tag':'privileged_goal_oracle_broad_search_v4','mode':'train','model':None,'instruction':truth['instruction_text'],'steps':steps,'env_url':env,'protocol':{'planner':'privileged_goal_oracle_broad_search_v4','max_steps':15,'goal_shuffle_seed':233,'env_seed':42},'complete':True,'done':state['done'],'score':state['reward'],'won':bool(state['done']) and state['reward']>=1-1e-8,'query':query,'candidate_asin':asin,'candidate_index':index,'replay_verified':False}
  if ep['won']:
   replay=http(env+'/replay',{'goal':goal,'actions':[s['action'] for s in steps]})
   assert replay['obs']==[steps[0]['obs']]+[s['feedback'] for s in steps]
   assert replay['dones']==[s['done'] for s in steps]
   assert [s['strict_key'] for s in replay['states'][1:]]==[s['state_after']['strict_key'] for s in steps]
   assert replay['rewards'][-1]>=1-1e-8;ep['replay_verified']=True
  return ep
 finally:http(env+'/close',{'episode':eid})


def normalized(x):return ' '.join(html.unescape(str(x)).lower().split())


def catalog_attempt(goal,candidate,query,page,env,index):
 eid='catalog_oracle_'+uuid.uuid4().hex;state=http(env+'/reset',{'episode':eid,'goal':goal});instruction=state['instruction'];steps=[];reason='not_completed'
 def take(action):
  nonlocal state
  assert len(steps)<15
  before=state;state=http(env+'/step',{'episode':eid,'action':action})
  steps.append({'turn':len(steps),'obs':before['obs'],'obs_fmt':before['obs'],'action':action,'raw_output':f'<action>{action}</action>','state_before':before['state'],'state_after':state['state'],'feedback':state['obs'],'reward':state['reward'],'done':state['done'],'executed':True,'format_error':None,'env_rejection':None,'attempts':[]})
 try:
  take(f'search[{query}]')
  for _ in range(page-1):take('click[next >]')
  if candidate['asin'].lower() not in state['available_actions']['clickables']:return None
  take('click['+candidate['asin'].lower()+']')
  for value in candidate['options'].values():
   names={normalized(v):v for v in state['available_actions']['clickables']}
   if normalized(value) not in names or len(steps)>=14:return None
   take('click['+names[normalized(value)]+']')
  if len(steps)<15:take('click[buy now]')
  ep={'goal':goal,'rep':17,'tag':'privileged_full_catalog_oracle_v5','mode':'train','model':None,'instruction':instruction,'steps':steps,'env_url':env,'protocol':{'planner':'privileged_full_catalog_oracle_v5','max_steps':15,'goal_shuffle_seed':233,'env_seed':42},'complete':True,'done':bool(state['done']),'score':state['reward'] if state['done'] else 0.,'won':bool(state['done']) and state['reward']>=1-1e-8,'candidate_asin':candidate['asin'],'candidate_query':query,'candidate_index':index,'replay_verified':False}
  if ep['won']:
   replay=http(env+'/replay',{'goal':goal,'actions':[s['action'] for s in steps]})
   assert replay['obs']==[steps[0]['obs']]+[s['feedback'] for s in steps]
   assert replay['dones']==[s['done'] for s in steps]
   assert [x['strict_key'] for x in replay['states'][1:]]==[s['state_after']['strict_key'] for s in steps]
   assert replay['rewards'][-1]>=1-1e-8;ep['replay_verified']=True
  return ep
 finally:
  try:http(env+'/close',{'episode':eid})
  except Exception:pass


def stream_array(path):
 decoder=json.JSONDecoder();buffer='';pos=0;ended=False
 with path.open() as f:
  while not ended:
   if pos>=len(buffer):buffer=f.read(4*1024*1024);pos=0
   if not buffer:break
   while pos<len(buffer) and (buffer[pos].isspace() or buffer[pos] in '[,'):pos+=1
   if pos<len(buffer) and buffer[pos]==']':ended=True;break
   try:obj,end=decoder.raw_decode(buffer,pos)
   except json.JSONDecodeError:
    extra=f.read(4*1024*1024)
    if not extra:raise
    buffer=buffer[pos:]+extra;pos=0;continue
   yield obj;pos=end
   if pos>4*1024*1024:buffer=buffer[pos:];pos=0
 assert ended,'Catalog did not end as a complete JSON array'


@functools.lru_cache(maxsize=300000)
def masks(value):
 value=normalize_color(value)
 return tuple(sum(1<<j for j,option in enumerate(goal['goal_options']) if fuzz.token_set_ratio(value,normalize_color(option))>85) for goal in GOALS)


def option_plan(options,index,n):
 full=(1<<n)-1;groups=[];union=0
 for key,values in options.items():
  choices={}
  for value in values:
   mask=masks(value)[index]
   if mask:choices.setdefault(mask,value);union|=mask
  if choices:groups.append((key,choices))
 if union!=full:return None
 states={0:[]}
 for key,choices in groups:
  updated=dict(states)
  for old,plan in states.items():
   for mask,value in choices.items():
    new=old|mask
    if new not in updated:updated[new]=plan+[(key,value)]
  states=updated
 return states.get(full)


def optimistic_batch(batch):
 counts=[0]*len(GOALS);candidates=[];errors=0
 for raw in batch:
  pricing=raw.get('pricing')
  try:lower=min([float(re.sub(r'[^\d.]','',p)) for p in pricing.split('$')[1:]][:2]) if pricing else 100.
  except Exception:lower=0.  # Conservative bound; never exclude on a parsing uncertainty.
  options={}
  for key,values in (raw.get('customization_options') or {}).items():
   if values is not None:options[key.lower()]=[str(v['value']).strip().replace('/',' | ').lower() for v in values]
  for index,goal in enumerate(GOALS):
   if lower>goal['price_upper']:continue
   plan=option_plan(options,index,len(goal['goal_options']))
   if plan is None:continue
   product={**raw,'Title':raw['name'],'Description':raw.get('full_description'),'BulletPoints':raw['small_description'] if isinstance(raw.get('small_description'),list) else [raw.get('small_description')],'Attributes':ATTRS.get(raw['asin'],{}).get('attributes',['DUMMY_ATTR']),'query':raw['query'].lower().strip()}
   try:reward=get_reward(product,goal,lower,dict(plan))
   except Exception:errors+=1;continue
   if reward>=1-1e-9:
    counts[index]+=1;candidates.append({'goal':goal['goal'],'asin':raw['asin'],'name':raw['name'],'query':product['query'],'options':dict(plan),'optimistic_price':lower,'not_environment_verified':True})
 return counts,candidates,errors


def exact_batch(batch):
 counts=[0]*len(GOALS);candidates=[];errors=0
 for raw in batch:
  pricing=raw.get('pricing')
  try:lower=min([float(re.sub(r'[^\d.]','',p)) for p in pricing.split('$')[1:]][:2]) if pricing else 100.
  except Exception:lower=0.
  lower=raw['_actual_price']
  options={}
  for key,values in (raw.get('customization_options') or {}).items():
   if values is not None:options[key.lower()]=[str(v['value']).strip().replace('/',' | ').lower() for v in values]
  owners={value:key for key,values in options.items() for value in values}
  options={key:[value for value in values if owners[value]==key] for key,values in options.items()}
  for index,goal in enumerate(GOALS):
   if lower>goal['price_upper']:continue
   plan=option_plan(options,index,len(goal['goal_options']))
   if plan is None:continue
   product={**raw,'Title':raw['name'],'Description':raw.get('full_description'),'BulletPoints':raw['small_description'] if isinstance(raw.get('small_description'),list) else [raw.get('small_description')],'Attributes':ATTRS.get(raw['asin'],{}).get('attributes',['DUMMY_ATTR']),'query':raw['query'].lower().strip()}
   try:reward=get_reward(product,goal,lower,dict(plan))
   except Exception:errors+=1;continue
   if reward>=1-1e-9:
    counts[index]+=1;candidates.append({'goal':goal['goal'],'asin':raw['asin'],'name':raw['name'],'query':product['query'],'options':dict(plan),'actual_seed42_price':lower,'not_environment_verified':True})
 return counts,candidates,errors


def query_filter_attempt(goal, candidates, query, env):
    eligible = {c['asin'].lower():c for c in candidates}
    eid = 'oracle_query_filter_' + uuid.uuid4().hex
    state = http(env+'/reset', {'episode':eid, 'goal':goal})
    instruction, steps = state['instruction'], []
    def take(action):
        nonlocal state
        if len(steps) >= 15:
            raise ValueError('Oracle action budget exceeded')
        before = state
        state = http(env+'/step', {'episode':eid, 'action':action})
        steps.append({'turn':len(steps), 'obs':before['obs'], 'obs_fmt':before['obs'],
            'action':action, 'raw_output':f'<action>{action}</action>',
            'state_before':before['state'], 'state_after':state['state'],
            'feedback':state['obs'], 'reward':state['reward'], 'done':state['done'],
            'executed':True, 'format_error':None, 'env_rejection':None, 'attempts':[]})
    try:
        take('search[<q> '+query+']')
        selected = None
        for page in range(1, 12):
            for name in state['available_actions']['clickables']:
                if name in eligible:
                    selected = eligible[name]
                    break
            if selected or page == 11 or 'next >' not in state['available_actions']['clickables']:
                break
            take('click[next >]')
        if selected and len(steps) + len(selected['options']) + 2 <= 15:
            take('click['+selected['asin'].lower()+']')
            for value in selected['options'].values():
                take('click['+value+']')
            take('click[buy now]')
        return {'goal':goal, 'rep':17, 'tag':'privileged_catalog_query_filter_oracle_v6',
            'mode':'train', 'model':None, 'instruction':instruction, 'steps':steps, 'env_url':env,
            'protocol':{'max_steps':15, 'env_seed':42, 'goal_shuffle_seed':233},
            'complete':True, 'done':bool(state.get('done')), 'score':state.get('reward',0) if state.get('done') else 0.,
            'won':bool(state.get('done')) and state.get('reward',0)>=1-1e-8, 'replay_verified':False}
    finally:
        http(env+'/close', {'episode':eid})


def catalog_candidates(webshop_root, truths, exact=False):
    global GOALS, ATTRS
    GOALS = truths
    ATTRS = json.loads((Path(webshop_root)/'data/items_ins_v2.json').read_text())
    masks.cache_clear()
    rng, seen, batch = random.Random(42), set(), []
    results = {g['goal']:[] for g in truths}
    def consume():
        _, candidates, errors = (exact_batch if exact else optimistic_batch)(batch)
        if errors:
            raise ValueError(f'Official WebShop reward failed for {errors} candidates')
        for candidate in candidates:
            if len(results[candidate['goal']]) < 10000:
                results[candidate['goal']].append(candidate)
    for raw in stream_array(Path(webshop_root)/'data/items_shuffle.json'):
        asin = raw['asin']
        if asin == 'nan' or len(asin)>10 or asin in seen:
            continue
        seen.add(asin)
        if exact:
            text = raw.get('pricing')
            prices = [float(re.sub(r'[^\d.]','',x)) for x in text.split('$')[1:]][:2] if text else [100.]
            raw['_actual_price'] = prices[0] if len(prices)==1 else rng.uniform(*prices)
        batch.append(raw)
        if len(batch) == 256:
            consume()
            batch = []
    if batch:
        consume()
    masks.cache_clear()
    return results, len(seen)
