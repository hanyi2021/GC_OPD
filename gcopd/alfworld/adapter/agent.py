"""ALFWorld native OPD rollout: shared single-user H5; every sampled token retained."""
import asyncio, hashlib, json, os, re, time, uuid, traceback
from pathlib import Path
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from gcopd.alfworld.protocol import build_messages, format_history
from gcopd.alfworld.protocol import parse_action, INVALID_FEEDBACK, INVALID_ACTION
from gcopd.training.context import finalize_teacher_prompts
from gcopd.alfworld.tasks import resolve_gamefile
from gcopd.common.io import ROOT
class ALFWorldLoop(AgentLoopBase):
 async def run(self,sampling_params,**kwargs):
  extra=kwargs.get('extra_info',{});trace=kwargs.get('_sw_trajectory_info',{})
  rep=int(extra.get('eval_rep',trace.get('rollout_n',0)));version=int(trace.get('step',1))-1
  is_eval=bool(trace.get('validate',False));gamefile=extra['gamefile']
  pad=self.tokenizer.pad_token_id;eos=self.tokenizer.eos_token_id
  if extra.get('padding',False):
   return AgentLoopOutput(prompt_ids=[pad],response_ids=[eos],response_mask=[0],response_logprobs=[0.],reward_score=0.,metrics=AgentLoopMetrics(),extra_fields={'sw_turns':[],'sw_padding':True,'sw_gamefile':gamefile,'sw_episode_file':None})
  dest=Path(os.environ['SW_RUN_DIR'])/'episodes';dest.mkdir(parents=True,exist_ok=True)
  target=dest/(uuid.uuid4().hex+'.json');turns=[];history=[];proc=None;started=time.time();generation=0.
  error_log=target.with_suffix('.env.log').open('w')
  async def rpc(req):
   proc.stdin.write((json.dumps(req)+'\n').encode());await proc.stdin.drain()
   line=await asyncio.wait_for(proc.stdout.readline(),120)
   if not line:raise RuntimeError('ALFWorld environment exited')
   state=json.loads(line)
   if 'error' in state:raise RuntimeError(state['error'])
   return state
  try:
   proc=await asyncio.create_subprocess_exec(os.environ['ALF_ENV_PYTHON'],str(ROOT/'gcopd/alfworld/state_worker.py'),str(resolve_gamefile(os.environ['ALF_DATA_ROOT'],gamefile)),gamefile,stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=error_log,limit=8*1024*1024)
   state=await rpc({'op':'reset','seed':1000+int(extra['task_index'])});raw_initial=state['feedback'];initial_state=state['private_state']
   match=re.search(r'Your task is to:\s*([^\n]+)',raw_initial)
   if not match:raise ValueError('Missing ALFWorld task')
   task=match.group(1).strip();observation=(raw_initial[:match.start()]+raw_initial[match.end():]).strip()
   limit=int(os.environ.get('ALF_MAX_DECISIONS','30'))
   for turn in range(limit):
    before=observation;private_before=state['private_state']
    user,messages=build_messages(task,observation,state['admissible_commands'],history)
    prompt=self.tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=False)
    if len(prompt)>10240:raise ValueError(f'Prompt overflow {len(prompt)}; no silent truncation')
    seed_text=f'{gamefile}|{rep}|{turn}|0' if is_eval else f'{gamefile}|{rep}|{turn}|{version}|42'
    params=dict(sampling_params);params.update(max_tokens=512,seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8],16)%2**31)
    begin=time.time();out=await self.server_manager.generate(request_id=uuid.uuid4().hex,prompt_ids=prompt,sampling_params=params);generation+=time.time()-begin
    ids=list(out.token_ids);logs=list(out.log_probs);assert 0<len(ids)<=512 and len(ids)==len(logs)
    raw=self.tokenizer.decode(ids,skip_special_tokens=True);action,error=parse_action(raw)
    if error:observation=INVALID_FEEDBACK;ha=INVALID_ACTION
    else:state=await rpc({'op':'step','action':action});observation=state['feedback'];ha=action
    history.append(format_history(before,turn+1,ha))
    turns.append({'student_step':turn,'decision':turn+1,'current_obs':before,'observation_before':before,'user_text':user,'prompt_ids':list(prompt),'response_ids':ids,'rollout_logprobs':logs,'raw_output':raw,'action':action,'semantic_action':action,'semantic_action_known':action is not None,'feedback':observation,'format_error':error,'parse_error':error,'env_rejection':observation.strip().lower()=='nothing happens.','environment_action_executed':not bool(error),'score':float(state['won']),'won':state['won'],'done':state['done'],'physical_state_before':private_before,'physical_state_after':state['private_state'],'seed':params['seed'],'requested_sampling_params':params})
    if state['won'] or state['done']:break
   record={'gamefile':gamefile,'task_index':extra['task_index'],'split':extra['split'],'rep':rep,'policy_version':version,'model_path':self.config.actor_rollout_ref.model.path,'is_evaluation':is_eval,'sample_index':trace.get('sample_index'),'won':state['won'],'steps':turns,'initial_obs':raw_initial,'initial_private_state':initial_state,'task_description':task,'final_score':float(state['won']),'final_outcome':{'won':state['won'],'score':float(state['won']),'done':state['done'],'decision_count':len(turns),'decision_limit_reached':len(turns)>=limit},'technical_incomplete':False,'teacher_reference_timing':'after_complete_student_trajectory','student_failure_rollout_retained_for_opd':True,'seconds':time.time()-started}
   if is_eval:
    for row in turns:row['teacher_prompt_ids']=list(row['prompt_ids']);row['graph_reference']={'mode':'evaluation_no_reference'}
   else:await asyncio.to_thread(finalize_teacher_prompts,self.tokenizer,record,environment="alfworld")
   temp=target.with_suffix('.tmp');temp.write_text(json.dumps(record,ensure_ascii=False));temp.replace(target)
   last=turns[-1]
   return AgentLoopOutput(prompt_ids=last['prompt_ids'],response_ids=last['response_ids'],response_mask=[1]*len(last['response_ids']),response_logprobs=last['rollout_logprobs'],reward_score=float(state['won']),num_turns=len(turns),metrics=AgentLoopMetrics(generate_sequences=generation),extra_fields={'sw_turns':turns,'sw_padding':False,'sw_gamefile':gamefile,'sw_episode_file':str(target)})
  except BaseException:
   target.write_text(json.dumps({'gamefile':gamefile,'rep':rep,'policy_version':version,'steps':turns,'technical_incomplete':True,'error':traceback.format_exc()}));raise
  finally:
   if proc is not None and proc.returncode is None:
    proc.terminate()
    try:await asyncio.wait_for(proc.wait(),10)
    except asyncio.TimeoutError:proc.kill();await proc.wait()
   error_log.close()
