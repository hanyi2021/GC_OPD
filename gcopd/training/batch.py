"""Expand episodes with equal episode/token weighting and k1 teacher targets."""
import numpy as np, torch
from tensordict import TensorDict
from verl import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

def expand(batch,dp_multiple):
 turns=batch.non_tensor_batch['sw_turns'];E=len(turns)
 valid=np.array([len(t)>0 for t in turns]);N=int(valid.sum());assert N>0
 rewards=batch.batch['rm_scores'].sum(-1).float().unsqueeze(-1)
 adv,_=compute_grpo_outcome_advantage(rewards,torch.ones_like(rewards),batch.non_tensor_batch['uid'])
 records=[];owners=[]
 for i,episode in enumerate(turns):
  for row in episode:records.append(row);owners.append(i)
 has_teacher=all('teacher_logprobs' in t for t in records) and bool(records)
 if has_teacher:adv.zero_()
 R0=len(records)
 while len(records)%dp_multiple:records.append({'prompt_ids':[151644],'response_ids':[151645],'rollout_logprobs':[0.]});owners.append(-1)
 R=len(records);P=max(len(t['prompt_ids']) for t in records);L=max(len(t['response_ids']) for t in records)
 prompt=torch.full((R,P),151643,dtype=torch.long);response=torch.full((R,L),151643,dtype=torch.long)
 attention=torch.zeros((R,P+L),dtype=torch.long);mask=torch.zeros((R,L),dtype=torch.long)
 advantages=torch.zeros((R,L));old=torch.zeros((R,L));weight=torch.zeros((R,L));scores=torch.zeros((R,L))
 nt=[sum(len(t['response_ids']) for t in episode) for episode in turns]
 for j,(t,i) in enumerate(zip(records,owners)):
  a,b=len(t['prompt_ids']),len(t['response_ids']);prompt[j,-a:]=torch.tensor(t['prompt_ids']);response[j,:b]=torch.tensor(t['response_ids']);attention[j,P-a:P+b]=1
  if i>=0:
   mask[j,:b]=1;old[j,:b]=torch.tensor(t['rollout_logprobs']);advantages[j,:b]=adv[i,0];weight[j,:b]=R/(N*nt[i]);scores[j,b-1]=rewards[i,0]
 ids=torch.cat([prompt,response],-1);pos=(attention.cumsum(-1)-1).clamp_min(0)
 tensors={'prompts':prompt,'responses':response,'input_ids':ids,'attention_mask':attention,'position_ids':pos,'response_mask':mask,'rollout_log_probs':old,'old_log_probs':old.clone(),'advantages':advantages,'returns':advantages.clone(),'sw_token_weight':weight,'rm_scores':scores}
 if has_teacher:
  tids=torch.zeros((R,P+L,1),dtype=torch.int32);tlps=torch.zeros((R,P+L,1),dtype=torch.float32)
  for j,(t,i) in enumerate(zip(records,owners)):
   if i<0:continue
   a,b=len(t['prompt_ids']),len(t['response_ids'])
   tids[j,P-a:P+b]=torch.tensor(t['teacher_ids'],dtype=torch.int32)
   tlps[j,P-a:P+b]=torch.tensor(t['teacher_logprobs'],dtype=torch.float32)
  tensors.update(teacher_ids=tids,teacher_logprobs=tlps)
 sel=np.array([max(i,0) for i in owners]);non={k:v[sel] for k,v in batch.non_tensor_batch.items() if k not in ['sw_turns','sw_episode_file']}
 non['multi_modal_inputs']=np.array([{} for _ in range(R)],dtype=object)
 meta=dict(batch.meta_info);meta.update(sw_expanded=True,sw_episode_count=N,sw_decisions=R0,sw_episode_success_rate=float(rewards[torch.as_tensor(valid)].mean()))
 return DataProto(batch=TensorDict(tensors,batch_size=R),non_tensor_batch=non,meta_info=meta)
