"""Collect fixed-budget ALFWorld teacher sources with full success/failure records."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
from gcopd.alfworld.environment import Environment
from gcopd.alfworld.tasks import load_tasks, resolve_gamefile
import re
import subprocess
import time
import traceback

from gcopd.alfworld.protocol import build_messages, format_history
from gcopd.alfworld.protocol import parse_action, INVALID_FEEDBACK, INVALID_ACTION
ROOT = Path(__file__).resolve().parent

def write_json(path, value):
    tmp = path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); tmp.replace(path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True); p.add_argument('--run-dir', required=True)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shards', type=int, default=1)
    p.add_argument('--tasks',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--env-python',required=True);p.add_argument('--reps',type=int,default=16)
    p.add_argument('--tp', type=int, default=1)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--memory-utilization', type=float, default=.9)
    args = p.parse_args()
    run = Path(args.run_dir); outdir = run/'episodes'; outdir.mkdir(parents=True, exist_ok=True)
    shard = run/f'shard_{args.shard}'; shard.mkdir(exist_ok=True)
    if not 0 <= args.shard < args.shards or args.reps < 1 or args.batch_size < 1:
        raise ValueError('Invalid shard, repetition or batch configuration')
    manifest = load_tasks(args.tasks)
    if any(t['split'] != 'train' for t in manifest):
        raise ValueError('Teacher-source collection requires training tasks')
    for task in manifest:
        resolve_gamefile(args.data_root,task['gamefile'])
    all_tasks = [dict(t,rep=rep) for t in manifest for rep in range(args.reps)]
    signature = {'tasks':all_tasks,'model':str(Path(args.model).resolve()),'reps':args.reps}
    receipt = run/'COLLECTION.json'
    if receipt.exists() and json.loads(receipt.read_text()) != signature:
        raise ValueError('Collection directory belongs to different tasks, model or repetitions')
    write_json(receipt,signature)
    assigned = all_tasks[args.shard::args.shards]
    tasks = []
    for task in assigned:
        target = outdir/f"task_{task['task_index']:04d}_rep_{task['rep']:02d}.json"
        if target.exists():
            existing = json.loads(target.read_text())
            assert existing['gamefile'] == task['gamefile'] and existing['rep'] == task['rep']
            assert not existing['technical_incomplete']
        else:
            tasks.append(task)
    if not tasks:
        write_json(shard/'COMPLETE.json', {'assigned_episodes':len(assigned),'new_episodes':0,'time':time.time()})
        return
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, distributed_executor_backend='mp',
        dtype='bfloat16', max_model_len=10752, gpu_memory_utilization=args.memory_utilization,
        max_num_seqs=args.batch_size, max_num_batched_tokens=4096, enable_chunked_prefill=True, enable_prefix_caching=False,
        enforce_eager=True, seed=42, disable_log_stats=True)
    tokenizer = llm.get_tokenizer()
    write_json(shard/'MODEL_READY.json', {'time':time.time(), 'model':args.model})
    active=[]
    try:
        def initialize(task):
            env = Environment(task['gamefile'], shard/f"env_{task['task_index']}_rep_{task['rep']}.log",env_python=args.env_python,data_root=args.data_root)
            try:
                state = env.call(op='reset', seed=1000+task['task_index'])
            except BaseException:
                env.close();raise
            match = re.search(r'Your task is to:\s*([^\n]+)', state['feedback'])
            if not match: env.close(); raise RuntimeError('Missing task')
            initial = (state['feedback'][:match.start()]+state['feedback'][match.end():]).strip()
            return {'task':task,'env':env,'state':state,'observation':initial,
                'instruction':match.group(1).strip(),'history':[],'steps':[],
                'initial_observation':state['feedback'],'initial_private_state':state['private_state']}
        for batch_start in range(0, len(tasks), args.batch_size):
            task_batch = tasks[batch_start:batch_start+args.batch_size]
            active = []
            for task in task_batch:
                active.append(initialize(task))
            for turn in range(30):
                requests=[];params=[];prepared=[]
                for ep in active:
                    user,messages=build_messages(ep['instruction'],ep['observation'],
                        ep['state']['admissible_commands'],ep['history'])
                    ids=tokenizer.apply_chat_template(messages,tokenize=True,
                        add_generation_prompt=True,enable_thinking=False,return_dict=False)
                    assert isinstance(ids, list) and all(isinstance(x, int) for x in ids), 'Expected integer prompt token IDs'
                    if len(ids)>10240: raise RuntimeError(f'Prompt overflow: {len(ids)}; no silent truncation')
                    seed=int(hashlib.sha256(f"{ep['task']['gamefile']}|{ep['task']['rep']}|{turn}|0".encode()).hexdigest()[:8],16)%2**31
                    requests.append({'prompt_token_ids':ids})
                    params.append(SamplingParams(temperature=.4,top_p=1.,top_k=-1,max_tokens=512,
                        seed=seed,repetition_penalty=1.,presence_penalty=0.,frequency_penalty=0.))
                    prepared.append((user,ids,seed))
                if not active: break
                results=llm.generate(requests,params,use_tqdm=False)
                remaining=[]
                for ep,result,(user,ids,seed) in zip(active,results,prepared):
                    generated=result.outputs[0];raw=generated.text
                    action,error=parse_action(raw);before=ep['observation'];private_before=ep['state']['private_state']
                    if error:
                        feedback=INVALID_FEEDBACK;history_action=INVALID_ACTION;executed=False
                    else:
                        ep['state']=ep['env'].call(op='step',action=action)
                        feedback=ep['state']['feedback'];history_action=action;executed=True
                    row={'decision':turn+1,'observation_before':before,'user_text':user,
                        'prompt_ids':ids,'response_ids':list(generated.token_ids),'raw_output':raw,
                        'action':action,'parse_error':error,'environment_action_executed':executed,
                        'physical_state_before':private_before,'physical_state_after':ep['state']['private_state'],'feedback':feedback,'won':ep['state']['won'],'done':ep['state']['done'],
                        'seed':seed,'finish_reason':generated.finish_reason,'stop_reason':generated.stop_reason,
                        'prompt_tokens':len(ids),'response_tokens':len(generated.token_ids),'truncated_tokens':0}
                    ep['steps'].append(row)
                    ep['history'].append(format_history(before,turn+1,history_action))
                    ep['observation']=feedback
                    done=ep['state']['won'] or ep['state']['done'] or turn==29
                    if done:
                        record={**ep['task'],'model':args.model,'protocol':'tcod_single_user_h5_thought_v1',
                            'won':ep['state']['won'],'steps':ep['steps'],'technical_incomplete':False,
                            'initial_observation':ep['initial_observation'],'initial_private_state':ep['initial_private_state'],'source_role':'train_k16_not_evaluation',
                            'reason':'success' if ep['state']['won'] else 'environment_done' if ep['state']['done'] else 'max_decisions'}
                        write_json(outdir/f"task_{ep['task']['task_index']:04d}_rep_{ep['task']['rep']:02d}.json",record)
                        ep['env'].close()
                    else: remaining.append(ep)
                active=remaining
                write_json(shard/'PROGRESS.json',{'batch_start':batch_start,'tasks_assigned':len(assigned),'tasks_pending_at_start':len(tasks),'turn':turn+1,'active':len(active),'time':time.time()})
                print(json.dumps({'shard':args.shard,'turn':turn+1,'active':len(active)}),flush=True)
        write_json(shard/'COMPLETE.json',{'assigned_episodes':len(assigned),'new_episodes':len(tasks),'time':time.time()})
    finally:
        for ep in active: ep['env'].close()

if __name__=='__main__':
    try: main()
    except BaseException:
        print(traceback.format_exc(),flush=True)
        raise
