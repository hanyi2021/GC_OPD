"""Collect K=16 source executions with accepted-action history and retries."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from gcopd.common.io import task_ids, new_output, write_json, digest
import gcopd.scienceworld.collection.episode as episode

def collect_one(job,output,config):
    episode.P=Path(output);episode.CFG=config
    from scienceworld import ScienceWorldEnv
    env=ScienceWorldEnv('',envStepLimit=config['env_step_limit'])
    try:
        record=episode.run_episode(env,job)
        return {'task':job[2],'rep':job[3],'won':record['won'],'accepted_actions':record['length']}
    finally:
        env.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-tasks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--endpoint',required=True,help='OpenAI-compatible teacher server URL ending in /v1')
    p.add_argument('--served-model',required=True);p.add_argument('--model-path',required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--resume',action='store_true')
    a=p.parse_args();ids=task_ids(a.train_tasks)
    cfg={'model':a.served_model,'model_path':str(Path(a.model_path).expanduser().resolve()),'model_id':'Qwen/Qwen3-32B',
         'endpoint':a.endpoint.rstrip('/'),'horizon':30,'env_step_limit':200,'temperature':0.4,'top_p':1.0,'top_k':20,
         'max_tokens':512,'max_prompt':10240,'k':16,'max_step_attempts':5,
         'history':'observations_and_actions_only','retry_environment_rejections':True,
         'train_task_manifest_sha256':digest(a.train_tasks)}
    out=Path(a.output).expanduser().resolve()
    if a.resume:
        if json.loads((out/'COLLECTION_CONFIG.json').read_text())!=cfg:
            raise ValueError('Resume requires the exact same collection configuration')
    else:out=new_output(out)
    for name in ['results','checkpoints']:(out/name).mkdir(exist_ok=True)
    write_json(out/'COLLECTION_CONFIG.json',cfg)
    jobs=[]
    for gf in ids:
        h=hashlib.sha256(gf.encode()).hexdigest()[:16]
        for rep in range(16):
            uid=f'train_{h}_r{rep:02d}'
            if not (out/'results'/f'{uid}.json').exists():jobs.append((uid,'train',gf,rep))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        fs=[pool.submit(collect_one,j,str(out),cfg) for j in jobs]
        for f in as_completed(fs):print(json.dumps(f.result()),flush=True)
    print('Source collection complete. Replay all records before admitting graph success paths.')

if __name__=='__main__':main()
