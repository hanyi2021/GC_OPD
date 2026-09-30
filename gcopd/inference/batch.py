from gcopd.common.io import ROOT
"""Evaluate one fixed ALFWorld/WebShop model; no checkpoint selection."""
import argparse
import copy
import json
from pathlib import Path
import statistics
from gcopd.inference.episode import run_episode
from gcopd.inference.generate import load_engine


def summarize(records, expected, reps):
    groups={}
    for row in records:
        key=(row['task'],row['rep'])
        if key in groups or row.get('technical_incomplete'):
            raise ValueError('Duplicate or incomplete episode')
        if row.get('backend') not in ('transformers','vllm'):
            raise ValueError('Missing inference backend')
        groups[key]=row
    if set(groups)!={(t,r) for t in expected for r in reps}:
        raise ValueError('Evaluation does not cover the exact requested tasks and repetitions')
    if len({x['model'] for x in records})!=1 or len({x['backend'] for x in records})!=1:
        raise ValueError('Evaluation mixed checkpoints or inference backends')
    per_seed=[]
    for rep in reps:
        rows=[groups[(task,rep)] for task in expected]
        per_seed.append({'seed':rep,'tasks':len(rows),'success_rate':100*statistics.mean(bool(x['won']) for x in rows),
                         'score':100*statistics.mean(x['score'] for x in rows),'rounds':statistics.mean(x['decisions'] for x in rows)})
    metrics={k:{'mean':statistics.mean(x[k] for x in per_seed),
                'sample_std':statistics.stdev(x[k] for x in per_seed) if len(per_seed)>1 else None}
             for k in ['success_rate','score','rounds']}
    return {'model':records[0]['model'],'backend':records[0]['backend'],'per_seed':per_seed,'metrics':metrics}


def main(environment=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',choices=['alfworld','webshop'],required=environment is None,default=environment)
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--backend',choices=['transformers','vllm'],default='transformers')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--data-root');p.add_argument('--env-python');p.add_argument('--env-url')
    p.add_argument('--tasks',help='ALF task indices or WebShop goal IDs as a JSON list; no automatic split creation')
    p.add_argument('--reps',nargs='+',type=int,default=[0,1,2,3]);p.add_argument('--gpu-memory-utilization',type=float,default=0.5)
    a=p.parse_args()
    if environment and a.environment!=environment:raise ValueError("Environment does not match this entrypoint")
    if not a.reps or any(x not in range(4) for x in a.reps) or len(set(a.reps))!=len(a.reps):raise ValueError('Use distinct seeds from 0,1,2,3')
    all_alf=json.loads((ROOT/'gcopd/alfworld/assets/evaluation_tasks.json').read_text())['tasks']
    indices=json.loads(Path(a.tasks).read_text()) if a.tasks else (list(range(len(all_alf))) if a.environment=='alfworld' else list(range(500)))
    if not indices or any(type(x) is not int or x<0 for x in indices) or len(set(indices))!=len(indices):raise ValueError('Task IDs must be unique nonnegative integers')
    if a.environment=='alfworld':
        lookup={x['task_index']:x for x in all_alf}
        if not set(indices)<=set(lookup):raise ValueError('Unknown ALFWorld task index')
        expected=[lookup[i]['gamefile'] for i in indices]
    else:expected=indices
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    signature={'environment':a.environment,'model':str(Path(a.model).resolve()),'backend':a.backend,'tasks':expected,'reps':a.reps}
    path=out/'EVALUATION.json'
    if path.exists() and json.loads(path.read_text())!=signature:raise ValueError('Evaluation directory belongs to another model/protocol/task set')
    path.write_text(json.dumps(signature,indent=2))
    engine=None;records=[]
    for i,task in zip(indices,expected):
        for rep in a.reps:
            target=out/'episodes'/f'task_{i}_rep_{rep}.json'
            if target.exists():row=json.loads(target.read_text())
            else:
                if engine is None:engine=load_engine(a.model,a.environment,a.gpu_memory_utilization,a.backend,a.device)
                args=copy.copy(a);args.output=str(target);args.rep=rep;args.task_index=i;args.goal=i;args.max_decisions=None
                row=run_episode(args,engine)
            if row['task']!=task or row['rep']!=rep or row['model']!=signature['model']:raise ValueError('Episode identity mismatch')
            records.append(row)
    if a.environment=='alfworld':
        report={}
        for split in ['valid_seen','valid_unseen']:
            ids=[lookup[i]['gamefile'] for i in indices if lookup[i]['split']==split]
            if ids:report[split]=summarize([x for x in records if x['task'] in set(ids)],ids,a.reps)
    else:report=summarize(records,expected,a.reps)
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2))
    print(out/'SUMMARY.json')


if __name__=='__main__':main()
