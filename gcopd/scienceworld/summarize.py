"""Aggregate success, best-progress scores and rounds across inference seeds."""
import argparse
import json
from pathlib import Path
import statistics
from gcopd.common.io import write_json

def summarize(directory):
    directory=Path(directory);manifest=json.loads((directory/'EVALUATION_TASKS.json').read_text())
    expected=set(manifest['task_ids']);records={}
    for file in sorted((directory/'episodes').glob('*.json')):
        r=json.loads(file.read_text());gf=r['gamefile']
        if not r.get('is_evaluation') or r.get('technical_incomplete') or r.get('rep')!=manifest['rep']:
            raise ValueError('Non-evaluation, incomplete or wrong-seed episode encountered')
        if gf not in expected or gf in records:raise ValueError('Unexpected or duplicated evaluation task')
        if r['model_path']!=manifest['model']:raise ValueError('Mixed model checkpoints')
        records[gf]=r
    if set(records)!=expected:raise ValueError(f'Expected {len(expected)} unique tasks, found {len(records)}')
    rec=list(records.values());won=[r for r in rec if r['won']];failed=[r for r in rec if not r['won']]
    length=lambda rs:statistics.mean(len(r['steps']) for r in rs) if rs else None
    return {'seed':manifest['rep'],'tasks':len(rec),'model':manifest['model'],
        'success_rate':100*len(won)/len(rec),
        'best_progress_score':statistics.mean(max([0]+[s['score'] for s in r['steps']]) for r in rec),
        'final_score':statistics.mean(r['final_score'] for r in rec),'rounds':length(rec),
        'success_rounds':length(won),'failure_rounds':length(failed),'task_ids':sorted(expected)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('runs',nargs='+');p.add_argument('--output',required=True);a=p.parse_args()
    rs=[summarize(d) for d in a.runs]
    if len({r['seed'] for r in rs})!=len(rs):raise ValueError('Repeated inference seed')
    if len({r['model'] for r in rs})!=1 or any(r['task_ids']!=rs[0]['task_ids'] for r in rs):raise ValueError('Different models or task sets')
    out={'seed_count':len(rs),'tasks':rs[0]['tasks'],'model':rs[0]['model'],'per_seed':rs,'metrics':{}}
    for key in ['success_rate','best_progress_score','final_score','rounds','success_rounds','failure_rounds']:
        values=[r[key] for r in rs]
        out['metrics'][key]={'mean':statistics.mean(values) if None not in values else None,
                           'sample_std':statistics.stdev(values) if len(values)>1 and None not in values else None}
    write_json(a.output,out);print(json.dumps(out['metrics'],indent=2))

if __name__=='__main__':main()
