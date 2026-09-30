"""Freeze a checkpoint using development SR only, with an explicit tie rule."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
from gcopd.common.io import task_ids, write_json, digest

def validate_summary(summary, expected):
    seeds = summary.get('per_seed')
    if not isinstance(seeds, list) or not seeds:
        raise ValueError('Selection requires at least one completed evaluation seed')
    seen = set()
    rates = []
    for seed in seeds:
        seed_id = seed.get('seed')
        if type(seed_id) is not int or seed_id < 0 or seed_id in seen:
            raise ValueError('Selection seeds must be distinct nonnegative integers')
        seen.add(seed_id)
        ids = seed.get('task_ids', [])
        if len(ids) != len(expected) or set(ids) != expected or seed.get('tasks') != len(expected):
            raise ValueError('Selection summary does not match the supplied development task set')
        if seed.get('model') != summary.get('model'):
            raise ValueError('Selection summary mixes model checkpoints')
        rate = seed.get('success_rate')
        if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 <= rate <= 100:
            raise ValueError('Development success rates must be finite percentages')
        rates.append(rate)
    mean = summary['metrics']['success_rate']['mean']
    if type(mean) not in (int, float) or not math.isfinite(mean) or not math.isclose(mean, statistics.mean(rates), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError('Development mean does not match its evaluation seeds')
    return mean

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev-tasks',required=True);p.add_argument('--summaries',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();expected=set(task_ids(a.dev_tasks));candidates=[]
    for file in a.summaries:
        r=json.loads(Path(file).read_text())
        development_sr = validate_summary(r, expected)
        model=Path(r['model']);matches=re.findall(r'global_step_(\d+)',str(model))
        if len(matches)!=1:raise ValueError('Expected exactly one global_step_N in candidate model path')
        candidates.append({'model':str(model),'step':int(matches[0]),'development_sr':development_sr,
            'inference_seeds':[s['seed'] for s in r['per_seed']],'summary_sha256':digest(file)})
    if any(c['inference_seeds']!=candidates[0]['inference_seeds'] for c in candidates):raise ValueError('Candidate selection seed sets differ')
    winner=min(candidates,key=lambda c:(-c['development_sr'],c['step']))
    write_json(a.output,{'selected':winner,'candidates':candidates,'rule':'Highest development SR; earliest step breaks exact ties.',
                         'dev_manifest_sha256':digest(a.dev_tasks),'test_results_used':False})
    print(winner['model'])

if __name__=='__main__':main()
