"""Run one full OPD epoch, then one GC epoch from its final checkpoint."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from gcopd.training.stages import training_tasks
import math


def main(environment=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--environment',choices=['alfworld','webshop'],required=environment is None,default=environment)
    p.add_argument('--tasks',required=True);p.add_argument('--output',required=True)
    p.add_argument('--execute',action='store_true')
    a,options=p.parse_known_args()
    if any(x in options for x in ['--phase','--parent']):raise ValueError('The two-stage runner fixes the phase and final OPD parent')
    if '--catalog' not in options:raise ValueError('Provide the replay-verified --catalog before training')
    if a.execute and '--skip-asset-check' in options:raise ValueError('Cannot execute unchecked training assets')
    steps=math.ceil(len(training_tasks(a.environment,a.tasks))/32)
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new or empty output directory')
    out.mkdir(parents=True,exist_ok=True)
    module=f'gcopd.{a.environment}.train';stages=[]
    if environment and a.environment!=environment:raise ValueError('Environment does not match this entrypoint')
    for phase in ['opd','gc']:
        target=out/phase
        cmd=[sys.executable,'-m',module,'prepare','--environment',a.environment,'--phase',phase,
             '--tasks',str(Path(a.tasks).resolve()),'--output',str(target),*options]
        if phase=='gc':cmd+=['--parent',str(out/'opd/checkpoints'/f'global_step_{steps}')]
        stages.extend([cmd,[sys.executable,'-m',module,'execute',str(target/'PREPARED_RUN.json')]])
    plan={'updates_per_phase':steps,'opd_final_step':steps,'gc_final_step':2*steps,
          'parent_rule':'Final checkpoint after one complete OPD epoch','commands':stages,'executed':False}
    path=out/'PIPELINE.json';path.write_text(json.dumps(plan,indent=2))
    if a.execute:
        for i,cmd in enumerate(stages):
            subprocess.run(cmd,check=True)
            plan['completed_commands']=i+1;path.write_text(json.dumps(plan,indent=2))
        plan['executed']=True;path.write_text(json.dumps(plan,indent=2))
    print(path)


if __name__=='__main__':main()
