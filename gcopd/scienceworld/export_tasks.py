"""Export the installed benchmark's official task lists without random splitting."""
import argparse
from gcopd.common.io import new_output, write_json
from gcopd.common.cleanup import close_owned_environment

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--simplification',default='easy');a=p.parse_args();out=new_output(a.output)
    from scienceworld import ScienceWorldEnv
    env=ScienceWorldEnv('',envStepLimit=200)
    splits={s:[] for s in ('train','dev','test')}
    try:
        for task in env.get_task_names():
            env.load(task,0,a.simplification)
            for split in splits:
                variations=getattr(env,'get_variations_'+split)()
                splits[split].extend(f'{task}|{v}|{a.simplification}' for v in variations)
    finally:
        close_owned_environment(env)
    for split,ids in splits.items():write_json(out/(split+'_tasks.json'),ids)
    write_json(out/'SPLIT_PROVENANCE.json',{'method':'Installed ScienceWorld official task/variation enumeration; no random split','counts':{k:len(v) for k,v in splits.items()},'task_order':'ScienceWorld API enumeration order'})
    print({k:len(v) for k,v in splits.items()})

if __name__=='__main__':main()
