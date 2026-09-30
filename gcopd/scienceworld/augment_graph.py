"""Execute the built-in planner and merge new actual replay into the source graph."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from gcopd.common.io import task_ids, new_output, write_json, digest
from gcopd.common.cleanup import close_owned_environment
from gcopd.scienceworld.collection.planner import execute_saved_plan
from gcopd.scienceworld.collection.replay import equivalent
from gcopd.scienceworld.graph import build_graph, save_graph
import gcopd.scienceworld.state as codec


def planner_record(gamefile,directory):
    """Generate once, execute at most 200 actions, then perform the production replay."""
    from scienceworld import ScienceWorldEnv
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('_JAVA_OPTIONS','-XX:ActiveProcessorCount=2 -Xms64m -Xmx768m')
    env=ScienceWorldEnv('',envStepLimit=200)
    try:
        task,var,simp=gamefile.split('|')
        env.load(task,int(var),simp,generateGoldPath=True)
        plan=list(env.get_gold_action_sequence())
        if not plan or any(not isinstance(x,str) or not x.strip() or x.startswith('ERROR') for x in plan):
            plan=[]
        obs,info=env.reset()
        initial=codec.capture_state(env,None,obs,info['score'])
        primary={'planned_actions':plan,'initial_observation':obs,'initial_physical_key':initial['key'],'first_success_decision':None,'steps':[]}
        for i,action in enumerate(plan[:200]):
            before,tick=obs,env.get_num_moves();obs,_,done,info=env.step(action)
            primary['steps'].append({'action':action,'observation_before':before,'feedback':obs,'score':info['score'],'done':bool(done),'tick_before':tick,'tick_after':env.get_num_moves()})
            if info['score']>=100:primary['first_success_decision']=i+1;break
            if done:break
        write_json(directory/'PLANNER_GENERATION.json',primary)
        # The library receives the new actual execution, even if it differs from generation.
        snapshots=[]
        def snapshot(state,kind):
            snapshots.append({**state,'snapshot_kind':kind})
            return {k:v for k,v in state.items() if k!='objects'}
        record=execute_saved_plan(env,gamefile,primary,snapshot,codec,equivalent)
        write_json(directory/'PLANNER_SNAPSHOTS.json',snapshots)
        raw={k:v for k,v in record.items() if k not in ('states','attempt_branches','replay_failures')}
        source=directory/'PLANNER_SOURCE.json'
        write_json(source,{'gamefile':gamefile,'split':'train','source_kind':'oracle_planner_new_replay','runs':[raw]})
        record.update(source_path=str(source),source_sha256=digest(source))
        write_json(directory/'PLANNER_RECORD.json',record)
        return record
    finally:
        cleanup=close_owned_environment(env)
        write_json(directory/'CLEANUP.json',cleanup)
        if cleanup.get('jvm_exited') is False:raise RuntimeError('Owned planner JVM did not exit')


def augment_one(job):
    gf,oldcase,newcase=job
    records=json.loads((Path(oldcase)/'REPLAY_AUDIT.json').read_text())['records']
    planner=planner_record(gf,newcase);records.append(planner)
    write_json(Path(newcase)/'REPLAY_AUDIT.json',{'gamefile':gf,'records':records})
    graph=build_graph([r for r in records if r['states']],gf)
    saved=save_graph(Path(newcase)/'graph.json.gz',graph)
    return gf,planner,saved


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',required=True);p.add_argument('--train-tasks',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();old=Path(a.catalog).resolve();out=new_output(a.output)
    ids=task_ids(a.train_tasks);manifest=json.loads((old/'MANIFEST.json').read_text())
    if set(ids)!={x['gamefile'] for x in manifest['tasks']}:raise ValueError('Catalog and train task sets differ')
    (out/'entries').mkdir();(out/'cases').mkdir()
    jobs=[]
    for gf in ids:
        h=hashlib.sha256(gf.encode()).hexdigest()[:16]
        jobs.append((gf,str(old/'cases'/h),str(out/'cases'/h)))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for gf,record,saved in pool.map(augment_one,jobs):
            task=next(t for t in manifest['tasks'] if t['gamefile']==gf)
            task['source_files'].append({'rep':1000000,'path':record['source_path'],'sha256':record['source_sha256'],'source_won':record['source_won'],'source_kind':record['source_kind'],'expected_missing':False})
            if record['source_won']:task['source_success_reps'].append(1000000)
            h=hashlib.sha256(gf.encode()).hexdigest()[:16]
            write_json(out/'entries'/(h+'.json'),{'gamefile':gf,'status':'complete','graph':saved})
            print(json.dumps({'task':gf,'planner_success':record['won'],'planner_decisions':len(record['steps'])}),flush=True)
    manifest['planner_augmentation']=True
    write_json(out/'MANIFEST.json',manifest)

if __name__=='__main__':main()
