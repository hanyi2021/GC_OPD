"""Replay and merge original oracle executions plus successful refinements into K16 graphs."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
from gcopd.common.io import new_output, write_json, digest
from gcopd.webshop.collect import load_goals
from gcopd.webshop.collect_planner import verify_episode
from gcopd.webshop.build_graph import normalized_source, trajectory_from_episode, build_graph, http


def merge_task(task, base, planner_root, records, out, env):
    goal=task['goal'];gamefile=f'goal{goal}';tid=hashlib.sha256(gamefile.encode()).hexdigest()[:16]
    entry=json.loads((base/'entries'/f'{tid}.json').read_text())
    graph_path=base/entry['graph']['path']
    if digest(graph_path)!=entry['graph']['sha256']:
        raise ValueError('Teacher graph checksum differs')
    with gzip.open(graph_path,'rt') as f:original=json.load(f)
    traces=copy.deepcopy(original['trajectories'])
    if [t['rep'] for t in traces]!=list(range(16)) or any(not t['trusted_complete'] for t in traces):
        raise ValueError('Expected a replay-verified, unaugmented K16 graph')
    for tr in traces:
        src=base/tr['source_path']
        if digest(src)!=tr['source_sha256']:
            raise ValueError('Teacher source checksum differs')
        relative=f"sources/goal{goal}_r{tr['rep']:02d}.json"
        (out/relative).write_bytes(src.read_bytes());tr['source_path']=relative
    extra=[]
    for record in records:
        src=planner_root/record['path']
        if digest(src)!=record['sha256']:
            raise ValueError('Oracle source checksum differs')
        ep=json.loads(src.read_text());extra.append(ep)
    if [e['rep'] for e in extra] not in ([16],[16,17]):
        raise ValueError('Keep the original oracle and at most one successful refinement')
    for ep in extra:
        if ep['goal']!=goal or ep['instruction']!=task['instruction'] or (ep['rep']==17 and not ep['won']):
            raise ValueError('Oracle identity/outcome differs from graph')
        if (ep['rep']==16 and ep['tag']!='privileged_goal_oracle_v1') or (ep['rep']==17 and ep['tag']=='privileged_goal_oracle_v1'):
            raise ValueError('Original oracle and refinement source kinds are swapped')
        verify_episode(ep,env)
        if ep['steps'][0]['state_before']['strict_key']!=traces[0]['states'][0]['key']:
            raise ValueError('Oracle and teacher start in different states')
        raw=f"planner_sources/g{goal:05d}_r{ep['rep']:02d}.json"
        write_json(out/raw,ep)
        relative=f"sources/goal{goal}_r{ep['rep']:02d}.json"
        normalized_source(ep,str(out/relative))
        tr=trajectory_from_episode(ep,str(out/relative),env)
        if not tr['trusted_complete']:
            raise ValueError('Oracle failed graph-builder replay')
        tr.update(source_path=relative,raw_episode_path=raw)
        traces.append(tr)
    graph=build_graph(goal,task['instruction'],traces)
    graph['semantics'].update(planner='original_oracle_plus_verified_success_refinement',teacher_K=16)
    relative=f'graphs/goal{goal}.json.gz'
    with gzip.open(out/relative,'wt') as f:json.dump(graph,f,ensure_ascii=False)
    entry.update(graph={'path':relative,'sha256':digest(out/relative)},stats=graph['stats'])
    write_json(out/'entries'/f'{tid}.json',entry)
    return {**task,'source_success_reps':[t['rep'] for t in traces if t['source_won']],
        'source_files':[{'rep':t['rep'],'path':t['source_path'],'sha256':t['source_sha256'],
            'source_won':t['source_won'],'source_kind':t['source_kind'],'expected_missing':False} for t in traces]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',required=True);p.add_argument('--catalog',required=True)
    p.add_argument('--planner-sources',required=True);p.add_argument('--env-url',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();goals=load_goals(a.tasks)
    base,src=Path(a.catalog).resolve(),Path(a.planner_sources).resolve()
    manifest=json.loads((base/'MANIFEST.json').read_text());complete=json.loads((src/'COMPLETE.json').read_text())
    tasks={t['goal']:t for t in manifest['tasks']};records={t['goal']:t['sources'] for t in complete['tasks']}
    if (set(tasks)!=set(goals) or set(records)!=set(goals) or len(tasks)!=len(manifest['tasks']) or
        len(records)!=len(complete['tasks']) or complete['catalog_sha256']!=digest(base/'MANIFEST.json')):
        raise ValueError('Teacher graph, oracle collection and training task identities differ')
    env=a.env_url.rstrip('/');health=http(env+'/health')
    if not health.get('ok') or health.get('seed')!=42 or health.get('goal_shuffle_seed')!=233 or health.get('observation_mode')!='text_rich':
        raise ValueError('Use the matching text_rich service')
    out=new_output(a.output)
    for name in ['entries','graphs','sources','planner_sources']:(out/name).mkdir()
    result=[merge_task(tasks[g],base,src,records[g],out,env) for g in goals]
    write_json(out/'MANIFEST.json',{'schema_version':'webshop_catalog_manifest_v1','tasks':result,
        'K':16,'teacher_K':16,'planner_augmentation':True})
    write_json(out/'READY.json',{'tasks':len(goals),'teacher_K':16,'all_sources_replayed':True,
        'planner_augmentation':True,'manifest_sha256':digest(out/'MANIFEST.json')})
    print(out/'MANIFEST.json')


if __name__=='__main__':main()
