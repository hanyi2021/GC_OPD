"""Replay all supplied training sources and construct the native graph catalog."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
from gcopd.common.io import task_ids, new_output, write_json, digest
from gcopd.scienceworld.collection.replay import collect_rep
from gcopd.scienceworld.graph import build_graph, save_graph

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-tasks',required=True);p.add_argument('--source-results',required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--runtime-deps',default='',help='Optional existing directory containing scienceworld/py4j; normally use installed dependencies')
    a=p.parse_args();ids=task_ids(a.train_tasks);out=new_output(a.output)
    (out/'entries').mkdir();(out/'cases').mkdir();source=Path(a.source_results).expanduser().resolve()
    manifest={'tasks':[],'fixed_reps':list(range(16)),'train_task_manifest_sha256':digest(a.train_tasks)}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for gf in ids:
            h=hashlib.sha256(gf.encode()).hexdigest()[:16];case=out/'cases'/h;case.mkdir()
            records=[f.result() for f in as_completed([pool.submit(collect_rep,gf,r,str(case),str(source),a.runtime_deps) for r in range(16)])]
            records.sort(key=lambda r:r['rep']);write_json(case/'REPLAY_AUDIT.json',{'gamefile':gf,'records':records})
            graph=build_graph([r for r in records if r['states']],gf)
            saved=save_graph(case/'graph.json.gz',graph)
            write_json(out/'entries'/f'{h}.json',{'gamefile':gf,'status':'complete' if all(r.get('trusted_complete') for r in records) else 'partial','graph':saved})
            files=[]
            for r in records:
                files.append({'rep':r['rep'],'path':r['source_path'],'sha256':r.get('source_sha256'),
                              'source_won':r.get('source_won',False),'expected_missing':r.get('source_missing',False)})
            manifest['tasks'].append({'gamefile':gf,'source_files':files,
                'source_success_reps':[r['rep'] for r in records if r.get('source_won')]})
            write_json(out/'MANIFEST.json',manifest)
            print(json.dumps({'task':gf,**graph['stats']}),flush=True)
    print('Catalog complete; tasks without trusted successes remain included.')

if __name__=='__main__':main()
