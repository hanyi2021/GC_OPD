"""Merge replay-verified successful 1–30-step planner executions into ALFWorld graphs."""
import argparse
import copy
import json
from pathlib import Path
from gcopd.alfworld.tasks import load_tasks
from gcopd.alfworld.build_graph import build_catalog
from gcopd.common.io import new_output, write_json, digest


def merge(base, record):
    if base.get('planner_augmented') or base['K'] != 16 or len(base['sources']) != 16:
        raise ValueError('Expected an unaugmented K16 teacher graph')
    if record['gamefile'] != base['gamefile'] or record['task_index'] != base['task_index']:
        raise ValueError('Planner task does not match the graph')
    if record.get('technical_incomplete') or not record.get('trusted_complete'):
        raise ValueError('Planner execution must complete independent replay')
    if record['initial_key'] != base['sources'][0]['initial_key']:
        raise ValueError('Planner and teacher initial states differ')
    sources = copy.deepcopy(base['sources'])
    eligible = record['won'] is True and 1 <= len(record['steps']) <= 30
    if eligible:
        fields = ['observation_before','action','feedback','key_before','key_after',
                  'env_step_called','format_error','rejection','decision_cost']
        sources.append({'gamefile':base['gamefile'], 'task_index':base['task_index'],
            'rep':16, 'source_kind':'planner', 'source_id':'fast_downward',
            'won':True, 'trusted_complete':True, 'trusted_success':True,
            'initial_key':record['initial_key'], 'codec':'alf_pddl_relative_v1',
            'steps':[{k:s[k] for k in fields} for s in record['steps']]})
    result = build_catalog(base['gamefile'], base['task_index'], sources)
    result.update(K=len(sources), teacher_K=16, planner_augmented=True,
        planner_max_success_steps=30, planner_sources=int(eligible),
        planner_excluded_reason=None if eligible else 'not_successful_within_1_to_30_steps')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    p.add_argument('--catalog', required=True)
    p.add_argument('--planner-sources', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    tasks = load_tasks(a.tasks)
    if any(t['split'] != 'train' for t in tasks):
        raise ValueError('Graph augmentation requires training tasks')
    base, source = Path(a.catalog).resolve(), Path(a.planner_sources).resolve()
    manifest = json.loads((base/'MANIFEST.json').read_text())
    indexed = {t['gamefile']:t for t in manifest['tasks']}
    if len(indexed) != len(manifest['tasks']) or set(indexed) != {t['gamefile'] for t in tasks}:
        raise ValueError('Graph and training task sets differ')
    if not (base/'READY.json').is_file() or not (source/'COMPLETE.json').is_file():
        raise ValueError('Complete source collection and base graph construction first')
    complete = json.loads((source/'COMPLETE.json').read_text())
    records = {t['gamefile']:t for t in complete['sources']}
    if len(records) != len(complete['sources']) or set(records) != set(indexed):
        raise ValueError('Planner records do not cover the exact training task set')
    out = new_output(a.output)
    (out/'planner_sources').mkdir()
    entries = []
    for task in tasks:
        entry = indexed[task['gamefile']]
        path = base/entry['catalog']
        if digest(path) != entry['sha256']:
            raise ValueError('Base graph checksum differs')
        src = source/f"task_{task['task_index']:04d}_planner.json"
        if digest(src) != records[task['gamefile']]['sha256']:
            raise ValueError('Planner source checksum differs')
        record = json.loads(src.read_text())
        graph = merge(json.loads(path.read_text()), record)
        copied = out/'planner_sources'/src.name
        copied.write_bytes(src.read_bytes())
        if graph['planner_sources']:
            graph['sources'][-1].update(source_file=str(copied.relative_to(out)), source_sha256=digest(copied))
        graph['planner_attempt_record'] = str(copied.relative_to(out))
        target = out/entry['catalog']
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, graph)
        entries.append({**entry, 'sha256':digest(target), 'K':graph['K'], 'planner_sources':graph['planner_sources']})
    write_json(out/'MANIFEST.json', {'tasks':entries, 'teacher_K':16,
        'planner_augmented':True, 'codec':'alf_pddl_relative_v1'})
    write_json(out/'READY.json', {'tasks':len(tasks), 'teacher_K':16, 'planner_augmented':True,
        'planner_max_success_steps':30, 'all_sources_replayed':True})
    print(out/'MANIFEST.json')


if __name__ == '__main__':
    main()
