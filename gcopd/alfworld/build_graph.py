"""Replay ALFWorld sources and build verified task-local state/visit graphs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from gcopd.alfworld.environment import Environment
from gcopd.alfworld.tasks import load_tasks
from gcopd.alfworld.protocol import INVALID_FEEDBACK


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def replay(path, *, env_python, data_root, output):
    d = json.loads(path.read_text())
    if d.get('technical_incomplete') or d.get('split') != 'train' or d.get('source_role') != 'train_k16_not_evaluation':
        raise ValueError(f'Not a complete training source: {path.name}')
    env = Environment(d['gamefile'], output/(path.stem+'.env.log'), env_python=env_python, data_root=data_root)
    try:
        state = env.call(op='reset', seed=1000+d['task_index'])
        assert state['feedback'] == d['initial_observation'], 'initial feedback mismatch'
        assert state['private_state']['key'] == d['initial_private_state']['key'], 'initial state mismatch'
        steps = []
        for s in d['steps']:
            assert state['private_state']['key'] == s['physical_state_before']['key'], 'pre-state mismatch'
            if s['environment_action_executed']:
                state = env.call(op='step', action=s['action'])
                feedback = state['feedback']
            else:
                assert s['parse_error']
                feedback = INVALID_FEEDBACK
            assert feedback == s['feedback'], 'feedback mismatch'
            assert state['private_state']['key'] == s['physical_state_after']['key'], 'post-state mismatch'
            assert bool(state['won']) == bool(s['won']) and bool(state['done']) == bool(s['done']), 'outcome mismatch'
            steps.append({'observation_before': s['observation_before'], 'action': s['action'],
                          'feedback': feedback, 'key_before': s['physical_state_before']['key'],
                          'key_after': s['physical_state_after']['key'], 'env_step_called': s['environment_action_executed'],
                          'format_error': s['parse_error'], 'rejection': feedback.strip().lower() == 'nothing happens.',
                          'decision_cost': 1})
        assert bool(state['won']) == bool(d['won'])
        return {'gamefile': d['gamefile'], 'task_index': d['task_index'], 'rep': d['rep'],
                'won': d['won'], 'trusted_complete': True, 'trusted_success': bool(d['won']),
                'initial_key': d['initial_private_state']['key'], 'steps': steps,
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'codec': 'alf_pddl_relative_v1'}
    finally:
        env.close()


def build_catalog(gamefile, index, sources):
    sources = sorted(sources, key=lambda x: x['rep'])
    if not sources or [s['rep'] for s in sources] != list(range(len(sources))):
        raise ValueError('Catalog needs exactly one source for every repetition, starting at zero')
    if any(s['gamefile'] != gamefile or not s['trusted_complete'] for s in sources):
        raise ValueError('Catalog sources must match the task and pass replay')
    visits, edges, internal = {}, [], []
    for source in sources:
        for j, step in enumerate(source['steps']):
            key = step['key_before']
            visits.setdefault(key, []).append({'rep': source['rep'], 'index': j})
            target = internal if key == step['key_after'] else edges
            target.append({'source_key': key, 'target_key': step['key_after'], 'rep': source['rep'],
                           'index': j, 'action': step['action'], 'feedback': step['feedback']})
        if source['steps']:
            visits.setdefault(source['steps'][-1]['key_after'], []).append({'rep': source['rep'], 'index': len(source['steps'])})
    return {'gamefile': gamefile, 'task_index': index, 'codec': 'alf_pddl_relative_v1', 'K': len(sources),
            'sources': sources, 'visits': visits, 'edges': edges, 'internal_observations': internal,
            'source_scope': 'training_only', 'trusted': True, 'selection_rule': 'alf_reference_retention_v2'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    p.add_argument('--episodes', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--env-python', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--reps', type=int, default=16)
    p.add_argument('--workers', type=int, default=4)
    a = p.parse_args()
    tasks = load_tasks(a.tasks)
    if a.reps < 1 or a.workers < 1 or any(t['split'] != 'train' for t in tasks):
        raise ValueError('Use training tasks and positive repetitions/workers')
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new or empty catalog directory')
    out.mkdir(parents=True, exist_ok=True)
    logs = out/'replay_logs'
    logs.mkdir()
    manifest = []
    with ThreadPoolExecutor(a.workers) as pool:
        for task in tasks:
            paths = [Path(a.episodes)/f"task_{task['task_index']:04d}_rep_{rep:02d}.json" for rep in range(a.reps)]
            if not all(x.is_file() for x in paths):
                raise ValueError(f"Missing sources for {task['gamefile']}")
            sources = list(pool.map(lambda path: replay(path, env_python=a.env_python, data_root=a.data_root, output=logs), paths))
            catalog = build_catalog(task['gamefile'], task['task_index'], sources)
            name = hashlib.sha256(task['gamefile'].encode()).hexdigest()+'.json'
            write_json(out/name, catalog)
            manifest.append({'gamefile': task['gamefile'], 'task_index': task['task_index'], 'catalog': name,
                             'sha256': hashlib.sha256((out/name).read_bytes()).hexdigest()})
    write_json(out/'MANIFEST.json', {'tasks': manifest, 'K': a.reps, 'codec': 'alf_pddl_relative_v1'})
    write_json(out/'READY.json', {'tasks': len(tasks), 'K': a.reps, 'all_sources_replayed': True})
    print(out/'MANIFEST.json')


if __name__ == '__main__':
    main()
