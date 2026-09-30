"""Execute TextWorld/Fast Downward plans and independently replay training trajectories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import sys

from gcopd.alfworld.tasks import load_tasks, resolve_gamefile
from gcopd.alfworld.state_worker import private_state
from gcopd.common.io import write_json, digest


def execute_task(task, data_root):
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    gamefile = resolve_gamefile(data_root, task['gamefile'])
    def create(planner):
        return textworld.start(str(gamefile), textworld.EnvInfos(
            won=True, lost=True, facts=True, admissible_commands=True,
            policy_commands=planner), wrappers=[AlfredDemangler, AlfredInfos])
    steps = []
    env = create(True)
    try:
        env.seed(1000 + task['task_index'])
        state = env.reset()
        initial = state['feedback']
        initial_state = private_state(state, task['gamefile'])
        match = re.search(r'Your task is to:\s*([^\n]+)', initial)
        observation = (initial[:match.start()] + initial[match.end():]).strip() if match else initial
        reason = 'max_planner_steps'
        for i in range(30):
            if state['won']:
                reason = 'success'
                break
            plan = list(state['policy_commands'] or [])
            if not plan:
                reason = 'planner_no_plan'
                break
            action = plan[0]
            before = private_state(state, task['gamefile'])
            if action not in state['admissible_commands']:
                raise ValueError('Planner proposed an inadmissible command')
            state, _, done = env.step(action)
            after = private_state(state, task['gamefile'])
            steps.append(dict(decision=i+1, observation_before=observation, action=action,
                feedback=state['feedback'], key_before=before['key'], key_after=after['key'],
                env_step_called=True, format_error=None,
                rejection=state['feedback'].strip().lower() == 'nothing happens.', decision_cost=1,
                won=bool(state['won']), done=bool(done), physical_state_before=before,
                physical_state_after=after, remaining_plan_length_before=len(plan)))
            observation = state['feedback']
            if done:
                reason = 'success' if state['won'] else 'environment_done'
                break
        won = bool(state['won'])
    finally:
        env.close()
    env = create(False)
    try:
        env.seed(1000 + task['task_index'])
        state = env.reset()
        if state['feedback'] != initial or private_state(state, task['gamefile'])['key'] != initial_state['key']:
            raise ValueError('Planner replay initial state differs')
        for step in steps:
            if private_state(state, task['gamefile'])['key'] != step['key_before']:
                raise ValueError('Planner replay pre-state differs')
            state, _, done = env.step(step['action'])
            if (state['feedback'] != step['feedback'] or
                private_state(state, task['gamefile'])['key'] != step['key_after'] or
                bool(state['won']) != step['won'] or bool(done) != step['done']):
                raise ValueError('Planner replay transition differs')
        if bool(state['won']) != won:
            raise ValueError('Planner replay outcome differs')
    finally:
        env.close()
    return {**task, 'rep':16, 'source_kind':'planner', 'source_id':'fast_downward',
        'planner_backend':'TextWorld PddlState.replan / Fast Downward', 'maximum_steps':30,
        'model_calls':0, 'won':won, 'steps':steps, 'initial_observation':initial,
        'initial_key':initial_state['key'], 'initial_private_state':initial_state,
        'reason':reason, 'trusted_complete':True, 'trusted_success':won,
        'technical_incomplete':False, 'codec':'alf_pddl_relative_v1'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--task-index', type=int, help=argparse.SUPPRESS)
    a = p.parse_args()
    tasks = load_tasks(a.tasks)
    if any(t['split'] != 'train' for t in tasks) or a.workers < 1 or a.timeout <= 0:
        raise ValueError('Use training tasks and positive workers/timeout')
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    signature = {'tasks':tasks, 'data_root':str(Path(a.data_root).resolve()), 'max_steps':30}
    receipt = out/'COLLECTION.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != signature:
            raise ValueError('Planner output belongs to a different collection')
    else:
        if any(out.iterdir()):
            raise ValueError('Use an empty output or resume the same collection')
        write_json(receipt, signature)
    def target(t):
        return out/f"task_{t['task_index']:04d}_planner.json"
    if a.task_index is not None:
        task = next(t for t in tasks if t['task_index'] == a.task_index)
        write_json(target(task), execute_task(task, a.data_root))
        return
    def run(task):
        dest = target(task)
        if dest.exists():
            d = json.loads(dest.read_text())
            if d['gamefile'] != task['gamefile'] or not d.get('trusted_complete') or d.get('technical_incomplete'):
                raise ValueError('Invalid planner record; use a new output')
            return
        subprocess.run([sys.executable, '-m', 'gcopd.alfworld.collect_planner',
            '--tasks', str(Path(a.tasks).resolve()), '--data-root', str(Path(a.data_root).resolve()),
            '--output', str(out), '--task-index', str(task['task_index'])],
            check=True, timeout=a.timeout)
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(run, tasks))
    write_json(out/'COMPLETE.json', {'tasks':len(tasks), 'max_steps':30, 'all_records_replayed':True,
        'sources':[{'gamefile':t['gamefile'], 'task_index':t['task_index'],
                    'path':target(t).name, 'sha256':digest(target(t))} for t in tasks]})


if __name__ == '__main__':
    main()
