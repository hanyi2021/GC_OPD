from gcopd.common.io import ROOT
#!/usr/bin/env python3
"""Run one ALFWorld game or one WebShop goal with a local student model."""
import argparse
import json
from pathlib import Path
import re
import select
import subprocess
import urllib.request
import uuid
from gcopd.inference.generate import load_engine, generate
from gcopd.inference.protocol import INVALID_ACTION, INVALID_FEEDBACK, format_obs


def http(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                    headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=120) as response:
        result = json.load(response)
    if result.get('error'):
        raise RuntimeError(result['error'])
    return result


class AlfWorld:
    def __init__(self, args):
        tasks = json.loads((ROOT/'gcopd/alfworld/assets/evaluation_tasks.json').read_text())['tasks']
        self.task = next(x for x in tasks if x['task_index'] == args.task_index)
        game = Path(args.data_root)/self.task['gamefile']
        if not game.is_file():
            raise FileNotFoundError(f'ALFWorld game not found: {game}')
        self.proc = subprocess.Popen([args.env_python, str(ROOT/'gcopd/alfworld/evaluation_worker.py'), str(game)],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.state = self.call(op='reset', seed=self.task['environment_seed'])
        except Exception:
            self.close()
            raise
        match = re.search(r'Your task is to:\s*([^\n]+)', self.state['feedback'])
        if not match:
            self.close()
            raise ValueError('ALFWorld reset did not include a task instruction')
        self.initial = {'instruction': match.group(1).strip(),
                        'observation': (self.state['feedback'][:match.start()] + self.state['feedback'][match.end():]).strip(),
                        'available_actions': self.state['admissible_commands'], 'history': [],
                        'task_id': self.task['gamefile'], 'rep': args.rep,
                        'request_seeds': self.task['request_seeds'][str(args.rep)]}
    def call(self, **request):
        self.proc.stdin.write(json.dumps(request)+'\n'); self.proc.stdin.flush()
        if not select.select([self.proc.stdout], [], [], 120)[0]:
            raise TimeoutError('ALFWorld worker response timeout')
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('ALFWorld worker exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result
    def step(self, action):
        self.state = self.call(op='step', action=action)
        return self.state['feedback'], self.state['admissible_commands'], bool(self.state['done'] or self.state['won']), float(self.state['won'])
    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({'op': 'close'})+'\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill(); self.proc.wait()


class WebShop:
    def __init__(self, args):
        self.url = args.env_url.rstrip('/')
        self.episode = 'gcopd_example_' + uuid.uuid4().hex
        result = http(self.url+'/reset', {'episode': self.episode, 'goal': args.goal})
        self.initial = {'instruction': result['instruction'], 'observation': result['obs'],
                        'available_actions': result['available_actions'], 'history': [], 'goal': args.goal, 'rep': args.rep}
    def step(self, action):
        result = http(self.url+'/step', {'episode': self.episode, 'action': action})
        return result['obs'], result['available_actions'], bool(result['done']), float(result['reward'])
    def close(self):
        http(self.url+'/close', {'episode': self.episode})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=['alfworld', 'webshop'], required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--backend', choices=['transformers', 'vllm'], default='transformers')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--rep', type=int, choices=range(4), default=0)
    parser.add_argument('--data-root', help='ALFWorld directory containing valid_seen/ and valid_unseen/')
    parser.add_argument('--env-python', help='Python with official alfworld and textworld installed')
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--env-url', help='URL of webshop_env.py')
    parser.add_argument('--goal', type=int, default=0)
    parser.add_argument('--max-decisions', type=int)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5)
    args = parser.parse_args()
    run_episode(args)


def run_episode(args, engine=None):
    if args.environment == 'alfworld' and (not args.data_root or not args.env_python):
        raise ValueError('ALFWorld requires --data-root and --env-python')
    if args.environment == 'webshop' and not args.env_url:
        raise ValueError('WebShop requires --env-url')
    limit = args.max_decisions if args.max_decisions is not None else (30 if args.environment == 'alfworld' else 15)
    if not 1 <= limit <= (30 if args.environment == 'alfworld' else 15):
        raise ValueError('Decision limit is outside the environment protocol')
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f'Choose a new output: {output}')
    env = AlfWorld(args) if args.environment == 'alfworld' else WebShop(args)
    try:
        if engine is None:
            engine = load_engine(args.model, args.environment, args.gpu_memory_utilization, args.backend, args.device)
        state = env.initial
        steps, done, score = [], False, 0.0
        for turn in range(limit):
            prediction = generate(engine, args.environment, [state])[0]
            before = state['observation']
            history_before = format_obs(before, state['instruction']) if args.environment == 'webshop' else before
            if prediction['parse_error']:
                feedback, available = INVALID_FEEDBACK, state['available_actions']
                action = INVALID_ACTION
            else:
                feedback, available, done, reward = env.step(prediction['action'])
                action = prediction['action']
                if done:
                    score = reward
            state['history'].append({'observation': history_before, 'action': action})
            state['observation'], state['available_actions'] = feedback, available
            steps.append({'decision': turn+1, **prediction, 'feedback': feedback, 'done': done})
            if done:
                break
        import torch, transformers
        versions = {'torch': torch.__version__, 'transformers': transformers.__version__}
        if args.backend == 'vllm':
            import vllm
            versions['vllm'] = vllm.__version__
        result = {'model': str(Path(args.model).resolve()), 'runtime': versions,
                  'loading_info': getattr(engine, 'loading_info', None), 'environment': args.environment, 'task': state.get('task_id', state.get('goal')),
                  'rep': args.rep, 'backend': args.backend, 'done': done, 'score': score, 'won': bool(done and score >= 1-1e-9),
                  'decision_limit': limit, 'decisions': len(steps), 'steps': steps}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        print(json.dumps({k: result[k] for k in ['environment','task','done','won','decisions']}))
        return result
    finally:
        env.close()

if __name__ == '__main__':
    main()
