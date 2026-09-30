"""JSON-lines TextWorld bridge with training-only PDDL state descriptors."""
import argparse
import hashlib
import json
import sys
import traceback


def private_state(state, task_id):
    canonical = {'gamefile': task_id, 'facts': sorted(str(f) for f in state['facts']),
                 'won': bool(state['won']), 'lost': bool(state.get('lost', False))}
    key = hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return {'key': key, 'facts': canonical['facts'], 'won': canonical['won'],
            'lost': canonical['lost'], 'capture_ok': True, 'codec': 'alf_pddl_relative_v1'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gamefile')
    parser.add_argument('task_id')
    args = parser.parse_args()
    wire = sys.stdout
    sys.stdout = sys.stderr
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    env = textworld.start(args.gamefile, textworld.EnvInfos(
        won=True, lost=True, facts=True, admissible_commands=True, extras=['gamefile']),
        wrappers=[AlfredDemangler, AlfredInfos])
    def pack(state, done=False):
        return {'feedback': state['feedback'], 'won': bool(state['won']),
                'admissible_commands': list(state['admissible_commands']),
                'done': bool(done), 'private_state': private_state(state, args.task_id)}
    try:
        for line in sys.stdin:
            try:
                req = json.loads(line)
                if req['op'] == 'reset':
                    env.seed(int(req.get('seed', 1000)))
                    result = pack(env.reset())
                elif req['op'] == 'step':
                    state, _, done = env.step(req['action'])
                    result = pack(state, done)
                elif req['op'] == 'close':
                    break
                else:
                    raise ValueError('Unknown environment operation')
                wire.write(json.dumps(result) + '\n')
            except Exception:
                wire.write(json.dumps({'error': traceback.format_exc()}) + '\n')
            wire.flush()
    finally:
        env.close()


if __name__ == '__main__':
    main()
