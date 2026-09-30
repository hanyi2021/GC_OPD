#!/usr/bin/env python3
"""JSON-lines worker for an installed official ALFWorld/TextWorld environment."""
import json
import sys
import traceback


def main():
    wire = sys.stdout
    sys.stdout = sys.stderr
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    env = textworld.start(sys.argv[1],
            textworld.EnvInfos(won=True, admissible_commands=True, extras=['gamefile']),
            wrappers=[AlfredDemangler, AlfredInfos])
    def pack(state, done=False):
        return {'feedback': state['feedback'], 'won': bool(state['won']),
                'admissible_commands': list(state['admissible_commands']), 'done': bool(done)}
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                if request['op'] == 'close':
                    break
                if request['op'] == 'reset':
                    env.seed(int(request['seed']))
                    result = pack(env.reset())
                elif request['op'] == 'step':
                    state, _, done = env.step(request['action'])
                    result = pack(state, done)
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
