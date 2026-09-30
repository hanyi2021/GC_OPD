#!/usr/bin/env python3
"""Minimal local HTTP bridge to the official WebShop text environment."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--webshop-root', required=True, help='Official WebShop checkout with its data/indexes')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8300)
    args = parser.parse_args()
    root = Path(args.webshop_root).resolve()
    sys.path.insert(0, str(root))
    from web_agent_site.envs import web_agent_text_env as module
    # Keep environment sampling at 42 and official goal ordering at 233.
    original_server = module.SimServer
    class OrderedServer(original_server):
        def __init__(self, seed, *a, **kwargs):
            super().__init__(233, *a, **kwargs)
    module.SimServer = OrderedServer
    class Environment(module.WebAgentTextEnv):
        # Supply the evaluated rich-text view without patching the checkout.
        def __init__(self, *a, **kwargs):
            self._episode_prefix = kwargs.get('session_prefix', 'gcopd_base_')
            super().__init__(*a, **kwargs)
        @property
        def observation(self):
            return self.convert_html_to_text(self.state['html'], simple=False)
        def reset(self, session=None, instruction_text=None):
            goal = int(session) if session is not None else None
            self.session = self._episode_prefix + (str(goal) if goal is not None else uuid.uuid4().hex)
            self.browser.get(f'{self.base_url}/{self.session}', session_id=self.session, session_int=goal)
            self.text_to_clickable = None
            self.instruction_text = self.get_instruction_text() if instruction_text is None else instruction_text
            observation = self.observation
            self.prev_obs, self.prev_actions = [observation], []
            return observation, None
    base = Environment(observation_mode='text_rich', file_path=str(root/'data/items_shuffle.json'),
                       attr_path=str(root/'data/items_ins_v2.json'), num_products=None,
                       human_goals=True, seed=42)
    server = base.server
    episodes = {}
    lock = threading.Lock()
    def close(eid):
        item = episodes.pop(eid, None)
        if item:
            server.user_sessions.pop(item['session'], None)
        for key in list(server.user_sessions):
            if str(key).startswith('gcopd_'+eid+'_'):
                server.user_sessions.pop(key, None)
        return {'ok': True}
    def reset(eid, goal):
        close(eid)
        if not 0 <= int(goal) < len(server.goals):
            raise ValueError('WebShop goal index is out of range')
        try:
            env = Environment(observation_mode='text_rich', server=server,
                              session_prefix='gcopd_'+eid+'_', seed=42)
            observation, _ = env.reset(session=int(goal))
            episodes[eid] = {'env': env, 'session': env.session}
            instruction = server.user_sessions[env.session]['goal']['instruction_text']
            return {'obs': observation, 'instruction': instruction, 'available_actions': env.get_available_actions()}
        except Exception:
            close(eid)
            raise
    def step(eid, action):
        env = episodes[eid]['env']
        observation, reward, done, _ = env.step(action)
        available = {'has_search_bar': False, 'clickables': []} if done else env.get_available_actions()
        return {'obs': observation, 'reward': float(reward), 'done': bool(done), 'available_actions': available}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def send(self, status, obj):
            data = json.dumps(obj).encode()
            self.send_response(status); self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path != '/health':
                return self.send(404, {'error': 'Unknown endpoint'})
            self.send(200, {'ok': True, 'goals': len(server.goals), 'products': len(server.all_products),
                            'seed': 42, 'goal_shuffle_seed': 233, 'observation_mode': 'text_rich'})
        def do_POST(self):
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                eid = str(request['episode'])
                with lock:
                    if self.path == '/reset':
                        result = reset(eid, request['goal'])
                    elif self.path == '/step':
                        result = step(eid, request['action'])
                    elif self.path == '/close':
                        result = close(eid)
                    else:
                        raise ValueError('Unknown endpoint')
                self.send(200, result)
            except Exception as error:
                self.send(500, {'error': f'{type(error).__name__}: {error}'})
    httpd = ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'WebShop environment ready at http://{args.host}:{args.port}',flush=True)
    httpd.serve_forever()

if __name__ == '__main__':
    main()
