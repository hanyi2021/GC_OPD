"""Launch the environment in its own Python environment."""
import json
from pathlib import Path
import select
import subprocess
from gcopd.alfworld.tasks import resolve_gamefile


class Environment:
    def __init__(self, gamefile, log, *, env_python, data_root):
        actual = resolve_gamefile(data_root, gamefile)
        self.log = Path(log).open('w')
        try:
            self.proc = subprocess.Popen([str(env_python), str(Path(__file__).with_name('state_worker.py')),
                                          str(actual), gamefile], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=self.log, text=True)
        except BaseException:
            self.log.close()
            raise

    def call(self, **request):
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        if not select.select([self.proc.stdout], [], [], 120)[0]:
            raise TimeoutError('Environment RPC timeout')
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('Environment process exited')
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.log.close()
