"""Package-local import paths; importing this module never starts a runtime."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
PYTHON_PATHS = [ROOT, ROOT / 'external/verl/scienceworld']
from scripts.setup_backend import require_backend
BACKEND = ROOT / 'external/verl/scienceworld/verl'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def task_ids(path):
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data['tasks']
    ids = [item if isinstance(item, str) else item['gamefile'] for item in data]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Task list must be nonempty and contain unique gamefile identities')
    for item in ids:
        task, variation, simplification = item.split('|')
        int(variation)
        if not task or not simplification:
            raise ValueError('Invalid gamefile identity')
    return ids

def new_output(path):
    out = Path(path).expanduser().resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    return out

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
