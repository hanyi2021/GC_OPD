"""Read caller-supplied relative ALFWorld task identities without creating splits."""
import json
from pathlib import Path, PurePosixPath


def task_id(value):
    p = PurePosixPath(str(value))
    if p.is_absolute() or '..' in p.parts or not p.parts or '\\' in str(value):
        raise ValueError('ALFWorld gamefile must be a stable relative path below the data root')
    if p.name != 'game.tw-pddl':
        raise ValueError('ALFWorld task must identify a game.tw-pddl file')
    return p.as_posix()


def resolve_gamefile(data_root, gamefile):
    root = Path(data_root).expanduser().resolve()
    path = (root / task_id(gamefile)).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'ALFWorld game is missing or outside the data root: {gamefile}')
    return path


def load_tasks(path):
    data = json.loads(Path(path).read_text())
    values = data['tasks'] if isinstance(data, dict) else data
    if not isinstance(values, list) or not values:
        raise ValueError('Supply a nonempty task list')
    result = []
    for index, item in enumerate(values):
        item = {'gamefile': item} if isinstance(item, str) else dict(item)
        item['gamefile'] = task_id(item['gamefile'])
        item.setdefault('task_index', index)
        item.setdefault('split', 'train')
        if type(item['task_index']) is not int or item['task_index'] < 0:
            raise ValueError('task_index must be a nonnegative integer')
        result.append(item)
    for key in ['gamefile', 'task_index']:
        if len({x[key] for x in result}) != len(result):
            raise ValueError(f'Duplicate ALFWorld {key}')
    return result
