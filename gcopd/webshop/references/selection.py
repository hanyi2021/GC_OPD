"""Frozen selection-only helpers from MAIN; no legacy thought renderer is included."""
from functools import lru_cache
import gzip, hashlib, json, os, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SHORTEST_SCOPE = 'recorded_trusted_source_suffixes'
import os as _os
SOURCE_RULE = 'binary'
SCORE_EPS = 1e-09

def source_score(trace):
    return 1.0 if trace.get('won') is True else 0.0
_CTX = {'student_score': None, 'best_score': None}

def set_student_context(student_score, best_score):
    _CTX['student_score'] = float(student_score)
    _CTX['best_score'] = float(best_score)

def better_than_student(trace):
    return trace.get('won') is True

def norm(value):
    return re.sub('\\s+', ' ', str(value or '')).strip()

def interaction(value):
    if not isinstance(value, dict):
        return None
    mode, options = (value.get('mode'), value.get('options'))
    if mode not in {'normal', 'pending_menu'} or not isinstance(options, dict):
        return None
    if mode == 'normal' and options or (mode == 'pending_menu' and (not options)):
        return None
    return (mode, tuple(sorted(((str(k), norm(v)) for k, v in options.items()))))

def locator(state):
    if not isinstance(state, dict) or state.get('capture_ok') is not True or state.get('matchable') is not True:
        return None
    signature = interaction(state.get('interaction'))
    key = state.get('key', state.get('node_key'))
    return (key, signature) if key and signature is not None else None

@lru_cache(maxsize=64)
def _read_json(path, mtime_ns, size):
    return json.loads(Path(path).read_text())

def read_json(path):
    path = Path(path)
    stat = path.stat()
    return _read_json(str(path), stat.st_mtime_ns, stat.st_size)

@lru_cache(maxsize=8)
def _read_graph(path, mtime_ns, size):
    with gzip.open(path, 'rt') as handle:
        return json.load(handle)

def load_catalog_graph(catalog_root, gamefile):
    root = Path(catalog_root).expanduser().resolve()
    manifest_path = root / 'MANIFEST.json'
    audit = {'catalog_root': str(root), 'fixed_main_scope': 'caller_supplied_training_tasks', 'manifest_membership_checked': False}
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        tasks = manifest.get('tasks')
        if not isinstance(tasks, list):
            raise ValueError('Covered catalog MANIFEST must have a tasks list')
        allowed = {item['gamefile'] for item in tasks}
        if gamefile not in allowed:
            raise ValueError(f'Task is outside the fixed covered catalog: {gamefile}')
        task = next((item for item in tasks if item['gamefile'] == gamefile))
        raw_success_files = []
        for item in task.get('source_files', []):
            if item.get('source_won') is not True or item.get('expected_missing'):
                continue
            resolved = dict(item)
            if resolved.get('path'):
                source_path = Path(resolved['path'])
                resolved['path'] = str(source_path if source_path.is_absolute() else root / source_path)
            raw_success_files.append(resolved)
        audit.update(manifest_membership_checked=True, manifest_task_count=len(allowed), raw_source_success_reps=task.get('source_success_reps', []), raw_success_source_files=raw_success_files)
    task_hash = hashlib.sha256(gamefile.encode()).hexdigest()[:16]
    entry_path = root / 'entries' / (task_hash + '.json')
    if not entry_path.is_file():
        return (None, {**audit, 'fallback': 'catalog_entry_not_ready'})
    entry = read_json(entry_path)
    if entry.get('gamefile') != gamefile:
        raise ValueError('Catalog entry gamefile mismatch')
    audit.update(entry_path=str(entry_path), entry_status=entry.get('status'))
    descriptor = entry.get('graph')
    if entry.get('status') not in {'complete', 'partial'} or not isinstance(descriptor, dict) or (not descriptor.get('path')):
        return (None, {**audit, 'fallback': 'catalog_has_no_readable_graph'})
    path = Path(descriptor['path'])
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        return (None, {**audit, 'fallback': 'published_graph_file_missing'})
    stat = path.stat()
    if descriptor.get('sha256') and hashlib.sha256(path.read_bytes()).hexdigest() != descriptor['sha256']:
        raise ValueError('Graph hash differs from its catalog entry')
    import copy
    graph = copy.deepcopy(_read_graph(str(path), stat.st_mtime_ns, stat.st_size))
    for trace in graph.get('trajectories', []):
        source_path = Path(trace['source_path'])
        if not source_path.is_absolute():
            trace['source_path'] = str(root / source_path)
    if graph.get('gamefile') != gamefile or not isinstance(graph.get('nodes'), dict) or (not isinstance(graph.get('trajectories'), list)):
        raise ValueError('Published graph has incompatible task or schema')
    audit.update(graph_path=str(path), published_graph_sha256=descriptor.get('sha256'))
    audit.setdefault('raw_source_success_reps', [t['rep'] for t in graph['trajectories'] if t.get('source_won') is True])
    return (graph, audit)

@lru_cache(maxsize=64)
def _original_source(path, mtime_ns, size, expected_sha256):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected_sha256 and actual != expected_sha256:
        return (None, 'Original source hash does not match the replayed source; original response text is unavailable.')
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        return (None, 'Original source is not a JSON episode object.')
    return (parsed, None)

def original_source(trace):
    path = trace.get('source_path')
    if not path or not Path(path).is_file():
        return (None, 'Original response file is unavailable; only recorded action/feedback is available.')
    try:
        stat = Path(path).stat()
        return _original_source(str(path), stat.st_mtime_ns, stat.st_size, trace.get('source_sha256'))
    except (OSError, ValueError, TypeError) as exc:
        return (None, 'Original response file cannot be read: ' + type(exc).__name__)

def trusted(trace):
    return trace.get('trusted_complete') is True

def successful(trace):
    return trusted(trace) and trace.get('won') is True and (trace.get('trusted_success', True) is True)

def failed_source(trace):
    return trusted(trace) and trace.get('won') is False and (trace.get('source_won', False) is False)

def positive_step(step):
    if step.get('positive_candidate_allowed') is False or step.get('rolled_back_attempt'):
        return False
    if step.get('format_error') or step.get('rejection') or step.get('env_rejection') or step.get('passive_during_rejection'):
        return False
    if step.get('kind') in {'rejection', 'passive_during_rejection', 'format_invalid', 'format_error'}:
        return False
    return bool(step.get('action')) or step.get('kind') == 'binding'

def continuous_suffix(trace, start):
    steps = trace['steps']
    for i in range(start + 1, len(steps)):
        row = steps[i]
        if row.get('recording_gap_before') or row.get('continuous_from_previous') is False:
            return False
        if norm(steps[i - 1].get('feedback')) != norm(row.get('observation_before')):
            return False
    return True

def committed_cost(trace, start):
    total = 0
    for row in trace['steps'][start:]:
        cost = row.get('decision_cost', 1)
        if not isinstance(cost, (int, float)) or cost < 1:
            raise ValueError('A recorded decision cannot be free')
        total += cost
    return total

def source_cost(trace, start):
    """Verified committed route cost; collection retries do not alter this path."""
    lower = committed_cost(trace, start)
    return (lower, 'verified_committed_decisions_including_reads_and_bindings', lower)

def source_collection_attempt_cost(trace, start):
    """Separate data-collection cost, never used as the verified route length."""
    lower = committed_cost(trace, start)
    raw, _ = original_source(trace)
    attempts = raw.get('attempts') if raw else None
    if isinstance(attempts, list) and attempts and all((isinstance(a.get('step'), int) for a in attempts)):
        recorded = sum((a['step'] >= start + 1 for a in attempts))
        if recorded >= lower:
            return recorded
    return None

def candidate(trace, start):
    cost, cost_scope, transitions = source_cost(trace, start)
    return {'kind': 'catalog_source', 'trace': trace, 'anchor': start, 'cost': cost, 'cost_scope': cost_scope, 'committed_suffix_cost': transitions, 'source_collection_attempt_cost': source_collection_attempt_cost(trace, start)}

def candidate_summary(value):
    if value is None:
        return None
    return {'kind': value['kind'], 'rep': value['trace']['rep'] if value['kind'] == 'catalog_source' else None, 'anchor_state_t': value['anchor'], 'decision_cost': value['cost'], 'committed_suffix_cost': value['committed_suffix_cost'], 'cost_scope': value['cost_scope'], 'source_collection_attempt_cost': value.get('source_collection_attempt_cost'), 'shortest_scope': SHORTEST_SCOPE}

def rank(value):
    return (value['cost'], value['kind'] == 'student_self', value['trace']['rep'] if value['kind'] == 'catalog_source' else -1, value['anchor'])

def build_index(graph):
    visits, successes, failures = ({}, [], [])
    if graph is None:
        return (visits, successes, failures)
    for trace in graph['trajectories']:
        if successful(trace):
            successes.append(trace)
        elif failed_source(trace):
            failures.append(trace)
        for t, state in enumerate(trace['states']):
            if t > len(trace['steps']) or state.get('rolled_back_attempt'):
                continue
            signature = locator(state)
            if signature and signature[0] in graph['nodes']:
                visits.setdefault(signature, []).append((trace, t))
    return (visits, successes, failures)

def successful_visits(visits, signature):
    if signature is None:
        return []
    output = []
    for trace, t in visits.get(signature, []):
        if not successful(trace) or not continuous_suffix(trace, t):
            continue
        if t < len(trace['steps']) and (not positive_step(trace['steps'][t])):
            continue
        output.append(candidate(trace, t))
    return sorted(output, key=rank)

def current_state(steps, t):
    row = steps[t]
    state = row.get('physical_state_before', row.get('physical_state'))
    if state is None and t > 0:
        state = steps[t - 1].get('physical_state_after')
    return state

def nearest_success_anchor(steps, t, visits, successes):
    for j in range(t, -1, -1):
        choices = successful_visits(visits, locator(current_state(steps, j)))
        if choices:
            return {'kind': 'current_exact_anchor' if j == t else 'historical_common_anchor', 'student_state_t': j, 'candidate': choices[0], 'reliable_physical_anchor': True, 'can_execute_from_current_state_certified': False}
    choices = [candidate(trace, 0) for trace in successes if continuous_suffix(trace, 0) and (not trace['steps'] or positive_step(trace['steps'][0]))]
    if choices:
        return {'kind': 'same_task_unanchored_success', 'student_state_t': None, 'candidate': min(choices, key=rank), 'reliable_physical_anchor': False, 'can_execute_from_current_state_certified': False}
    return None

def related_failure(steps, t, visits, failures):
    for j in range(t, -1, -1):
        choices = [(trace, at) for trace, at in visits.get(locator(current_state(steps, j)), []) if failed_source(trace)]
        if choices:
            trace, at = min(choices, key=lambda item: (abs(item[1] - j), source_cost(item[0], 0)[0], item[0]['rep'], item[1]))
            return {'trace': trace, 'anchor': at, 'student_state_t': j, 'relation': 'current_exact_visit' if j == t else 'latest_shared_student_history_visit'}
    if failures:
        trace = min(failures, key=lambda x: (source_cost(x, 0)[0], x['rep']))
        return {'trace': trace, 'anchor': None, 'student_state_t': None, 'relation': 'same_task_only_no_reliable_shared_anchor'}
    return None

def unverified_raw_success(graph, catalog_audit):
    """Whole raw history fallback; never admitted to trusted suffix ranking."""
    candidates = list(catalog_audit.get('raw_success_source_files', []))
    traces = {t['rep']: t for t in graph['trajectories']} if graph else {}
    known = {c['rep'] for c in candidates}
    for trace in traces.values():
        if trace.get('source_won') is True and trace['rep'] not in known:
            candidates.append({'rep': trace['rep'], 'path': trace.get('source_path'), 'sha256': trace.get('source_sha256')})
    readable = []
    for item in candidates:
        stub = {'source_path': item.get('path'), 'source_sha256': item.get('sha256')}
        raw, issue = original_source(stub)
        if issue or not raw or raw.get('won') is not True or (not isinstance(raw.get('steps'), list)):
            continue
        original = traces.get(item['rep'], {})
        trace = {**stub, 'rep': item['rep'], 'steps': raw['steps'], 'trusted_complete': False, 'won': False, 'source_won': True, 'raw_recorded_success_reference': True, 'replay_failures': original.get('replay_failures', []), 'replay_limit': 'The source file records success, but this complete physical replay/connecting seam is not verified as usable. No exact physical LCA or executable suffix is claimed.'}
        count = len(raw.get('attempts', raw['steps']))
        readable.append((count, item['rep'], trace))
    return min(readable, key=lambda x: x[:2])[2] if readable else None

def task_best_reference(steps, t, visits, traces):
    """整个图里最终 reward 最高的可信轨迹（须高于学生），锚在学生历史里最近的共同访问位置；无共同访问则未对齐。"""
    cands = [tr for tr in traces if trusted(tr) and better_than_student(tr) and continuous_suffix(tr, 0)]
    if not cands:
        return None
    best = min(cands, key=lambda tr: (-source_score(tr), committed_cost(tr, 0), tr['rep']))
    for j in range(t, -1, -1):
        sig = locator(current_state(steps, j))
        for tr, at in visits.get(sig, []) if sig else []:
            if tr is best and continuous_suffix(tr, at):
                return {'trace': best, 'anchor': at, 'student_state_t': j, 'score': source_score(best), 'relation': 'current_exact_visit' if j == t else 'latest_shared_student_history_visit'}
    return {'trace': best, 'anchor': None, 'student_state_t': None, 'score': source_score(best), 'relation': 'same_task_unanchored'}

def plan_step(trajectory, t, visits, successes, failures, graph):
    steps = trajectory['steps']
    student_score = 1.0 if trajectory.get('won') else 0.0
    best = max([source_score(tr) for tr in (graph['trajectories'] if graph else [])] + [student_score])
    set_student_context(student_score, best)
    signature = locator(current_state(steps, t))
    matches = visits.get(signature, []) if signature else []
    source_choices = successful_visits(visits, signature)
    won = trajectory['won'] is True
    self_choice = {'kind': 'student_self', 'trace': None, 'anchor': t, 'cost': len(steps) - t, 'committed_suffix_cost': len(steps) - t, 'source_collection_attempt_cost': len(steps) - t, 'cost_scope': 'all_actual_student_decisions_including_format_errors_and_rejections'} if won else None
    choices = source_choices + ([self_choice] if self_choice else [])
    selected = min(choices, key=rank) if choices else None
    if matches and source_choices:
        case = 'graph_inside_with_success_suffix'
    elif matches:
        case = 'graph_inside_without_usable_success_suffix'
    elif won:
        case = 'graph_outside_student_succeeded'
    else:
        case = 'graph_outside_student_failed'
    membership = 'matched_committed_visit' if matches else 'unavailable_or_unmatchable_locator' if signature is None else 'no_exact_trusted_visit_for_key_and_parser'
    historical = nearest_success_anchor(steps, t, visits, successes) if not source_choices else None
    failed = related_failure(steps, t, visits, failures) if _os.environ.get('WS_FAILED_CONTRAST', '1') == '1' and case in {'graph_inside_without_usable_success_suffix', 'graph_outside_student_failed'} else None
    task_best = None
    return {'case': case, 't': t, 'graph_membership_status': membership, 'current_graph_visit_match': bool(matches), 'current_physical_key_present': bool(graph and signature and (signature[0] in graph['nodes'])), 'source_success_candidates': source_choices, 'selected_success': selected, 'historical_success': historical, 'failed_reference': failed, 'task_best': task_best, 'student_final_outcome': 'success' if trajectory['won'] else 'failure', 'student_final_score': student_score, 'task_best_source_score': best, 'source_rule': SOURCE_RULE}

def insert_reference(tokenizer, base_prompt, text):
    ids = list(base_prompt)
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    positions = [i for i, value in enumerate(ids) if value == end]
    if not positions:
        raise ValueError('Original student prompt has no ChatML user-message end')
    i = positions[-1]
    if not tokenizer.decode(ids[i:], skip_special_tokens=False).startswith('<|im_end|>\n<|im_start|>assistant'):
        raise ValueError('Unexpected original student prompt suffix')
    return ids[:i] + tokenizer.encode('\n' + text + '\n', add_special_tokens=False) + ids[i:]
