from gcopd.common.io import ROOT
"""The selected students' public text interaction protocol (no training dependency)."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re
from gcopd.alfworld.protocol import build_messages, format_history
from gcopd.webshop.protocol import WEBSHOP_TEMPLATE, WEBSHOP_TEMPLATE_NO_HIS

INVALID_ACTION = '[invalid response format; no environment action]'
INVALID_FEEDBACK = ('Invalid response format. Reply with optional <thought>...</thought> '
                    'followed by exactly one <action>...</action>. This decision round '
                    'has been consumed; the environment state is unchanged.')


def parse_action(text):
    if text.count('<action>') != 1 or text.count('</action>') != 1:
        return None, 'missing_or_multiple_action_tags'
    match = re.fullmatch(r'\s*(?:<thought>(.*?)</thought>\s*)?<action>(.*?)</action>\s*', text, re.S)
    if not match:
        return None, 'unclosed_tags_or_extra_text'
    if text.count('<thought>') != text.count('</thought>') or text.count('<thought>') > 1:
        return None, 'invalid_thought_tags'
    control = re.compile(r'<\s*/?\s*(?:thought|action)(?=[\s>/]|$)', re.I)
    if control.search(match.group(1) or '') or control.search(match.group(2)):
        return None, 'nested_control_tags'
    action = match.group(2).strip().lower()
    return (action, None) if action else (None, 'empty_action')


def format_obs(observation, instruction):
    parts = observation.split(' [SEP] ')
    try:
        index = parts.index(instruction)
        return ' [SEP] '.join(f"'{part}'" for part in parts[index + 1:])
    except ValueError:
        return observation


def build_prompt(environment, state):
    """State has instruction, observation, available_actions, and completed history.

    History entries contain observation BEFORE the action and the parsed action.
    The complete history preserves global decision numbering; render only H5/H2.
    """
    history = state.get('history', [])
    if environment == 'alfworld':
        ledger = [format_history(x['observation'], index + 1, x['action'])
                  for index, x in enumerate(history)]
        return build_messages(state['instruction'], state['observation'],
                              state['available_actions'], ledger)[0]
    if environment != 'webshop':
        raise ValueError('Expected alfworld or webshop')
    observation = format_obs(state['observation'], state['instruction'])
    available = state['available_actions']
    actions = (['search[<your query>]'] if available.get('has_search_bar') else [])
    actions += [f'click[{c}]' for c in available.get('clickables', [])]
    fields = {'task_description': state['instruction'], 'current_observation': observation,
              'available_actions': '\n'.join(f"'{a}'," for a in actions)}
    if not history:
        return WEBSHOP_TEMPLATE_NO_HIS.format(**fields)
    recent = history[-2:]
    start = len(history) - len(recent)
    fields.update(step_count=len(history), history_length=len(recent), current_step=len(history) + 1,
                  action_history='\n'.join(
                      f"[Observation {start+j+1}: '{x['observation']}', Action {start+j+1}: '{x['action']}']"
                      for j, x in enumerate(recent)))
    return WEBSHOP_TEMPLATE.format(**fields)


@lru_cache(maxsize=1)
def _alfworld_seeds():
    path = ROOT/'gcopd/alfworld/assets/evaluation_tasks.json'
    return {task['gamefile']: task['request_seeds'] for task in json.loads(path.read_text())['tasks']}


def request_seed(environment, state):
    turn = len(state.get('history', []))
    if 'request_seeds' in state:
        return int(state['request_seeds'][turn])
    rep = int(state.get('rep', 0))
    if environment == 'alfworld':
        recorded = _alfworld_seeds().get(state['task_id'])
        if recorded is not None:
            return int(recorded[str(rep)][turn])
        key = f"{state['task_id']}|{rep}|{turn}|0"
    else:
        key = f"{int(state['goal'])}|{rep}|{turn}|probe32_paired_v1"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2**31
