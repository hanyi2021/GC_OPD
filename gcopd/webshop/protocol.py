# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import re


WEBSHOP_TEMPLATE_NO_HIS = "\nYou are an expert autonomous agent operating in the WebShop e‑commerce environment. \nYour task is to: {task_description}.\nYour current observation is: {current_observation}.\nYour admissible actions of the current situation are: \n[\n{available_actions}\n].\n\nNow it's your turn to take one action for the current step.\nYou should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> </thought> tags. \nOnce you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\n"

WEBSHOP_TEMPLATE = "\nYou are an expert autonomous agent operating in the WebShop e‑commerce environment.\nYour task is to: {task_description}.\nPrior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}\nYou are now at step {current_step} and your current observation is: {current_observation}.\nYour admissible actions of the current situation are: \n[\n{available_actions}\n].\n\nNow it's your turn to take one action for the current step.\nYou should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <thought> </thought> tags. \nOnce you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\n"

INVALID_FEEDBACK = 'Invalid response format. Reply with optional <thought>...</thought> followed by exactly one <action>...</action>. This decision round has been consumed; the environment state is unchanged.'

INVALID_ACTION = '[invalid response format; no environment action]'

def format_obs(obs, task):
    parts = obs.split(' [SEP] ')
    try:
        i = parts.index(task)
        return ' [SEP] '.join(f"'{p}'" for p in parts[i + 1:])
    except ValueError:
        return obs

def format_avail(av):
    acts = (['search[<your query>]'] if av.get('has_search_bar') else []) + [f'click[{c}]' for c in av.get('clickables', [])]
    return '\n'.join(f"'{a}'," for a in acts)

def _render_prompt(task, obs_fmt, avail_fmt, history, h):
    """Render exactly the delivered template; history entries are (observation, action)."""
    if not history or h <= 0:
        return WEBSHOP_TEMPLATE_NO_HIS.format(task_description=task, current_observation=obs_fmt, available_actions=avail_fmt)
    recent = history[-h:]
    start = len(history) - len(recent)
    text = '\n'.join(f"[Observation {start+j+1}: '{o}', Action {start+j+1}: '{a}']" for j, (o, a) in enumerate(recent))
    return WEBSHOP_TEMPLATE.format(task_description=task, step_count=len(history), history_length=len(recent), action_history=text, current_step=len(history)+1, current_observation=obs_fmt, available_actions=avail_fmt)

def parse_action(text):
    # Keep WebShop's own delivered parser/check order rather than substituting an
    # environment-specific H5 parser or the historical WS_PROTO=tcod fallback.
    if text.count('<action>') != 1 or text.count('</action>') != 1:
        return None, 'missing_or_multiple_action_tags'
    m = re.fullmatch(r'\s*(?:<thought>.*?</thought>\s*)?<action>(.*?)</action>\s*', text, re.S)
    if not m:
        return None, 'unclosed_tags_or_extra_text'
    if text.count('<thought>') != text.count('</thought>') or text.count('<thought>') > 1:
        return None, 'invalid_thought_tags'
    action = m.group(1).strip().lower()
    return (action, None) if action else (None, 'empty_action')

def rejected_by_environment(action, available):
    """Original dispatcher diagnostic; deliberately not a stricter fullmatch check."""
    m = re.match(r'(.+)\[(.+)\]', action)
    if not m:
        return True
    kind, arg = m.groups()
    arg = arg.lower()
    if kind == 'search':
        return not bool(arg)
    return not (kind == 'click' and arg in {x.lower() for x in available.get('clickables', [])} and arg != 'search')

def build_prompt(task, obs_fmt, avail_fmt, history, history_length=2, max_rounds=None):
    if history_length != 2:
        raise ValueError("The main WebShop protocol uses H2")
    return _render_prompt(task, obs_fmt, avail_fmt, history, 2)

