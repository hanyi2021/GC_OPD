"""Shared ALFWorld H5 prompt builder for training rollout and evaluation.

Derived from official TCOD 465eef4406ad0cff675b36bd46f37f28b1736ff9.
History contains completed decisions, each paired with its PRE-action observation.
No accumulated chat messages or historical assistant thoughts are accepted.
"""

ALFWORLD_TEMPLATE_NO_HIS = "\nYou are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}\nYour current observation is: {current_observation}\nYour admissible actions of the current situation are: [{admissible_actions}].\n\nNow it's your turn to take an action.\nYou should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> </thought> tags. \nOnce you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\n"

ALFWORLD_TEMPLATE = "\nYou are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}\nPrior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}\nYou are now at step {current_step} and your current observation is: {current_observation}\nYour admissible actions of the current situation are: [{admissible_actions}].\n\nNow it's your turn to take an action.\nYou should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> </thought> tags. \nOnce you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.\n"

MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"

def format_history(observation_before, decision_number, action):
    """Call once after each decision, including invalid/rejected decisions."""
    if decision_number < 1:
        raise ValueError("decision_number must be one-based")
    return MEMORY_FORMAT.format(step_num=decision_number,
                                obs=str(observation_before).strip(), act=action)


def build_messages(task, observation, admissible_actions, history):
    """Return fresh single-user messages; history is the full decision ledger.

    Keep the full ledger to preserve global numbering; slice only for rendering.
    After a decision append format_history(before, turn + 1, action_or_placeholder),
    then use its feedback as the next observation. Never append raw model replies.
    """
    fields = dict(task_description=task,
                  current_observation=str(observation).strip(),
                  admissible_actions="\n ".join(
                      f"'{a}'" for a in admissible_actions if a != "help"))
    if history:
        fields.update(step_count=len(history), current_step=len(history) + 1,
                      history_length=min(5, len(history)),
                      action_history="\n".join(history[-5:]))
        user = ALFWORLD_TEMPLATE.format(**fields)
    else:
        user = ALFWORLD_TEMPLATE_NO_HIS.format(**fields)
    return user, [{"role": "user", "content": user}]

"""ScienceWorld standard parser semantics; errors consume a decision, never terminate."""
import re

INVALID_FEEDBACK = ('Invalid response format. Reply with optional <thought>...</thought> '
                    'followed by exactly one <action>...</action>. This decision round '
                    'has been consumed; the environment state is unchanged.')
INVALID_ACTION = '[invalid response format; no environment action]'

def parse_action(text):
    if text.count('<action>') != 1 or text.count('</action>') != 1:
        return None, 'missing_or_multiple_action_tags'
    match = re.fullmatch(r'\s*(?:<thought>.*?</thought>\s*)?<action>(.*?)</action>\s*', text, re.S)
    if not match:
        return None, 'unclosed_tags_or_extra_text'
    if text.count('<thought>') != text.count('</thought>') or text.count('<thought>') > 1:
        return None, 'invalid_thought_tags'
    action = match.group(1).strip().lower()
    return (action, None) if action else (None, 'empty_action')
