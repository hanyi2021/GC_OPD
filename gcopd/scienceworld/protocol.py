"""ScienceWorld public protocol with task templates adapted from TCOD.

The two template strings change <think> to <thought>; the public student
protocol adds H5 history and strict action parsing. Upstream: kokolerk/TCOD,
commit 465eef4406ad0cff675b36bd46f37f28b1736ff9,
trinity/common/workflows/envs/TCOD/scienceworld/utils.py.
See README.md (License and attribution) and LICENSE.
"""

from gcopd.common.protocol import format_history_entry, parse_tagged_action, render_single_user

TCOD_MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"

TCOD_SCIWORLD_TEMPLATE_NO_HIS = "\nYour ScienceWorld task is: {task_description}\nYour current observation is: {current_observation}\nAvailable action commands: [{action_templates}]\nAvailable objects you can interact with: [{objects}]\n\nNow it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.\nYou should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> </thought> tags.\nOnce you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.\n"

TCOD_SCIWORLD_TEMPLATE = "\nYour ScienceWorld task is: {task_description}\nPrior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}\nYou are now at step {current_step} and your current observation is: {current_observation}\nAvailable action commands: [{action_templates}]\nAvailable objects you can interact with: [{objects}]\n\nNow it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.\nYou should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> </thought> tags.\nOnce you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.\n"


def tcod_format_history(observation, step_num, action):
    """Legacy signature; observation is the observation before this action."""
    return format_history_entry(observation, step_num, action)


def build_messages(prompts, task, observation, actions, objects, history):
    """Legacy signature, including the caller-supplied ScienceWorld templates."""
    fields = dict(
        task_description=task,
        current_observation=str(observation).strip(),
        action_templates=", ".join(f"'{a}'" for a in actions if a != "help"),
        objects=", ".join(f"'{o}'" for o in objects),
    )
    # Read only the selected attribute, as the original builder does.
    template = (prompts.TCOD_SCIWORLD_TEMPLATE if history
                else prompts.TCOD_SCIWORLD_TEMPLATE_NO_HIS)
    return render_single_user(template, template, fields, history)


def parse_format(text, obs=""):
    """Allow a blank action only for the recorded ambiguity/cancel observation."""
    return parse_tagged_action(
        text, allow_empty=lambda: "Ambiguous request:" in obs and "blank to cancel" in obs
    )


def parse_action(text, observation=""):
    """Convenience alias with the uniform observation argument name."""
    return parse_format(text, observation)


__all__ = [
    "TCOD_MEMORY_FORMAT", "TCOD_SCIWORLD_TEMPLATE_NO_HIS", "TCOD_SCIWORLD_TEMPLATE",
    "tcod_format_history", "build_messages", "parse_format", "parse_action",
]
