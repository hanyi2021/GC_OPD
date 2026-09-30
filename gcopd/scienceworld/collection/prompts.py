"""ScienceWorld source-collection prompts adapted from TCOD (Apache-2.0).

Upstream: kokolerk/TCOD, commit 465eef4406ad0cff675b36bd46f37f28b1736ff9,
trinity/common/workflows/envs/TCOD/scienceworld/utils.py.
The two task templates replace <think> with <thought>; the history format
is unchanged. Collection retains its accepted-action/retry protocol.
See README.md (License and attribution) and LICENSE.
"""
TCOD_HISTORY_LENGTH = 2
TCOD_MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"

TCOD_SCIWORLD_TEMPLATE_NO_HIS = """
Your ScienceWorld task is: {task_description}
Your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> </thought> tags.
Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
"""

TCOD_SCIWORLD_TEMPLATE = """
Your ScienceWorld task is: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Available action commands: [{action_templates}]
Available objects you can interact with: [{objects}]

Now it's your turn to take an action. Combine an action command with appropriate object(s) to form a valid action.
You should first reason step-by-step about the current situation. This reasoning process MUST be enclosed within <thought> </thought> tags.
Once you've finished your reasoning, you should choose a valid action for the current step and present it within <action> </action> tags.
"""


def tcod_build_user_msg(task_description, observation, action_templates, objects, history):
    """Render the source collector's user message from accepted-action history."""
    acts = ", ".join(f"'{s}'" for s in action_templates if s != "help")
    objs = ", ".join(f"'{s}'" for s in objects)
    obs = str(observation).strip()          # TCOD format_observation
    if len(history) < TCOD_HISTORY_LENGTH:
        return TCOD_SCIWORLD_TEMPLATE_NO_HIS.format(
            task_description=task_description, current_observation=obs,
            action_templates=acts, objects=objs)
    return TCOD_SCIWORLD_TEMPLATE.format(
        task_description=task_description,
        step_count=len(history),
        history_length=min(TCOD_HISTORY_LENGTH, len(history)),
        action_history="\n".join(history[-TCOD_HISTORY_LENGTH:]),
        current_step=len(history) + 1,
        current_observation=obs,
        action_templates=acts, objects=objs)


def tcod_format_history(observation, step_num, action):
    """Format an observation/action pair following TCOD's _format_history."""
    return TCOD_MEMORY_FORMAT.format(step_num=step_num,
                                     obs=str(observation).strip(), act=action)
