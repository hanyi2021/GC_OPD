"""Pure protocol primitives shared by the recorded environment implementations.

Existing parser diagnostics are retained, with an additional nested-control-tag
check. These helpers validate syntax, not environment action admissibility.
"""

import re


MEMORY_FORMAT = "[Observation {step_num}: '{obs}', Action {step_num}: '{act}']"
HISTORY_LENGTH = 5
_CONTROL_TAG = re.compile(r"<\s*/?\s*(?:thought|action)(?=[\s>/]|$)", re.I)


def contains_control_tag(text):
    """Recognize reserved tags, including malformed or mixed-case openings."""
    return bool(_CONTROL_TAG.search(str(text)))


def format_history_entry(observation_before, decision_number, action):
    return MEMORY_FORMAT.format(
        step_num=decision_number, obs=str(observation_before).strip(), act=action
    )


def render_single_user(template_no_history, template_with_history, fields, history):
    """Render the last five entries while numbering from the full decision ledger."""
    if history:
        fields.update(
            step_count=len(history),
            current_step=len(history) + 1,
            history_length=min(HISTORY_LENGTH, len(history)),
            action_history="\n".join(history[-HISTORY_LENGTH:]),
        )
        user = template_with_history.format(**fields)
    else:
        user = template_no_history.format(**fields)
    return user, [{"role": "user", "content": user}]


def parse_tagged_action(text, *, allow_empty=False):
    """Parse one optional thought block followed by one plain action block."""
    if text.count("<action>") != 1 or text.count("</action>") != 1:
        return None, "missing_or_multiple_action_tags"
    match = re.fullmatch(
        r"\s*(?:<thought>(.*?)</thought>\s*)?<action>(.*?)</action>\s*", text, re.S
    )
    if not match:
        return None, "unclosed_tags_or_extra_text"
    if text.count("<thought>") != text.count("</thought>") or text.count("<thought>") > 1:
        return None, "invalid_thought_tags"
    if contains_control_tag(match.group(1) or "") or contains_control_tag(match.group(2)):
        return None, "nested_control_tags"
    action = match.group(2).strip().lower()
    if not action:
        permitted = allow_empty() if callable(allow_empty) else allow_empty
        if not permitted:
            return None, "empty_action"
    return action, None
