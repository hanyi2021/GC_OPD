"""ScienceWorld parsing, action validation and environment helpers.

These helpers are independent of source collection and service clients."""
import os
import re

from gcopd.scienceworld import protocol as prompts
from gcopd.scienceworld.protocol import parse_format

os.environ.setdefault("_JAVA_OPTIONS", "-XX:ActiveProcessorCount=2 -Xms64m -Xmx768m")

PATTERNS = [
    ('unknown_command', r'^(?:No known action matches that input\.|Unknown action\.)'),
    ('unsupported_use', r"^I'm not sure how to use "),
    ('unsupported_operation', r"^It's not clear how to (?:read|flush) that\."),
    ('invalid_destination', r"^That can't be moved there"),
    ('liquid_pickup', r"^You can't pick up a liquid directly\."),
    ('no_liquid', r'^The .+ does not contain any liquids to dunk into\.'),
    ('not_openable', r'^The .+ is not (?:openable|closeable|moveable|open)\.'),
    ('not_activatable', r'^The .+ is not something that can be activated\.'),
    ('closed_container', r"^The .+ isn't open, so you can't see inside\."),
    ('empty_mix', r'^That container is empty, so there are no items to mix\.'),
    ('single_item_mix', r'^There is only one thing \('),
]


def rejection_reason(feedback):
    if feedback is None:
        return None
    for reason, pattern in PATTERNS:
        if re.search(pattern, feedback.strip(), re.I):
            return reason
    return None


def choices(text):
    return dict(re.findall(r'(?m)^\s*(\d+):\s*(.+)$', text)) if 'Ambiguous request:' in text else {}
