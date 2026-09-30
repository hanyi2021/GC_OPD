"""Observable ScienceWorld state and semantic command identity.

Rendering order is normalized only inside recognized unordered scene lists.
Histories, scores, budgets, numeric properties and parser bindings are retained.
These keys propose state matches; they do not certify hidden-state equivalence.
"""
import dataclasses
import hashlib
import json
import re

STATE_SCHEMA = 'scienceworld_observable_history_v2'
PARSER_REJECTION = 'No known action matches that input.'
_NOUN = re.compile(r'^(?:a|an|the) ', re.I)
_CHOICE = re.compile(r'^(\d+):\s*(.+)$')


def action_rejection_reason(observation):
    """Explicit failures from simulator action validators, not generic negation.

    ActionUseDevice, ActionConnectElectrical and ActionMoveObject return false
    for these strings. An off light, closed-container observation or broken
    device discovery is not automatically rejected by this conservative list.
    """
    text = str(observation).strip()
    if text == PARSER_REJECTION:
        return 'parser_rejection'
    if (text == "I'm not sure how to use those two things together."
            or re.fullmatch(r"I'm not sure how to use the .+\.", text)):
        return 'unsupported_device_use'
    if text.endswith(' is already connected, and must be disconnected before it can be connected to something else'):
        return 'occupied_electrical_terminal'
    if text == "That can't be moved there." or text.startswith("That can't be moved there, because "):
        return 'invalid_move_destination'
    return None


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def object_key(objects):
    # Compatible with the original retrieval buckets, which are proposals only.
    return hashlib.sha256(json.dumps(sorted(str(x).strip().lower() for x in objects),
                                     ensure_ascii=False).encode()).hexdigest()[:24]


def split_user(user):
    prefix, rest = str(user).rsplit('current observation is:', 1)
    observation, suffix = rest.split('\nAvailable action commands:', 1)
    return prefix, observation.strip(), suffix


def observation_from_user(user):
    return split_user(user)[1]


def parser_choices(observation):
    if 'Ambiguous request:' not in observation:
        return {}
    result = {}
    for line in observation.splitlines():
        match = _CHOICE.match(line.strip())
        if match:
            result[match.group(1)] = ' '.join(match.group(2).split())
    return result


def _normalize_enumeration(body):
    """Sort noun phrases, preserving parentheses, properties and containment."""
    if "'" in body or '"' in body:
        return body
    period = '.' if body.endswith('.') else ''
    text = body[:-1] if period else body
    boundaries, depth = [0], 0
    for i, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
            if depth < 0:
                return body
        elif char == ',' and depth == 0 and _NOUN.match(text[i + 1:].lstrip()):
            boundaries.append(i + 1)
    if depth or len(boundaries) < 2:
        return body
    ends = [x - 1 for x in boundaries[1:]] + [len(text)]
    items = [text[start:end].strip() for start, end in zip(boundaries, ends)]
    if not all(_NOUN.match(item) for item in items):
        return body
    return ', '.join(sorted(items)) + period


def _scene_line(line):
    # Normalize only lists whose owning container is explicit and unchanged.
    match = re.search(r'(\. (?:On|In) the .+? is: |You also see: )', line)
    if match:
        return line[:match.end()] + _normalize_enumeration(line[match.end():])
    return line


def canonical_observation(observation):
    text = str(observation).strip()
    if parser_choices(text):
        # Numeric bindings remain exact in the node representation.
        return text
    if not any(line.startswith('This room is called ') for line in text.splitlines()):
        return text
    output, siblings = [], []

    def flush():
        output.extend(sorted(siblings))
        siblings.clear()

    for raw_line in text.splitlines():
        line = _scene_line(raw_line)
        if re.match(r'^\t(?:a |an |the |A |An |The )', line):
            siblings.append(line)
        else:
            flush()
            output.append(line)
    flush()
    return '\n'.join(output)


def observation_relation(expected, actual):
    if expected.strip() == actual.strip():
        return 'exact'
    expected_choices, actual_choices = parser_choices(expected), parser_choices(actual)
    if expected_choices or actual_choices:
        # Permit alpha-renaming of menu numbers only with a unique semantic map.
        left, right = list(expected_choices.values()), list(actual_choices.values())
        if (left and len(left) == len(right) and len(set(left)) == len(left)
                and len(set(right)) == len(right) and sorted(left) == sorted(right)):
            expected_head = expected.split('\n0:', 1)[0]
            actual_head = actual.split('\n0:', 1)[0]
            if expected_head == actual_head:
                return 'choice_permutation'
        return None
    if canonical_observation(expected) == canonical_observation(actual):
        return 'scene_permutation'
    return None


@dataclasses.dataclass(frozen=True)
class ActionSpec:
    raw: str
    semantic: str
    choice: str | None = None
    rejected_in_source: bool = False

    def as_dict(self):
        return dataclasses.asdict(self)


def action_spec(command, observation, observation_after=None):
    raw = str(command).strip().lower()
    choices = parser_choices(observation)
    binding = choices.get(raw) if raw.isdigit() else None
    semantic = 'choice:' + binding if binding is not None else 'command:' + raw
    rejected = observation_after is not None and action_rejection_reason(observation_after) is not None
    return ActionSpec(raw, semantic, binding, rejected)


def resolve_action(spec, observation):
    """Return the local menu number for the same semantic choice; never guess."""
    if spec.choice is None:
        return spec.raw
    matches = [number for number, text in parser_choices(observation).items()
               if text == spec.choice]
    if len(matches) != 1:
        raise ValueError('semantic parser choice is absent or ambiguous')
    return matches[0]


def visible_key(observation, objects, score, templates=()):
    return digest({'observation': canonical_observation(observation),
                   'objects': object_key(objects), 'score': score,
                   'templates': sorted(templates)})


def initial_history(gamefile):
    return digest([STATE_SCHEMA, gamefile])


def advance_history(previous, observation, spec, score_after):
    return digest([previous, canonical_observation(observation), spec.semantic, score_after])


def state_descriptor(gamefile, observation, objects, templates, score, outer_turn,
                     internal_moves, history):
    state = {'schema': STATE_SCHEMA, 'gamefile': gamefile,
             'observation': canonical_observation(observation),
             'objects_key': object_key(objects), 'templates': sorted(templates),
             'score': score, 'outer_turn': outer_turn, 'internal_moves': internal_moves,
             'history': history, 'parser_bindings': parser_choices(observation)}
    state['visible_key'] = visible_key(observation, objects, score, templates)
    state['id'] = digest(state)
    return state
