"""Conservative comparison of rendered ScienceWorld observations.

Only complete sibling items in recognized scopes may be reordered.
This comparison concerns rendered observations rather than hidden-state equivalence."""
import hashlib
import json
import re


VERSION = "supported_observation_item_enumeration_v2"
ROOM_HEADER = re.compile(
    r"^(?:This room is called the .+\. In it, you see:|"
    r"This outside location is called the .+\. Here you see:)$")
CONTAINER_HEADER = re.compile(r"^(?:Inside the .+ is:|In your inventory, you see:)$")
RELATION = re.compile(r"\b(?P<relation>On|In) the (?P<container>[^:()]+?) is:\s*")
ITEM_PREFIX = re.compile(r"^(?:a|an|the)\s+\S", re.I)
PREDICATE_WORDS = re.compile(r"\b(?:is|are|was|were|has|have|contains|which|that|then|will|and|or)\b", re.I)


def norm_inline(text):
    return re.sub(r"[\t ]+", " ", str(text)).strip()


def parse_menu(text):
    if "Ambiguous request:" not in str(text):
        return {}
    return {m.group(1): re.sub(r"\s+", " ", m.group(2)).strip()
            for m in re.finditer(r"^\s*(\d+)\s*:\s*(.+)$", str(text), re.M)}


def balanced_parentheses(text):
    depth = 0
    for character in text:
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def split_top_level_items(text):
    """Split commas only outside balanced parentheses, keeping nested text intact."""
    if any(character in text for character in '"[]{};') or not balanced_parentheses(text):
        return None, "unsupported_quotes_brackets_semicolon_or_unbalanced_parentheses"
    depth, start, parts = 0, 0, []
    for index, character in enumerate(text):
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            parts.append(norm_inline(text[start:index]))
            start = index + 1
    parts.append(norm_inline(text[start:]))
    if any(not item for item in parts):
        return None, "empty_enumeration_item"
    for item in parts:
        # Parenthesized contents/attributes stay attached to this complete item.
        # Their own comma order is deliberately NOT normalized.
        outside, level = [], 0
        for character in item:
            if character == "(":
                level += 1
            elif character == ")":
                level -= 1
            elif level == 0:
                outside.append(character)
        noun_phrase = norm_inline("".join(outside))
        if not ITEM_PREFIX.match(noun_phrase):
            return None, "not_all_items_have_supported_object_articles"
        if any(c in noun_phrase for c in ".:,;") or PREDICATE_WORDS.search(noun_phrase):
            return None, "descriptive_clause_not_supported_as_an_object_atom"
    return parts, None


def canonical_item_line(line, transformations):
    line = norm_inline(line)
    if not balanced_parentheses(line):
        raise ValueError("unbalanced_parentheses_in_item_line")
    matches = list(RELATION.finditer(line))
    if not matches:
        return {"kind": "literal_item", "text": line}
    if len(matches) != 1:
        # Chained On/In descriptions are retained verbatim; sorting their commas
        # could detach a nested child's contents from its parent container.
        transformations.append({"kind": "left_opaque", "reason": "multiple_inline_container_clauses", "text": line})
        return {"kind": "literal_item", "text": line}
    match = matches[0]
    prefix, body = line[:match.end()], line[match.end():]
    punctuation = "." if body.endswith(".") else ""
    if punctuation:
        body = body[:-1].rstrip()
    if body == "nothing":
        return {"kind": "inline_container", "prefix": prefix, "items": [],
                "empty_literal": "nothing", "punctuation": punctuation}
    items, reason = split_top_level_items(body)
    if items is None:
        transformations.append({"kind": "left_opaque", "reason": reason, "text": line})
        return {"kind": "literal_item", "text": line}
    sorted_items = sorted(items)  # A list, never a set: duplicate multiplicity matters.
    transformations.append({"kind": "inline_container_enumeration", "relation": match.group("relation"),
        "parent_container": norm_inline(match.group("container")), "item_count": len(items),
        "permutation_applied": items != sorted_items, "original_items": items, "sorted_items": sorted_items})
    return {"kind": "inline_container", "prefix": prefix, "items": sorted_items,
            "punctuation": punctuation}


def canonicalize_observation(text):
    """Return a structured canonical form only for explicitly recognized scopes."""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = normalized.splitlines()
    operations = []
    if not lines:
        return {"supported": False, "reason": "empty_text", "transformations": operations}
    header = norm_inline(lines[0])
    is_room = bool(ROOM_HEADER.fullmatch(header))
    is_container = bool(CONTAINER_HEADER.fullmatch(header))
    if not is_room and not is_container:
        if len(lines) == 1 and re.match(r"^(?:On|In) the [^:()]+ is:\s*", header):
            item = canonical_item_line(header, operations)
            if item["kind"] == "inline_container":
                return {"supported": True, "canonical": {"scope": "standalone_container", "item": item}, "transformations": operations}
        return {"supported": False, "reason": "unrecognized_observation_scope", "transformations": operations}
    sections = []
    section_header, items, indentation = "primary_objects", [], None

    def finish_section():
        sections.append({"header": section_header,
                         "items": sorted(items, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))})
        operations.append({"kind": "complete_sibling_item_lines", "section": section_header,
                           "item_count": len(items), "nested_parent_text_retained": True})

    try:
        for raw in lines[1:]:
            if not raw.strip():
                continue
            expanded = raw.expandtabs(4)
            indent = len(expanded) - len(expanded.lstrip())
            line = norm_inline(raw)
            if indent == 0 and line == "You also see:" and is_room and section_header == "primary_objects":
                finish_section()
                section_header, items, indentation = line, [], None
                continue
            if indent == 0:
                raise ValueError("unexpected_unindented_line_or_section")
            if indentation is None:
                indentation = indent
            if indent != indentation:
                raise ValueError("nested_multiline_indentation_not_supported")
            if not ITEM_PREFIX.match(line) and line not in {"nothing", "nothing."}:
                raise ValueError("non_object_line_inside_item_section")
            items.append(canonical_item_line(line, operations))
        finish_section()
    except ValueError as error:
        return {"supported": False, "recognized_scope": True, "reason": str(error), "transformations": operations}
    return {"supported": True, "canonical": {"scope_header": header, "sections": sections}, "transformations": operations}


def literal_lines(text):
    # Outside recognized object descriptions, preserve line order AND indentation.
    rows = str(text).replace("\r\n", "\n").replace("\r", "\n").strip().splitlines()
    return [(len(row.expandtabs(4)) - len(row.expandtabs(4).lstrip()), norm_inline(row)) for row in rows]


def equivalent(actual, expected):
    if str(actual).strip() == str(expected).strip():
        return {"equal": True, "rule": "literal_exact", "version": VERSION}
    left_menu, right_menu = parse_menu(actual), parse_menu(expected)
    if left_menu or right_menu:
        unique = bool(left_menu) and bool(right_menu) and len(set(left_menu.values())) == len(left_menu) and len(set(right_menu.values())) == len(right_menu)
        return {"equal": bool(unique and sorted(left_menu.values()) == sorted(right_menu.values())),
                "rule": "legacy_unique_semantic_menu_permutation", "version": VERSION,
                "menu_note": "Preserves existing collector behavior only; executing a numbered option still requires semantic binding."}
    left, right = canonicalize_observation(actual), canonicalize_observation(expected)
    if left["supported"] and right["supported"]:
        same = left["canonical"] == right["canonical"]
        return {"equal": same, "rule": "supported_object_enumerations" if same else "supported_observation_difference",
                "version": VERSION, "actual_transformations": left["transformations"],
                "expected_transformations": right["transformations"],
                "canonical_sha256": hashlib.sha256(json.dumps(left["canonical"], sort_keys=True, ensure_ascii=False).encode()).hexdigest() if same else None}
    return {"equal": literal_lines(actual) == literal_lines(expected), "rule": "literal_layout_fallback_no_reordering",
            "version": VERSION, "actual_supported": left["supported"], "expected_supported": right["supported"],
            "actual_reason": left.get("reason"), "expected_reason": right.get("reason")}
