"""Shared physical state codec: legacy public object tree + focus + ordered goals.

No model, environment construction, training or file I/O. Getter-based capture
does not call env.step/look/inventory. Parser interaction is separate from the
legacy physical/control key and must be checked when selecting a visit.
"""
from collections import defaultdict
import copy
import hashlib
import json
import math
import re

CODEC_VERSION = "physical_tree_legacy1c_focus_ordered_v1"
TEMPERATURE_QUANTUM_C = 1.0
READ_VERBS = {"look", "read", "inventory", "task", "inspect", "examine"}


def norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def parse_menu(observation):
    if "Ambiguous request:" not in str(observation):
        return {}
    return {m.group(1): norm(m.group(2)) for m in re.finditer(r"^\s*(\d+)\s*:\s*(.+)$", str(observation), re.M)}


def interaction_state(observation, previous=None, env_step_called=True):
    if not env_step_called and previous is not None:
        return {**copy.deepcopy(previous), "preserved_without_env_step": True}
    options = parse_menu(observation)
    uncertain_rejection = (previous or {}).get("mode") in {"pending_menu", "unknown"} and not options and bool(re.search(
        r"^(?:No known action|Unknown action|I'm not sure|You must |The .+ (?:is not|isn't)|That can't|You can't|Connections must)",
        str(observation or "").strip(), re.I))
    if uncertain_rejection:
        return {"mode": "unknown", "options": {}, "source": "rejection_after_pending_without_new_menu",
                "actual_hidden_parser_state_certified": False, "preserved_without_env_step": False,
                "previous_mode": previous.get("mode"),
                "last_known_options": copy.deepcopy(previous.get("options") or previous.get("last_known_options", {}))}
    return {"mode": "pending_menu" if options else "normal", "options": options,
            "duplicate_option_descriptions": len(set(options.values())) != len(options),
            "source": "observed_parser_response", "actual_hidden_parser_state_certified": False,
            "preserved_without_env_step": False}


def update_focus(previous, feedback):
    text = str(feedback or "")
    if "You reset the goal progress and focus." in text:
        return None
    match = re.search(r"You focus on (?:the )?(.+?)\.", text)
    return match.group(1).strip() if match else previous


def ordered_goal_flags(progress_text):
    section = str(progress_text).split("Unordered")[0]
    return [[m[0], m[1]] for m in re.findall(r"^\s*(\d+)\s+(true|false)\s+(.+)$", section, re.M)]


def flatten_tree(tree):
    result = {}
    root_uuid = str(tree["uuid"])
    def walk(obj, parent=None, room=None):
        oid, name = str(obj["uuid"]), obj.get("name", "")
        if parent is not None and parent == root_uuid:
            room = name
        result[oid] = {"id": oid, "name": name, "parent": parent, "room": room,
                       "properties": {k:v for k,v in obj.items() if k not in ["uuid", "name", "contents"] and v is not None}}
        children = obj.get("contents", {})
        for child in children.values() if isinstance(children, dict) else children:
            walk(child, oid, room)
    walk(tree)
    return result


def temperature_signature(properties):
    material = properties.get("propMaterial", {}) or {}
    temperature = material.get("temperatureC")
    if not isinstance(temperature, (int, float)):
        return None
    thresholds = {k: material[k] for k in ["meltingPoint", "boilingPoint"] if isinstance(material.get(k), (int, float))}
    life = properties.get("propLife", {}) or {}
    thresholds.update({k:life[k] for k in ["minTemp", "maxTemp"] if isinstance(life.get(k), (int, float))})
    return {"bin": math.floor(temperature / TEMPERATURE_QUANTUM_C + .5),
            "threshold_sides": {k: -1 if temperature < v else 1 if temperature > v else 0 for k,v in thresholds.items()}}


def canonical_physical_tree(objects):
    children = defaultdict(list)
    for obj in objects.values():
        children[obj["parent"]].append(obj)
    def walk(obj):
        properties = dict(obj["properties"])
        material = dict(properties.get("propMaterial", {}) or {})
        if "temperatureC" in material:
            material["temperatureC"] = temperature_signature(properties)
            properties["propMaterial"] = material
        return {"name": obj["name"], "properties": properties,
                "contents": sorted([walk(child) for child in children[obj["id"]]], key=lambda x:json.dumps(x, sort_keys=True))}
    return sorted([walk(obj) for obj in children[None]], key=lambda x:json.dumps(x, sort_keys=True))


def encode_snapshot(objects, control):
    # Keep legacy JSON serialization exactly, including separators/ASCII policy.
    tree = canonical_physical_tree(objects)
    physical_hash = hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()
    fixed_control = {"focus_reference": control.get("focus_reference"), "goal_flags": control.get("goal_flags", [])}
    key = hashlib.sha256(json.dumps([physical_hash, fixed_control], sort_keys=True).encode()).hexdigest()
    return {"key": key, "physical_hash": physical_hash, "control": fixed_control, "codec_version": CODEC_VERSION}


def normalized_snapshot(snapshot):
    """Legacy-compatible in-place adapter; does not alter snapshot.objects."""
    snapshot.setdefault("raw_physical_hash", snapshot.get("physical_hash"))
    snapshot.update(encode_snapshot(snapshot["objects"], snapshot["control"]))
    return snapshot


def capture_state(env, focus_reference=None, current_observation="", score=None, *, info=None,
                  include_objects=True, previous_interaction=None, env_step_called=True, observation=None):
    if observation is not None:
        current_observation = observation
    try:
        objects = flatten_tree(env.getObjectTree())
        control = {"focus_reference": focus_reference, "goal_flags": ordered_goal_flags(env.get_goal_progress())}
        result = encode_snapshot(objects, control)
        interaction = interaction_state(current_observation, previous_interaction, env_step_called)
        result.update(room=next((o["room"] for o in objects.values() if o["name"] == "agent"), None),
                      tick=env.get_num_moves(), score=score if score is not None else (info or {}).get("score"),
                      object_count=len(objects), observation=current_observation,
                      interaction=interaction, capture_ok=True, matchable=interaction["mode"] != "unknown",
                      key_scope="public physical configuration + focus + ordered goal flags; parser interaction separate",
                      hidden_simulator_state_certified=False)
        if include_objects:
            result["objects"] = objects
        return result
    except Exception as exc:
        return {"key": None, "physical_hash": None, "control": None, "room": None, "tick": None,
                "score": score if score is not None else (info or {}).get("score"), "capture_ok": False,
                "matchable": False, "capture_error": repr(exc), "interaction": {"mode": "unknown", "options": {}},
                "codec_version": CODEC_VERSION}


def object_changes(before, after):
    changes = []
    for oid in sorted(set(before) | set(after), key=lambda x:int(x)):
        a, b = before.get(oid), after.get(oid)
        if a is None or b is None:
            changes.append({"object": (a or b)["name"], "id": oid, "field": "存在", "before": a is not None, "after": b is not None})
            continue
        def diff(x, y, field=""):
            if isinstance(x, dict) and isinstance(y, dict):
                for key in sorted(set(x) | set(y)):
                    diff(x.get(key), y.get(key), field + "." + key if field else key)
            elif x != y:
                changes.append({"object": b["name"], "id": oid, "field": field, "before": x, "after": y})
        diff(a, b)
    return changes


def meaningful_changes(before, after):
    result = []
    for change in object_changes(before, after):
        if change["field"] == "parent":
            change["before_label"] = before.get(str(change["before"]), {}).get("name", str(change["before"]))
            change["after_label"] = after.get(str(change["after"]), {}).get("name", str(change["after"]))
        if change["field"] == "properties.propMaterial.temperatureC":
            oid = change["id"]
            a, b = temperature_signature(before[oid]["properties"]), temperature_signature(after[oid]["properties"])
            if a == b:
                continue
            change["temperature_before_bin"], change["temperature_after_bin"] = a, b
        result.append(change)
    return result


def classify_transition(action, feedback, changed, rejection=None, format_error=None):
    binding = str(action).strip().isdigit() or bool(parse_menu(feedback))
    read = str(action).strip().split(" ")[0].lower() in READ_VERBS and not binding
    if changed:
        kind = "state_change"
    elif format_error:
        kind = "format_invalid"
    elif binding and not rejection:
        kind = "binding"
    elif read and not rejection:
        kind = "internal_observation"
    else:
        kind = "rejection" if rejection else "unchanged_non_observation"
    return {"kind": kind, "is_binding": binding, "is_observation": read,
            "passive_during_rejection": bool(changed and rejection), "decision_cost": 1}
