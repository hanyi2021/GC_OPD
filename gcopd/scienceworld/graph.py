"""Build physical configuration graph with separate visits and decision records."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import gcopd.scienceworld.state as codec

_REFUSALS = [r'^(?:No known action matches that input\.|Unknown action\.)', r"^I'm not sure how to use ",
             r"^It's not clear how to (?:read|flush) that\.", r"^That can't be moved there", r"^You can't pick up a liquid directly\.",
             r'^The .+ does not contain any liquids to dunk into\.', r'^The .+ is not (?:openable|closeable|moveable|open)\.',
             r'^The .+ is not something that can be activated\.', r"^The .+ isn't open, so you can't see inside\.",
             r'^That container is empty, so there are no items to mix\.', r'^There is only one thing \(']


def rejection_reason(feedback):
    return next((pattern for pattern in _REFUSALS if re.search(pattern, str(feedback or '').strip(), re.I)), None)


def compact_state(state, rep, state_t, observation):
    return {k:v for k,v in state.items() if k not in {"objects", "canonical_tree"}} | {
        "node_key": state["key"], "state_t": state_t, "visit_id": f"r{rep}:t{state_t}",
        "observation": observation, "interaction": state.get("interaction", codec.interaction_state(observation))}


def build_graph(records, gamefile):
    nodes, edges, edge_index, trajectories, counts = {}, [], {}, [], Counter()
    source_by_rep = {}
    def add_state(state, member):
        key = state["key"]
        if key is None:
            raise ValueError("Unmatchable state cannot enter graph")
        if key not in nodes:
            nodes[key] = {"id": f"N{len(nodes)}", "key": key, "physical_hash": state["physical_hash"],
                          "control": state["control"], "room": state.get("room"), "members": [], "scores": [], "ticks": [],
                          "observations": [], "entry_observations": [], "interactions": [], "audit": [], "entry_effects": [],
                          "start": member.get("state_t") == 0 and not member.get("rolled_back_attempt"), "won": False}
        node = nodes[key]
        node["members"].append(member)
        node["start"] |= member.get("state_t") == 0 and not member.get("rolled_back_attempt")
        node["scores"].append(state.get("score"))
        node["ticks"].append(state.get("tick"))
        return node
    # Allocate committed states in source order. Node keys exclude hidden
    # observation history.
    for record in records:
        rep = record.get("rep", record.get("seed"))
        source_by_rep[rep] = record
        states, steps = record["states"], record["steps"]
        if len(states) != len(steps)+1:
            raise ValueError("Trusted committed path must have T+1 states")
        visits = []
        for t, state in enumerate(states):
            if "objects" in state:
                codec.normalized_snapshot(state)
            observation = state.get("observation", steps[t].get("observation_before", steps[t].get("source_observation", "")) if t < len(steps) else steps[-1].get("feedback", "") if steps else "")
            visit = compact_state(state, rep, t, observation)
            visits.append(visit)
            add_state(state, {"rep": rep, "state_t": t, "step_index": t,
                              "visit_id": visit["visit_id"], "snapshot_ref": state.get("snapshot_ref"), "rolled_back_attempt": False})
        replayed_won = bool(states and isinstance(states[-1].get("score"), (int,float)) and states[-1]["score"] >= 100)
        trusted_complete = record.get("trusted_complete", True)
        trajectories.append({"rep": rep, "source_path": record.get("source_path"), "source_sha256": record.get("source_sha256"),
                             "source_won": record.get("source_won", record.get("won", False)),
                             "trusted_complete": trusted_complete, "replayed_won": replayed_won,
                             "trusted_success": replayed_won and trusted_complete,
                             "won": replayed_won and trusted_complete,
                             "states": visits, "steps": [], "source_attempt_summary": record.get("source_attempt_summary", {}),
                             "replay_failures": record.get("replay_failures", [])})
    def add_transition(step, before, after, rep, index, prior, *, attempt=False):
        source, target = before["key"], after["key"]
        changed = source != target
        rejection = step.get("rejection") or rejection_reason(step.get("feedback"))
        semantic = step.get("semantic_action", step.get("action"))
        kind = codec.classify_transition(step.get("action"), step.get("feedback"), changed, rejection, step.get("format_error"))
        effect = step.get("effects")
        if effect is None:
            effect = codec.meaningful_changes(before["objects"], after["objects"]) if changed and "objects" in before and "objects" in after else []
        event = {**step, **kind, "rep": rep, "step_index": index,
                 "decision_index": step.get("decision_index", index), "accepted_step_index": None if attempt else index,
                 "source": source, "target": target, "observation_before": step.get("observation_before", step.get("source_observation", before.get("observation", ""))),
                 "semantic_action": semantic, "rejection": rejection, "changed": changed,
                 "physical_changed": before["physical_hash"] != after["physical_hash"], "control_changed": before["control"] != after["control"],
                 "tick_before": before.get("tick"), "tick_after": after.get("tick"), "effects": effect,
                 "prior_observation_evidence": list(prior), "rolled_back_attempt": attempt,
                 "positive_candidate_allowed": not attempt and not rejection and not step.get("format_error") and not step.get("source_terminal_failure", False)}
        counts["decision_records"] += 1
        counts["attempt_branch_records" if attempt else "committed_decision_records"] += 1
        if changed:
            ekey = (source, target, semantic)
            if ekey not in edge_index:
                edge_index[ekey] = len(edges)
                edges.append({"id": f"E{len(edges)}", "source": source, "target": target, "action": semantic,
                              "members": [], "effects": effect, "witnesses": [], "passive_during_rejection": False,
                              "physical_changed": event["physical_changed"], "control_changed": event["control_changed"],
                              "observation_with_change": kind["is_observation"], "non_rejected_committed_members": 0})
            edge = edges[edge_index[ekey]]
            member = {"rep": rep, "step_index": index, "decision_index": event["decision_index"],
                      "visit_id": f"r{rep}:a{event['decision_index']}" if attempt else f"r{rep}:t{index}", "rolled_back_attempt": attempt}
            edge["members"].append(member)
            edge["witnesses"].append({**member, "prior_observation_evidence": list(prior), "effects": effect,
                                       "feedback": event.get("feedback"), "rejection": rejection,
                                       "tick_before": before.get("tick"), "tick_after": after.get("tick"), "decision_cost": 1})
            edge["passive_during_rejection"] |= bool(rejection)
            edge["non_rejected_committed_members"] += int(event["positive_candidate_allowed"])
            event["edge"] = edge["id"]
            nodes[target]["entry_effects"].append({"edge": edge["id"], "rep": rep, "step_index": index, "effects": effect})
            if kind["is_observation"] and not rejection:
                nodes[target]["entry_observations"].append({"rep": rep, "step_index": index, "action": step["action"], "feedback": step["feedback"], "edge": edge["id"]})
                prior.append({"node_key": target, "entry_edge": edge["id"], "rep": rep, "step_index": index})
        elif kind["kind"] == "binding":
            nodes[source]["interactions"].append(event)
        elif kind["kind"] == "internal_observation":
            node = nodes[source]
            existing = next((o for o in node["observations"] if o["action"] == step["action"] and codec.norm(o["feedback"]) == codec.norm(step["feedback"])), None)
            if existing is None:
                existing = {"id": f"O{len(node['observations'])}", "action": step["action"], "feedback": step["feedback"], "members": []}
                node["observations"].append(existing)
            existing["members"].append({"rep": rep, "step_index": index, "decision_index": event["decision_index"], "decision_cost": 1})
            prior.append({"node_key": source, "observation": existing["id"], "rep": rep, "step_index": index})
        else:
            nodes[source]["audit"].append(event)
        counts[kind["kind"]] += 1
        return event
    for trajectory in trajectories:
        rep = trajectory["rep"]
        record = source_by_rep[rep]
        prior = []
        for index, step in enumerate(record["steps"]):
            trajectory["steps"].append(add_transition(step, record["states"][index], record["states"][index+1], rep, index, prior))
    attempts = []
    for trajectory in trajectories:
        rep, record = trajectory["rep"], source_by_rep[trajectory["rep"]]
        for branch in record.get("attempt_branches", []):
            before, after = branch["before"], branch["after"]
            index = branch["step"]["decision_index"]
            for tag, state in [("before", before), ("after", after)]:
                add_state(state, {"rep": rep, "state_t": None, "step_index": None, "visit_id": f"r{rep}:a{index}:{tag}",
                                  "snapshot_ref": state.get("snapshot_ref"), "rolled_back_attempt": True})
            parent_index = branch.get("accepted_prefix_length", 0)
            prior = trajectory["steps"][parent_index]["prior_observation_evidence"] if parent_index < len(trajectory["steps"]) else []
            attempts.append(add_transition(branch["step"], before, after, rep, index, list(prior), attempt=True))
    for node in nodes.values():
        node["scores"] = sorted({x for x in node["scores"] if x is not None})
        ticks = [x for x in node.pop("ticks") if x is not None]
        node["tick_range"] = [min(ticks), max(ticks)] if ticks else None
        node["endpoints"] = []
    for trajectory in trajectories:
        if trajectory["states"]:
            node = nodes[trajectory["states"][-1]["node_key"]]
            node["won"] |= trajectory["trusted_success"]
            node["endpoints"].append({"rep": trajectory["rep"], "source_won": trajectory["source_won"],
                                       "replayed_won": trajectory["replayed_won"], "trusted_complete": trajectory["trusted_complete"],
                                       "trusted_success": trajectory["trusted_success"]})
    return {"schema_version": "physical_graph_v1", "gamefile": gamefile, "codec_version": codec.CODEC_VERSION,
            "nodes": nodes, "edges": edges, "trajectories": trajectories, "attempt_branches": attempts,
            "stats": {**counts, "nodes": len(nodes), "edges": len(edges), "trajectories": len(trajectories),
                      "committed_state_visits": sum(len(t["states"]) for t in trajectories),
                      "trusted_success_reps": sum(t["won"] for t in trajectories)},
            "semantics": {"physical_key_not_h5_text": True, "interaction_mode_separate_visit_gate_required": True,
                          "all_internal_observations_have_decision_cost": True, "rolled_back_attempts_not_contiguous_success_routes": True,
                          "raw_trees_stored_only_in_compressed_snapshot_files": True}}


def save_graph(path, graph):
    path = Path(path)
    body = json.dumps(graph, ensure_ascii=False, separators=(",", ":")).encode()
    data = gzip.compress(body, compresslevel=3, mtime=0)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "compressed_bytes": len(data), "json_bytes": len(body)}
