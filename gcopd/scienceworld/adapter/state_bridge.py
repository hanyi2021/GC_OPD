"""Private environment-state capture and strict reference-input boundaries."""
from copy import deepcopy
import gzip
import importlib
import json
import os
from pathlib import Path
import sys
import time


def get_codec():
    from gcopd.scienceworld import state
    return state


def matching_descriptor(snapshot):
    """Only state identity/control metadata; never full objects or hidden values."""
    keys = ["key", "physical_hash", "control", "room", "tick", "interaction",
            "capture_ok", "matchable", "capture_error", "codec_version", "key_scope",
            "hidden_simulator_state_certified", "object_count", "visit_index"]
    result = {key: deepcopy(snapshot[key]) for key in keys if key in snapshot}
    interaction_known = snapshot.get("interaction", {}).get("mode") in {"normal", "pending_menu"}
    valid = bool(snapshot.get("capture_ok") and snapshot.get("matchable") and snapshot.get("key") and interaction_known)
    result["capture_ok"] = bool(snapshot.get("capture_ok"))
    result["matchable"] = valid
    if not valid:
        result["key"] = None
        result["physical_hash"] = None
    assert "objects" not in result and "score" not in result and "observation" not in result
    return result


def reference_past(turns):
    """Full already-executed public history; no state snapshots or future output."""
    keys = ["action", "feedback", "format_error", "env_rejection"]
    return [{key: deepcopy(turn.get(key)) for key in keys} for turn in turns]


class PhysicalStateTracker:
    def __init__(self, env):
        self.env = env
        self.codec = get_codec()
        self.focus_reference = None
        self.current = None
        self.last_environment_observation = ""
        self.snapshots = []
        self.capture_seconds = 0.0

    def _capture(self, observation, info, env_step_called):
        started = time.perf_counter()
        previous_interaction = self.current.get("interaction") if self.current else None
        snapshot = self.codec.capture_state(
            self.env, focus_reference=self.focus_reference, observation=observation,
            info=info, include_objects=True, previous_interaction=previous_interaction,
            env_step_called=env_step_called,
        )
        snapshot = deepcopy(snapshot)
        snapshot["visit_index"] = len(self.snapshots)
        snapshot["environment_step_was_called"] = bool(env_step_called)
        self.capture_seconds += time.perf_counter() - started
        self.snapshots.append(snapshot)
        self.current = snapshot
        return matching_descriptor(snapshot)

    def start(self, observation, info):
        self.last_environment_observation = observation
        return self._capture(observation, info, True)

    def before(self):
        if self.current is None:
            raise RuntimeError("Physical tracker must be started after reset")
        return matching_descriptor(self.current)

    def after_decision(self, observation, info, action, *, env_step_called,
                       rejection=None, format_error=None):
        before = self.before()
        if env_step_called:
            self.focus_reference = self.codec.update_focus(self.focus_reference, observation)
            self.last_environment_observation = observation
        after = self._capture(observation, info, env_step_called)
        if (env_step_called and rejection and before.get("interaction", {}).get("mode") in {"pending_menu", "unknown"}
                and self.current.get("interaction", {}).get("mode") == "normal"):
            self.current["interaction"] = {"mode": "unknown", "options": {},
                "source": "rejection_after_pending_interaction_without_fresh_menu",
                "actual_hidden_parser_state_certified": False}
            self.current["matchable"] = False
            after = matching_descriptor(self.current)
        known = before["matchable"] and after["matchable"]
        changed = before["key"] != after["key"] if known else None
        if known:
            transition = self.codec.classify_transition(action, observation, changed,
                                                        rejection=rejection, format_error=format_error)
        else:
            transition = {"kind": "state_capture_unknown", "decision_cost": 1,
                          "is_observation": None, "is_binding": None}
        transition.update({
            "state_key_changed": changed,
            "physical_changed": before["physical_hash"] != after["physical_hash"] if known else None,
            "control_changed": before["control"] != after["control"] if known else None,
            "environment_action_executed": bool(env_step_called),
            "before_visit_index": before["visit_index"], "after_visit_index": after["visit_index"],
            "both_states_captured": bool(known),
            "parser_interaction_changed": (
                (before.get("interaction", {}).get("mode"), before.get("interaction", {}).get("options")) !=
                (after.get("interaction", {}).get("mode"), after.get("interaction", {}).get("options"))),
            "cross_trajectory_equivalence_certified": False,
        })
        return after, transition

    def write_private_audit(self, directory, episode_id, gamefile, rep, policy_version):
        """Objects are stored only here, never in the reference function's inputs."""
        if os.environ.get("SW_PHYSICAL_SAVE_STATES", "1") != "1":
            return None
        destination = Path(directory) / "physical_states"
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / f"{episode_id}.json.gz"
        temporary = destination / f"{episode_id}.tmp.json.gz"
        payload = {"gamefile": gamefile, "rep": rep, "policy_version": policy_version,
                   "scope": "Environment state diagnostics for graph indexing; excluded from model prompts",
                   "capture_seconds": self.capture_seconds, "states": self.snapshots}
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=1) as handle:
            json.dump(payload, handle, ensure_ascii=False)
        temporary.replace(path)
        return str(path)
