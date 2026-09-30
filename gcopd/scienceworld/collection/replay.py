"""Replay Qwen3 source executions into physical snapshots and visit graphs.

Refused attempts are replayed as independent rollback branches: restore and
verify the accepted prefix, execute the attempt, then discard that branch."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from copy import deepcopy
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

import gcopd.scienceworld.state as codec
import gcopd.scienceworld.observation as observation_equivalence
from gcopd.scienceworld.graph import build_graph, rejection_reason, save_graph

HERE = Path(__file__).resolve().parent



def dump(path, value):
    temporary = Path(str(path)+f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def equivalent(actual, expected):
    return observation_equivalence.equivalent(actual, expected)


def action_from_output(output):
    if str(output).count("<action>") != 1 or str(output).count("</action>") != 1:
        raise ValueError("Recorded attempt does not contain one complete action")
    match = re.fullmatch(r"\s*(?:<thought>.*?</thought>\s*)?<action>(.*?)</action>\s*", str(output), re.S)
    if not match:
        raise ValueError("Recorded attempted reply has invalid surrounding format")
    return match.group(1).strip().lower()


def resolve_action(row, current_observation, restoring=False):
    action = row["action"]
    if action.isdigit():
        original, current = codec.parse_menu(row["observation_before"]), codec.parse_menu(current_observation)
        if action not in original:
            raise ValueError("Recorded numeric action has no recorded menu binding")
        semantic = row.get("semantic_action", original[action])
        if codec.norm(semantic) != codec.norm(original[action]):
            raise ValueError("Recorded semantic_action disagrees with recorded numeric menu")
        if original == current:
            return action, semantic, "identical_complete_menu_map"
        matches = [key for key,value in current.items() if codec.norm(value) == codec.norm(semantic)]
        if len(matches) != 1:
            raise ValueError("Numeric menu changed and has no unique semantic rebind")
        return matches[0], semantic, "unique_actual_menu_rebind"
    if restoring:
        # Restore recorded action bindings. A feedback, score or tick mismatch
        # stops replay.
        feedback = row["feedback"]
        if action.startswith("pick up "):
            match = re.fullmatch(r"You move (?:the )?(.+?) to (?:the )?inventory\.", feedback.strip())
            if match and re.search(r"\b\d+\b", match.group(1)):
                return "pick up "+match.group(1).lower(), row.get("semantic_action",action), "source_recorded_numbered_restore_binding"
        elif action.startswith("move "):
            match = re.fullmatch(r"You move (?:the )?(.+?) to (?:the )?(.+?)\.", feedback.strip())
            if match and re.search(r"\b\d+\b", match.group(1)):
                return "move "+match.group(1).lower()+" to "+match.group(2).lower(), row.get("semantic_action",action), "source_recorded_numbered_restore_binding"
        elif action.startswith("focus on "):
            match = re.fullmatch(r"You focus on (?:the )?(.+?)\.", feedback.strip())
            if match and re.search(r"\b\d+\b", match.group(1)):
                return "focus on "+match.group(1).lower(), row.get("semantic_action",action), "source_recorded_numbered_restore_binding"
    return action, row.get("semantic_action", action), "literal_recorded_action"


def check_step(row, before_obs, before_tick, after_obs, info, after_tick, done=None):
    checks = {"observation_before": equivalent(before_obs,row["observation_before"]),
              "feedback": equivalent(after_obs,row["feedback"]),
              "score": info.get("score") == row.get("score"),
              "tick_before": before_tick == row.get("tick_before") if row.get("tick_before") is not None else None,
              "tick_after": after_tick == row.get("tick_after") if row.get("tick_after") is not None else None,
              "done": bool(done) == bool(row["done"]) if done is not None and "done" in row else None}
    return checks, checks["observation_before"]["equal"] and checks["feedback"]["equal"] and checks["score"] and checks["tick_before"] is not False and checks["tick_after"] is not False and checks["done"] is not False


def source_teacher_matches(protocol):
    """Use explicit model identity; preserve basename compatibility for legacy records."""
    if protocol.get("model_id") is not None:
        return protocol["model_id"] == "Qwen/Qwen3-32B"
    return str(protocol.get("model_path", "")).rstrip("/").endswith("/Qwen3-32B")


def collect_rep(gamefile, rep, out_dir, source_root, runtime_root, expected_source_sha256=None):
    started = time.time()
    out_dir, source_root = Path(out_dir), Path(source_root)
    task_hash = hashlib.sha256(gamefile.encode()).hexdigest()[:16]
    path = source_root/f"train_{task_hash}_r{rep:02d}.json"
    if not path.is_file():
        return {"rep": rep, "source_path": str(path), "source_missing": True, "states": [], "steps": [], "attempt_branches": [], "replay_failures": [{"kind":"missing_source"}]}
    content = path.read_bytes()
    if expected_source_sha256 is not None and hashlib.sha256(content).hexdigest() != expected_source_sha256:
        raise ValueError("Source hash differs from fixed production manifest")
    data = json.loads(content)
    assert data["gamefile"] == gamefile and data["split"] == "train" and int(data["gen_seed"]) == rep
    assert source_teacher_matches(data.get("sampling_protocol", {})), "Unexpected source teacher identity"
    record = {"rep": rep, "source_path": str(path), "source_sha256": hashlib.sha256(content).hexdigest(),
              "source_won": bool(data["won"]), "source_end_reason": data.get("end_reason"),
              "source_accepted_steps": len(data["steps"]), "states": [], "steps": [], "attempt_branches": [],
              "attempt_audit": [], "replay_failures": [], "trusted_complete": False,
              "source_attempt_summary": {"raw_model_attempts": len(data.get("attempts",[])), "types": dict(Counter(a.get("type") for a in data.get("attempts",[]))),
                                         "accepted_decisions": len(data["steps"]), "source_max_accepted_steps":30,"source_retries_are_not_free_student_decisions":True}}
    snapshot_path = out_dir/f"snapshots_rep{rep:02d}.jsonl.gz"
    snapshot_count, env_calls, restore_calls, env = 0, 0, 0, None
    from gcopd.common import cleanup as public
    def save_snapshot(state, kind):
        nonlocal snapshot_count
        snapshot_count += 1
        state["snapshot_ref"] = {"file": str(snapshot_path), "line": snapshot_count, "kind": kind}
        writer.write(json.dumps(state, ensure_ascii=False, separators=(",",":"))+"\n")
        return {k:v for k,v in state.items() if k != "objects"}
    try:
        os.environ["_JAVA_OPTIONS"] = "-XX:ActiveProcessorCount=2 -Xms64m -Xmx768m"
        sys.path.insert(0, str(runtime_root))
        from scienceworld import ScienceWorldEnv
        env=ScienceWorldEnv.__new__(ScienceWorldEnv);ScienceWorldEnv.__init__(env,"",envStepLimit=200)
        task, variation, simplification = gamefile.split("|")
        env.load(task,int(variation),simplification)
        observation, info = env.reset()
        focus, previous_interaction = None, None
        accepted_attempts = {}
        for ai,attempt in enumerate(data.get("attempts",[])):
            if attempt.get("type") == "accepted":
                accepted_attempts.setdefault(int(attempt["step"])-1, []).append((ai,attempt))
        with gzip.open(snapshot_path,"wt",encoding="utf8",compresslevel=3) as writer:
            if not data["steps"]:
                record["replay_failures"].append({"kind":"no_recorded_initial_observation_in_zero_accepted_source"})
            elif not equivalent(observation,data["steps"][0]["observation_before"])["equal"]:
                record["replay_failures"].append({"kind":"initial_observation_mismatch","actual":observation,"expected":data["steps"][0]["observation_before"]})
            else:
                current = codec.capture_state(env,focus,observation,info["score"])
                if not current["capture_ok"]:
                    raise RuntimeError(current["capture_error"])
                current["verified"] = True
                record["states"].append(save_snapshot(current,"committed_initial"))
                for index,row in enumerate(data["steps"]):
                    entries = accepted_attempts.get(index,[])
                    if len(entries) != 1 or entries[0][1].get("raw_output") != row.get("raw_output"):
                        record["replay_failures"].append({"kind":"accepted_attempt_alignment_mismatch","step_index":index})
                        break
                    if not equivalent(observation,row["observation_before"])["equal"]:
                        record["replay_failures"].append({"kind":"committed_observation_before_mismatch","step_index":index,"actual":observation,"expected":row["observation_before"]})
                        break
                    try:
                        action, semantic, binding = resolve_action(row,observation)
                    except ValueError as exc:
                        record["replay_failures"].append({"kind":"committed_numeric_binding_mismatch","step_index":index,"error":str(exc)})
                        break
                    before_obs, before_tick = observation,env.get_num_moves()
                    observation,_,done,info = env.step(action);env_calls += 1
                    checks, verified = check_step(row,before_obs,before_tick,observation,info,env.get_num_moves(),done)
                    if not verified:
                        record["replay_failures"].append({"kind":"committed_step_mismatch","step_index":index,"checks":checks,"actual_feedback":observation,"expected_feedback":row["feedback"]})
                        break
                    focus = codec.update_focus(focus,observation)
                    after = codec.capture_state(env,focus,observation,info["score"],previous_interaction=current["interaction"])
                    if not after["capture_ok"]:
                        record["replay_failures"].append({"kind":"capture_failed","step_index":index,"error":after["capture_error"]})
                        break
                    after["verified"] = True
                    effect = codec.meaningful_changes(current["objects"],after["objects"])
                    record["steps"].append({"step_index":index,"decision_index":entries[0][0],"source_recorded_step":row["step"],
                        "action":row["action"],"semantic_action":semantic,"executed_action":action,"binding_strategy":binding,
                        "observation_before":row["observation_before"],"feedback":row["feedback"],"replay_feedback":observation,
                        "source_score":row["score"],"score":info["score"],"done":bool(done),
                        "source_terminal_failure":bool(done and info["score"]<100),"effects":effect,"verification":checks,
                        "source_raw_output_ref":{"file":str(path),"steps_index":index},"env_step_called":True,"decision_cost":1})
                    record["states"].append(save_snapshot(after,"committed_after"))
                    current = after
                    if done and index+1 < len(data["steps"]):
                        record["replay_failures"].append({"kind":"terminal_before_source_end","step_index":index})
                        break
                original_trusted_count = len(record["steps"])
                cut_at = original_trusted_count
                # Source run.py restores exactly the accepted prefix after every
                # rejected attempt. Recreate each rejected attempt as a separate
                # branch; never connect its successor to the next accepted step.
                for ai,attempt in enumerate(data.get("attempts",[])):
                    if attempt.get("type") == "accepted":
                        continue
                    prefix_length = int(attempt.get("step",0))-1
                    audit = {"attempt_index":ai,"source_recorded_step":attempt.get("step"),"attempt":attempt.get("attempt"),
                             "type":attempt.get("type"),"reason":attempt.get("reason"),"accepted_prefix_length":prefix_length,
                             "source_ref":{"file":str(path),"attempts_index":ai}}
                    record["attempt_audit"].append(audit)
                    if prefix_length < 0 or prefix_length > cut_at or not record["states"]:
                        audit["status"]="outside_trusted_prefix";continue
                    if attempt.get("type") == "format_invalid":
                        before=record["states"][prefix_length]
                        step={"action":None,"semantic_action":None,"feedback":None,"format_error":attempt.get("reason") or "recorded_format_invalid",
                              "decision_index":ai,"source_attempt_index":ai,"env_step_called":False,"effects":[],"source_ref":audit["source_ref"]}
                        record["attempt_branches"].append({"before":before,"after":before,"step":step,"accepted_prefix_length":prefix_length})
                        audit["status"]="verified_no_environment_step_by_source_protocol";continue
                    if attempt.get("type") != "environment_rejection":
                        audit["status"]="uncompleted_or_unknown_attempt_type";cut_at=min(cut_at,prefix_length);continue
                    try:
                        env.load(task,int(variation),simplification);observation,info=env.reset();restore_calls += 1;focus=None
                        interaction=codec.interaction_state(observation)
                        for past in data["steps"][:prefix_length]:
                            if not equivalent(observation,past["observation_before"])["equal"]:
                                raise ValueError("restore prefix observation mismatch: "+json.dumps({"actual":observation,"expected":past["observation_before"],"comparison":equivalent(observation,past["observation_before"])}))
                            restored,_,_=resolve_action(past,observation,restoring=True)
                            before_obs,before_tick=observation,env.get_num_moves()
                            observation,_,done,info=env.step(restored);env_calls += 1
                            checks,ok=check_step(past,before_obs,before_tick,observation,info,env.get_num_moves(),done)
                            if not ok:raise ValueError("restore prefix feedback/score/tick mismatch: "+json.dumps(checks))
                            focus=codec.update_focus(focus,observation)
                            interaction=codec.interaction_state(observation,interaction,True)
                        before=codec.capture_state(env,focus,observation,info["score"],previous_interaction=interaction,env_step_called=False)
                        expected=record["states"][prefix_length]
                        if not before["capture_ok"] or before["key"]!=expected["key"] or before["score"]!=expected["score"]:
                            raise ValueError("restored physical/control key or score differs from committed-prefix snapshot")
                        raw_action=action_from_output(attempt["raw_output"])
                        semantic=codec.parse_menu(observation).get(raw_action,raw_action)
                        tick_before=env.get_num_moves()
                        observation,_,done,info=env.step(raw_action);env_calls += 1
                        matches=equivalent(observation,attempt.get("feedback",""))["equal"]
                        ticks=(attempt.get("tick_before") in [None,tick_before] and attempt.get("tick_after") in [None,env.get_num_moves()])
                        if not matches or not ticks:raise ValueError("rejected attempt feedback/tick mismatch: "+json.dumps({"actual":observation,"expected":attempt.get("feedback", ""),"comparison":equivalent(observation,attempt.get("feedback", "")),"ticks_equal":ticks}))
                        focus=codec.update_focus(focus,observation)
                        after=codec.capture_state(env,focus,observation,info["score"],previous_interaction=before["interaction"])
                        if not after["capture_ok"]:raise ValueError(after["capture_error"])
                        effect=codec.meaningful_changes(before["objects"],after["objects"])
                        bcompact=save_snapshot(before,"rejected_attempt_before")
                        acompact=save_snapshot(after,"rejected_attempt_after")
                        step={"action":raw_action,"semantic_action":semantic,"executed_action":raw_action,"feedback":attempt["feedback"],
                              "replay_feedback":observation,"observation_before":before["observation"],"rejection":attempt.get("reason") or rejection_reason(observation),
                              "decision_index":ai,"source_attempt_index":ai,"env_step_called":True,"effects":effect,
                              "source_ref":audit["source_ref"],"done":bool(done),"score":info["score"],"source_score_not_recorded_for_rejected_attempt":True}
                        record["attempt_branches"].append({"before":bcompact,"after":acompact,"step":step,"accepted_prefix_length":prefix_length})
                        audit["status"]="verified_independent_rolled_back_branch"
                    except Exception as exc:
                        audit.update(status="attempt_replay_mismatch",error=str(exc))
                        record["replay_failures"].append({"kind":"attempt_or_restore_mismatch","attempt_index":ai,"accepted_prefix_length":prefix_length,"error":str(exc)})
                        cut_at=min(cut_at,prefix_length)
                if cut_at < original_trusted_count:
                    record["steps"]=record["steps"][:cut_at]
                    record["states"]=record["states"][:cut_at+1]
                    record["attempt_branches"]=[b for b in record["attempt_branches"] if b["accepted_prefix_length"]<=cut_at]
                record["trusted_complete"]=len(record["steps"])==len(data["steps"]) and not record["replay_failures"]
    except Exception:
        record["replay_failures"].append({"kind":"technical_error","error":traceback.format_exc()})
    finally:
        record["cleanup"]=public.close_owned_environment(env)
    record["snapshot_file"] = str(snapshot_path)
    record["snapshot_count"] = snapshot_count
    record["snapshot_compressed_bytes"] = snapshot_path.stat().st_size if snapshot_path.exists() else 0
    record["source_input_bytes"] = len(content)
    record["env_step_calls_including_technical_restore"] = env_calls
    record["technical_restore_resets"] = restore_calls
    record["seconds"] = time.time()-started
    return record
