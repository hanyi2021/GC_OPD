"""The user's four cases, with complete action/observation evidence and no thoughts.

Student terminal outcomes and all actual actions/feedback may be used after
rollout completion. Original response IDs are never rewritten. No original
student/source reply or thought is serialized into reference text.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from gcopd.webshop.references import selection as core

ROOT = Path(__file__).resolve().parent
PHYS = ROOT.parents[2]
DEFAULT_CATALOG = Path(os.environ.get("WS_PHYSICAL_GRAPH_ROOT", str(ROOT / "catalog")))
REMINDER = (
    "CURRENT RESPONSE REQUIREMENT: Act as the WebShop shopping agent in the current student task and page state. "
    "Return exactly one <action>...</action>. If you include reasoning, put ALL of it inside one "
    "<thought>...</thought> before that action. Do not put narration outside these tags. "
    "The source above is historical reference, not the student's current state or a request to review the source."
)


class ReferenceBudgetError(ValueError):
    pass


class ReferenceIntegrityError(ValueError):
    pass


def source_id(trace):
    return str(trace.get("source_kind") or "webshop_teacher_k16") + ":" + str(trace["rep"])


@lru_cache(maxsize=128)
def _source_document(path, mtime, size, digest):
    data = Path(path).read_bytes()
    if digest and hashlib.sha256(data).hexdigest() != digest:
        raise ReferenceIntegrityError("Source document hash differs from its recorded provenance")
    outer = json.loads(data)
    if isinstance(outer.get("runs"), list):
        if len(outer["runs"]) != 1:
            raise ReferenceIntegrityError("Expected one actual planner replay per published source")
        raw = dict(outer["runs"][0])
        raw.setdefault("won", raw.get("source_won", False))
        raw["source_kind"] = outer.get("source_kind", raw.get("source_kind", "planner_actual_replay"))
        return raw
    return outer


def source_document(trace):
    path = Path(trace["source_path"])
    stat = path.stat()
    return _source_document(str(path), stat.st_mtime_ns, stat.st_size, trace.get("source_sha256"))


def proposed_action(attempt):
    if isinstance(attempt.get("action"), str):
        return attempt["action"]
    # Only a source action is extracted if its attempt record stores it inside
    # action tags. The surrounding source thought/reply is never retained.
    matches = re.findall(r"<action>(.*?)</action>", str(attempt.get("raw_output", "")), re.S)
    return matches[0].strip() if len(matches) == 1 else None


def events_from_source(trace):
    raw = source_document(trace)
    steps = raw.get("steps")
    if not isinstance(steps, list):
        raise ReferenceIntegrityError("Original source has no complete event stream")
    attempts = raw.get("attempts", [])
    events = []
    seen = set()
    if attempts:
        if not all(isinstance(a.get("step"), int) for a in attempts):
            raise ReferenceIntegrityError("Cannot align source attempts without recorded step slots")
        for attempt in attempts:
            j = attempt["step"] - 1
            committed = attempt.get("type") == "accepted"
            if committed and not 0 <= j < len(steps):
                raise ReferenceIntegrityError("Source accepted attempt points outside complete source steps")
            row = steps[j] if committed else attempt
            if committed:
                seen.add(j)
            events.append({"source_step": j, "kind": "committed" if committed else attempt.get("type", "noncommitted"),
                           "committed": committed, "action": row.get("action") if committed else proposed_action(attempt),
                           "semantic_action": row.get("semantic_action") if committed else None,
                           "observation_before": row.get("observation_before") if committed else None,
                           "feedback": row.get("feedback"), "reason": attempt.get("reason") if not committed else None})
        # WebShop protocol: a format-error round consumes the decision slot without any accepted
        # action (no retry). Such a recorded round is covered by its own non-accepted attempt.
        covered = set(seen)
        for attempt in attempts:
            j = attempt["step"] - 1
            if attempt.get("type") != "accepted" and 0 <= j < len(steps) and (
                    steps[j].get("format_error") or steps[j].get("rejection") or steps[j].get("executed") is False):
                covered.add(j)
        if covered != set(range(len(steps))):
            raise ReferenceIntegrityError("Source attempt stream does not cover every committed source step")
    else:
        for j, row in enumerate(steps):
            events.append({"source_step": j, "kind": "committed", "committed": True,
                           "action": row.get("action"), "semantic_action": row.get("semantic_action"),
                           "observation_before": row.get("observation_before"), "feedback": row.get("feedback"),
                           "reason": row.get("format_error", row.get("rejection"))})
    initial = raw.get("initial_observation")
    if initial is None:
        initial = steps[0].get("observation_before") if steps else None
    return raw, events, initial


def raw_failed_fallback(catalog_root, gamefile, graph):
    """For an all-failed source task, retain a complete recorded failure."""
    manifest_path = Path(catalog_root) / "MANIFEST.json"
    candidates = []
    if manifest_path.is_file():
        task = next((x for x in core.read_json(manifest_path).get("tasks", []) if x["gamefile"] == gamefile), {})
        for item in task.get("source_files", []):
            if item.get("source_won") is False and not item.get("expected_missing"):
                candidates.append({"rep": item["rep"], "source_path": str(Path(catalog_root)/item["path"]),
                                   "source_sha256": item.get("sha256"), "source_kind": item.get("source_kind", "original_qwen32b_k16")})
    if graph:
        candidates += [t for t in graph["trajectories"] if t.get("source_won") is False]
    for candidate in sorted(candidates, key=lambda t: t["rep"]):
        try:
            raw = source_document(candidate)
        except (OSError, ValueError, KeyError):
            continue
        if raw.get("won", raw.get("source_won")) is False and isinstance(raw.get("steps"), list):
            return {**candidate, "steps": raw["steps"], "states": [],
                    "trusted_complete": False, "trusted_success": False, "won": False, "source_won": False,
                    "raw_recorded_failure_reference": True,
                    "replay_limit": "The full source record reports failure; full replay or a shared physical anchor is not certified."}
    return None


class Observations:
    def __init__(self, compressed):
        self.compressed = compressed
        self.seen = {}
        self.references = 0

    def show(self, text, owner):
        if text is None:
            return "[No observation/feedback was recorded for this event.]"
        text = str(text)
        if not self.compressed:
            return text
        if text in self.seen:
            self.references += 1
            label, first_owner = self.seen[text]
            return f"[Exact same recorded observation text as {label} ({first_owner}) above; no state identity or shared knowledge is implied.]"
        label = "OBS" + str(len(self.seen))
        self.seen[text] = label, owner
        return f"[{label}: {owner}]\n{text}"


def render_student(trajectory, t, obs):
    won = trajectory["won"]; sc = trajectory.get("final_score", 1.0 if won else 0.0)
    lines = ["STUDENT'S COMPLETE ACTUAL ACTION/OBSERVATION EXECUTION",
             ("This complete student execution SUCCEEDED: every task requirement was satisfied." if won else "This complete student execution FAILED to satisfy every task requirement. This does not mean every earlier action was wrong."),
             f"Actual student events: {len(trajectory['steps'])}; current scored decision: {t + 1}.",
             "All recorded actions and feedback, including later events and the final outcome, are supplied after rollout completion. No historical thought or raw reply is supplied.",
             "Student initial observation:", obs.show(trajectory.get("initial_obs"), "student initial")]
    for i, row in enumerate(trajectory["steps"]):
        lines.append(f"Student event {i + 1}" + (" [CURRENT SCORED DECISION]" if i == t else ""))
        before = row.get("current_obs")
        if before is not None and (i == 0 or core.norm(before) != core.norm(trajectory["steps"][i - 1].get("feedback"))):
            lines.extend(["Actual input observation:", obs.show(before, f"student event {i + 1} before")])
        lines.append("Recorded action: " + (str(row["action"]) if row.get("action") is not None else "[no valid parsed environment action]"))
        if row.get("semantic_action") and row["semantic_action"] != row.get("action"):
            lines.append("Recorded action binding: " + str(row["semantic_action"]))
        if row.get("format_error"):
            lines.append("Actual format failure: " + str(row["format_error"]))
        if row.get("env_rejection") or row.get("rejection"):
            lines.append("Actual environment rejection: " + str(row.get("env_rejection") or row.get("rejection")))
        lines.extend(["Actual feedback:", obs.show(row.get("feedback"), f"student event {i + 1} feedback")])
    lines.append("END OF COMPLETE STUDENT EXECUTION: " + ("SUCCESS" if won else "FAILURE"))
    return "\n".join(lines)


def render_source(trace, anchor, role, obs):
    raw, events, initial = events_from_source(trace)
    sid = source_id(trace)
    outcome = raw.get("won", raw.get("source_won", trace.get("source_won", False))) is True
    ssc = core.source_score(trace); stu = core._CTX.get("student_score")
    trusted = core.trusted(trace)
    lines = [f"COMPLETE SOURCE ACTION/OBSERVATION EXECUTION: {sid}",
             f"Original file: {Path(trace['source_path']).name}; role: {role}.",
             "The COMPLETE original source record ended in " + ("SUCCESS: every requirement was satisfied." if outcome else "FAILURE: not every requirement was satisfied."),
             f"Complete source replay trusted: {trusted}. Displaying ALL {len(events)} recorded events and all {len(raw['steps'])} committed steps.",
             "Source thoughts and raw replies are not supplied. Noncommitted attempts are explicitly separate from the replayed committed path."]
    if anchor is not None:
        lines.append(f"Source anchor: before zero-based committed step {anchor}. Its COMPLETE prefix and COMPLETE remaining recorded execution are both retained.")
    if trace.get("replay_limit"):
        lines.append(trace["replay_limit"])
    lines.extend(["Source initial observation:", obs.show(initial, sid + " initial")])
    previous = initial
    for i, event in enumerate(events):
        j = event["source_step"]
        relation = "unanchored history" if anchor is None else "source prefix" if j < anchor else "source anchor" if j == anchor else "source continuation"
        lines.append(f"Source event {i + 1}; committed-step slot {j} [{relation}; {event['kind']}]")
        before = event.get("observation_before")
        if before is not None and (i == 0 or core.norm(before) != core.norm(previous)):
            lines.extend(["Actual source input observation:", obs.show(before, f"{sid} event {i + 1} before")])
        lines.append("Recorded action: " + (str(event["action"]) if event["action"] is not None else "[no valid source action recorded]"))
        if event.get("semantic_action") and event["semantic_action"] != event.get("action"):
            lines.append("Recorded action binding: " + str(event["semantic_action"]))
        if not event["committed"]:
            lines.append("NONCOMMITTED COLLECTION ATTEMPT: recorded failure evidence; no successful transition or free student retry is implied.")
        if event.get("reason"):
            lines.append("Recorded event status: " + str(event["reason"]))
        lines.extend(["Recorded feedback:", obs.show(event.get("feedback"), f"{sid} event {i + 1} feedback")])
        previous = event.get("feedback")
    lines.append(f"END OF COMPLETE SOURCE {sid}: " + ("SUCCESS" if outcome else "FAILURE"))
    lines.append("Source pages, search results and product details belong to this source execution. They are not automatically facts observed by the student.")
    return "\n".join(lines), {"source_id": sid, "rep": trace["rep"], "source_path": trace["source_path"],
            "source_sha256": trace.get("source_sha256"), "source_outcome": "success" if outcome else "failure",
            "trusted_complete_replay": trusted, "anchor_state_t": anchor, "role": role,
            "source_event_count": len(events), "source_committed_step_count": len(raw["steps"]),
            "shown_event_count": len(events), "complete_source_shown": True, "omitted_events": [],
            "thoughts_rendered": False, "planner_thought_fabricated": False}


def explain(plan, raw_success_exists):
    lines = ["FOUR-CASE SELECTION AND ANCHOR SCOPE", "Category: " + plan["case"] + ".",
             "Source eligibility uses only complete SUCCESS/FAILURE labels, never partial reward ranking. Only successful complete executions can supply a successful continuation; eligible suffixes are ranked by actual decision count."]
    if plan["graph_membership_status"] == "unavailable_or_unmatchable_locator":
        lines.append("The current locator is unavailable/unmatchable. The outside fallback is not proof that the physical state is absent from the graph.")
    selected = plan["selected_success"]
    if selected:
        s = core.candidate_summary(selected)
        lines.append(f"Selected applicable successful continuation: {s}. Self success and current compatible catalog suffixes are ranked together by recorded decision count; self has no fixed priority.")
        if selected["kind"] == "catalog_source":
            lines.append("The source matches the current abstract physical/control and parser locator. Its full prefix and successful continuation are shown; execution after the student's different prefix is not independently certified.")
    hist = plan["historical_success"]
    if hist:
        candidate = hist["candidate"]
        if hist["student_state_t"] is None:
            lines.append(f"Source {source_id(candidate['trace'])} is unanchored same-task historical success evidence. No physical LCA or executable connection is claimed.")
        else:
            lines.append(f"Latest actual historical common anchor: student position {hist['student_state_t']}, source {source_id(candidate['trace'])} position {candidate['anchor']}. This is a past occurrence in the real student history, including cycles. It is not permission to roll back and not a route from the current state.")
    elif not selected:
        lines.append("No trustworthy successful continuation or shared success anchor is available." if raw_success_exists else
                     "NO SUCCESSFUL TEACHER/PLANNER DEMONSTRATION IS AVAILABLE FOR THIS TASK. Complete failures remain evidence; no successful LCA or route is invented.")
    tb = plan.get("task_best")
    if tb is not None:
        lines.append(f"Task-best recorded execution: {source_id(tb['trace'])} with final reward {tb['score']:.2f}; relation to the student's actual history: {tb['relation']}" + (f" at student position {tb['student_state_t']}, source position {tb['anchor']}." if tb["anchor"] is not None else ". No shared visit; it is same-task evidence only, not a route from the current state."))
    if plan.get("raw_success_fallback"):
        lines.append("An original record reports success but its full physical replay/seam is unverified. It is shown only as unanchored history, outside trusted suffix ranking.")
    failure = plan["failed_reference"]
    if failure:
        lines.append(f"Complete failed-source evidence: {source_id(failure['trace'])}; relation={failure['relation']}; student actual history position={failure['student_state_t']}; source position={failure['anchor']}. Its entire failed ending is preserved.")
    else:
        lines.append("No additional related failed source is available; no failed source is fabricated.")
    return "\n".join(lines)


def finalize_trajectory(tokenizer, gamefile, trajectory, *, catalog_root=None, prompt_budget=40000):
    if not isinstance(trajectory.get("won"), bool):
        raise ReferenceIntegrityError("Four-case finalization requires the actual complete student outcome")
    catalog_root = catalog_root or os.environ.get("SW_PHYSICAL_GRAPH_ROOT", str(DEFAULT_CATALOG))
    graph, catalog = core.load_catalog_graph(catalog_root, gamefile)
    catalog["fixed_main_scope"] = "four_case_task_manifest"
    visits, successes, failures = core.build_index(graph)
    raw_success_exists = bool(successes or catalog.get("raw_source_success_reps"))
    if not failures:
        fallback = raw_failed_fallback(catalog_root, gamefile, graph)
        if fallback is not None:
            failures = [fallback]
    raw_success = core.unverified_raw_success(graph, catalog) if not successes and raw_success_exists else None
    outputs = []
    for t, step in enumerate(trajectory["steps"]):
        plan = core.plan_step(trajectory, t, visits, successes, failures, graph)
        plan["raw_success_fallback"] = raw_success if not plan["source_success_candidates"] and not plan["historical_success"] else None
        if plan["case"] == "graph_outside_student_succeeded":
            # The actual successful student continuation is the only applicable
            # evidence in case4. An earlier graph anchor is not a current route.
            plan["historical_success"] = None
            plan["raw_success_fallback"] = None
            plan["failed_reference"] = None
        sources = []
        seen = set()
        def add(trace, anchor, role):
            key = source_id(trace), trace.get("source_sha256")
            if key not in seen:
                seen.add(key)
                sources.append((trace, anchor, role))
        selected = plan["selected_success"]
        if selected and selected["kind"] == "catalog_source":
            add(selected["trace"], selected["anchor"], "current matched successful source")
        if plan["historical_success"]:
            h = plan["historical_success"]
            add(h["candidate"]["trace"], h["candidate"]["anchor"], h["kind"])
        if plan["raw_success_fallback"]:
            add(plan["raw_success_fallback"], None, "raw recorded success; unverified replay/seam")
        tb = plan.get("task_best")
        if tb is not None and plan["case"] != "graph_outside_student_succeeded":
            sel_score = core.source_score(selected["trace"]) if selected and selected["kind"] == "catalog_source" else None
            if plan["case"] != "graph_inside_with_success_suffix" or (sel_score is not None and tb["score"] > sel_score + 1e-9):
                add(tb["trace"], tb["anchor"], f"task-best recorded execution (final reward {tb['score']:.2f}; {tb['relation']})")
        if plan["failed_reference"]:
            f = plan["failed_reference"]
            add(f["trace"], f["anchor"], "complete failed-source contrast")
        # Selection is frozen before serialization. No token-budget reranking,
        # source dropping or event truncation is allowed.
        limit = int(prompt_budget)
        for compressed in ((True,) if os.environ.get("WS_REFERENCE_DEDUP", "0") == "1" else (False,)):
            observations = Observations(compressed)
            student = render_student(trajectory, t, observations)
            rendered = [render_source(trace, anchor, role, observations) for trace, anchor, role in sources]
            text = ("<four_case_reference>\nThis reference was assembled after the student rollout. Only complete recorded actions, observations/feedback, outcomes and anchor explanations are supplied. Historical thoughts and raw replies are excluded.\n\n"
                    + student + "\n\n" + explain(plan, raw_success_exists) + "\n\n"
                    + "\n\n".join(x[0] for x in rendered)
                    + "\n</four_case_reference>\n\n" + REMINDER)
            ids = core.insert_reference(tokenizer, step["prompt_ids"], text)
            if len(ids) <= limit:
                break
        else:
            raise ReferenceBudgetError(f"Complete four-case evidence needs {len(ids)} teacher tokens (observation backreferences={compressed}), budget={limit}, student_events={len(trajectory['steps'])}, sources={[source_id(x[0]) for x in sources]}. No failed ending, source, or student event was dropped.")
        hist = plan["historical_success"]
        hist_meta = None if hist is None else {**{k: v for k, v in hist.items() if k != "candidate"},
                                              "source": core.candidate_summary(hist["candidate"])}
        outputs.append({"teacher_prompt_ids": ids, "meta": {
            "protocol": "four_case_complete_action_observation_webshop_v1", "case": plan["case"],
            "student_step": t, "student_event_count": len(trajectory["steps"]),
            "student_complete_events_shown": len(trajectory["steps"]),
            "student_final_outcome": plan["student_final_outcome"], "student_final_score": plan.get("student_final_score"), "task_best_source_score": plan.get("task_best_source_score"), "source_rule": plan.get("source_rule"), "student_reference_complete": True,
            "student_thoughts_rendered": False, "source_thoughts_rendered": False,
            "student_raw_reply_rendered": False, "source_raw_reply_rendered": False,
            "current_action_in_complete_student_record": True, "current_target_response_ids_unchanged": True,
            "future_actions_feedback_and_outcome_allowed": True,
            "selected_success": core.candidate_summary(selected),
            "applicable_source_success_candidates": [core.candidate_summary(x) for x in plan["source_success_candidates"]],
            "self_success_candidate_included": trajectory["won"],
            "historical_success_reference": hist_meta, "task_best_reference": (None if plan.get("task_best") is None else {"rep": plan["task_best"]["trace"]["rep"], "score": plan["task_best"]["score"], "relation": plan["task_best"]["relation"], "anchor": plan["task_best"]["anchor"], "student_state_t": plan["task_best"]["student_state_t"]}), "source_full_display": [x[1] for x in rendered],
            "raw_success_exists": raw_success_exists, "trusted_success_source_count": len(successes),
            "no_successful_teacher_or_planner_demonstration": not raw_success_exists,
            "graph_membership_status": plan["graph_membership_status"], "catalog": catalog,
            "shortest_scope": core.SHORTEST_SCOPE, "source_selection_independent_of_render_budget": True,
            "omitted_sources": [], "omitted_student_events": [], "omitted_source_events": [],
            "exact_observation_backreferences": observations.references,
            "observation_text_compressed": compressed, "teacher_prompt_tokens": len(ids),
            "teacher_prompt_budget": limit, "reserved_response_tokens": int(os.environ.get("WS_MAX_RESPONSE_TOKENS", "2048")), "teacher_max_model_len": int(os.environ.get("WS_TEACHER_MAX_MODEL_LEN", "262144")),
            "raw_physical_state_in_teacher_text": False, "reference_text": text}})
    return outputs
