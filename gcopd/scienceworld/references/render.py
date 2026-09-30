"""State/history reference selection with complete action/observation evidence.

Student terminal outcomes and all actual actions/feedback may be used after
rollout completion. Original response IDs are never rewritten. No original
student/source reply or thought is serialized into reference text.
"""
from __future__ import annotations
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import gcopd.scienceworld.references.selection as core
from gcopd.common.protocol import contains_control_tag, parse_tagged_action

ROOT = Path(__file__).resolve().parent
REMINDER = (
    "CURRENT RESPONSE REQUIREMENT: Act as the ScienceWorld agent in the current student task and state. "
    "Return exactly one <action>...</action>. If you include reasoning, put ALL of it inside one "
    "<thought>...</thought> before that action. Do not put narration outside these tags. "
    "The source above is historical reference, not the student's current state or a request to review the source."
)


class ReferenceBudgetError(ValueError):
    pass


class ReferenceIntegrityError(ValueError):
    pass


def source_id(trace):
    planner = trace.get("source_kind", "").startswith("planner") or trace.get("rep", 0) >= 1000000
    return ("planner_actual_replay" if planner else "original_qwen32b_k16") + ":" + str(trace["rep"])


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
        return None if contains_control_tag(attempt["action"]) else attempt["action"]
    # Validate the complete block structure before extracting historical action
    # text. Empty recorded actions may represent cancellation; this does not
    # authorize execution or bypass the live parser's observation-dependent gate.
    raw = str(attempt.get("raw_output", ""))
    _, error = parse_tagged_action(raw, allow_empty=True)
    if error is not None:
        return None
    # Preserve the recorded action's spelling after validating its boundaries.
    return raw.partition("<action>")[2].partition("</action>")[0].strip()


def render_action(value, missing):
    """Keep malformed historical control-tag content out of action evidence."""
    if value is None:
        return missing
    if contains_control_tag(value):
        return "[recorded action text withheld: invalid control tags]"
    return str(value)


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
        if seen != set(range(len(steps))):
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
                candidates.append({"rep": item["rep"], "source_path": item["path"],
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
    won = trajectory["won"]
    lines = ["STUDENT'S COMPLETE ACTUAL ACTION/OBSERVATION EXECUTION",
             "This complete student execution SUCCEEDED." if won else
             "This complete student execution FAILED. This outcome does not mean each visited state or action must fail.",
             f"Actual student events: {len(trajectory['steps'])}; current scored decision: {t + 1}.",
             "All recorded actions and feedback, including later events and the final outcome, are supplied after rollout completion. No historical thought or raw reply is supplied.",
             "Student initial observation:", obs.show(trajectory.get("initial_obs"), "student initial")]
    for i, row in enumerate(trajectory["steps"]):
        lines.append(f"Student event {i + 1}" + (" [CURRENT SCORED DECISION]" if i == t else ""))
        before = row.get("current_obs")
        if before is not None and (i == 0 or core.norm(before) != core.norm(trajectory["steps"][i - 1].get("feedback"))):
            lines.extend(["Actual input observation:", obs.show(before, f"student event {i + 1} before")])
        lines.append("Recorded action: " + render_action(row.get("action"), "[no valid parsed environment action]"))
        if row.get("semantic_action") and row["semantic_action"] != row.get("action"):
            lines.append("Recorded action binding: " + render_action(row["semantic_action"], "[no recorded binding]"))
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
    trusted = core.successful(trace) if outcome else core.failed_source(trace)
    lines = [f"COMPLETE SOURCE ACTION/OBSERVATION EXECUTION: {sid}",
             f"Original file: {Path(trace['source_path']).name}; role: {role}.",
             "The COMPLETE original source record SUCCEEDED." if outcome else
             "The COMPLETE original source record FAILED. This is not a claim that its states necessarily fail or that every action was wrong.",
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
        lines.append("Recorded action: " + render_action(event["action"], "[no valid source action recorded]"))
        if event.get("semantic_action") and event["semantic_action"] != event.get("action"):
            lines.append("Recorded action binding: " + render_action(event["semantic_action"], "[no recorded binding]"))
        if not event["committed"]:
            lines.append("NONCOMMITTED COLLECTION ATTEMPT: recorded failure evidence; no successful transition or free student retry is implied.")
        if event.get("reason"):
            lines.append("Recorded event status: " + str(event["reason"]))
        lines.extend(["Recorded feedback:", obs.show(event.get("feedback"), f"{sid} event {i + 1} feedback")])
        previous = event.get("feedback")
    lines.append(f"END OF COMPLETE SOURCE {sid}: " + ("SUCCESS" if outcome else "FAILURE"))
    lines.append("Source observations, measurements and preparations belong to this source execution. They are not automatically facts observed by the student.")
    return "\n".join(lines), {"source_id": sid, "rep": trace["rep"], "source_path": trace["source_path"],
            "source_sha256": trace.get("source_sha256"), "source_outcome": "success" if outcome else "failure",
            "trusted_complete_replay": trusted, "anchor_state_t": anchor, "role": role,
            "source_event_count": len(events), "source_committed_step_count": len(raw["steps"]),
            "shown_event_count": len(events), "complete_source_shown": True, "omitted_events": [],
            "thoughts_rendered": False, "planner_thought_fabricated": False}


def explain(plan, raw_success_exists):
    lines = ["FOUR-CASE SELECTION AND ANCHOR SCOPE", "Category: " + plan["case"] + "."]
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
    if plan.get("raw_success_fallback"):
        lines.append("An original record reports success but its full physical replay/seam is unverified. It is shown only as unanchored history, outside trusted suffix ranking.")
    failure = plan["failed_reference"]
    if failure:
        lines.append(f"Complete failed-source evidence: {source_id(failure['trace'])}; relation={failure['relation']}; student actual history position={failure['student_state_t']}; source position={failure['anchor']}. Its entire failed ending is preserved.")
    else:
        lines.append("No additional related failed source is available; no failed source is fabricated.")
    return "\n".join(lines)


def finalize_hindsight(tokenizer, trajectory, prompt_budget):
    """Use the GC student renderer while disabling every external-record path."""
    outputs=[]
    limit=min(int(prompt_budget),40000)
    for t,step in enumerate(trajectory['steps']):
        for compressed in (False,True):
            observations=Observations(compressed)
            student=render_student(trajectory,t,observations)
            text=('<student_hindsight>\n'+student+'\n</student_hindsight>\n\n'+REMINDER)
            ids=core.insert_reference(tokenizer,step['prompt_ids'],text)
            if len(ids)<=limit:break
        else:
            raise ReferenceBudgetError(f'Complete student hindsight needs {len(ids)} tokens; budget={limit}. No student event was dropped.')
        outputs.append({'teacher_prompt_ids':ids,'meta':{
            'protocol':'student_hindsight_only_v1','student_step':t,
            'student_event_count':len(trajectory['steps']),
            'student_complete_events_shown':len(trajectory['steps']),
            'student_final_outcome':'success' if trajectory['won'] else 'failure',
            'student_reference_complete':True,'student_thoughts_rendered':False,
            'current_target_response_ids_unchanged':True,'external_records_enabled':False,
            'source_full_display':[],'omitted_student_events':[],
            'teacher_prompt_tokens':len(ids),'teacher_prompt_budget':limit,
            'exact_observation_backreferences':observations.references,
            'observation_text_compressed':compressed,'reference_text':text}})
    return outputs


def _finalize_trajectory(tokenizer, gamefile, trajectory, *, catalog_root=None, prompt_budget=40000):
    if not isinstance(trajectory.get("won"), bool):
        raise ReferenceIntegrityError("Four-case finalization requires the actual complete student outcome")
    if os.environ.get('SW_EXTERNAL_RECORDS','1')=='0':
        return finalize_hindsight(tokenizer,trajectory,prompt_budget)
    catalog_root = catalog_root or os.environ.get("SW_PHYSICAL_GRAPH_ROOT")
    if not catalog_root:raise ReferenceIntegrityError('Set catalog_root or SW_PHYSICAL_GRAPH_ROOT for external records')
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
        if plan["failed_reference"]:
            f = plan["failed_reference"]
            add(f["trace"], f["anchor"], "complete failed-source contrast")
        # Selection is frozen before serialization. No token-budget reranking,
        # source dropping or event truncation is allowed.
        limit = min(int(prompt_budget), 40000)
        for compressed in (False, True):
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
            raise ReferenceBudgetError(f"Complete four-case evidence needs {len(ids)} teacher tokens after exact observation back-references, budget={limit}, student_events={len(trajectory['steps'])}, sources={[source_id(x[0]) for x in sources]}. No failed ending, source, or student event was dropped.")
        hist = plan["historical_success"]
        hist_meta = None if hist is None else {**{k: v for k, v in hist.items() if k != "candidate"},
                                              "source": core.candidate_summary(hist["candidate"])}
        outputs.append({"teacher_prompt_ids": ids, "meta": {
            "protocol": "four_case_complete_action_observation_v1", "case": plan["case"],
            "student_step": t, "student_event_count": len(trajectory["steps"]),
            "student_complete_events_shown": len(trajectory["steps"]),
            "student_final_outcome": plan["student_final_outcome"], "student_reference_complete": True,
            "student_thoughts_rendered": False, "source_thoughts_rendered": False,
            "student_raw_reply_rendered": False, "source_raw_reply_rendered": False,
            "current_action_in_complete_student_record": True, "current_target_response_ids_unchanged": True,
            "future_actions_feedback_and_outcome_allowed": True,
            "selected_success": core.candidate_summary(selected),
            "applicable_source_success_candidates": [core.candidate_summary(x) for x in plan["source_success_candidates"]],
            "self_success_candidate_included": trajectory["won"],
            "historical_success_reference": hist_meta, "source_full_display": [x[1] for x in rendered],
            "raw_success_exists": raw_success_exists, "trusted_success_source_count": len(successes),
            "no_successful_teacher_or_planner_demonstration": not raw_success_exists,
            "graph_membership_status": plan["graph_membership_status"], "catalog": catalog,
            "shortest_scope": core.SHORTEST_SCOPE, "source_selection_independent_of_render_budget": True,
            "omitted_sources": [], "omitted_student_events": [], "omitted_source_events": [],
            "exact_observation_backreferences": observations.references,
            "observation_text_compressed": compressed, "teacher_prompt_tokens": len(ids),
            "teacher_prompt_budget": limit, "reserved_response_tokens": 512, "teacher_max_model_len": 40513,
            "raw_physical_state_in_teacher_text": False, "reference_text": text}})
    return outputs


def terminal_failure_note(trajectory):
    outcome=trajectory.get('final_outcome') or {}
    if trajectory.get('technical_incomplete') or outcome.get('done') is not True or outcome.get('decision_limit_reached') is not False:
        return None
    score=trajectory.get('final_score')
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
        raise ReferenceIntegrityError('Terminal annotation requires finite recorded final score')
    if score>=0 or trajectory.get('won') is True:
        return None
    steps=trajectory.get('steps') or []
    if not steps or trajectory.get('won') is not False or outcome.get('won') is not False or outcome.get('score')!=score or outcome.get('decision_count')!=len(steps) or steps[-1].get('score')!=score:
        raise ReferenceIntegrityError('Inconsistent recorded terminal failure metadata')
    action=steps[-1].get('action')
    if not isinstance(action,str) or not action.strip():
        raise ReferenceIntegrityError('Terminal failure annotation requires the recorded action')
    return (f'CURRENT ACTION AND ITS OBSERVED CONSEQUENCE: The recorded action "{action}" immediately ended this task with FAILURE (score {score:g}). In this execution, that action was a task-ending decision, not a preliminary observation or the beginning of a later experiment. No later student decision followed it. Assess the action and any claims about its purpose against this observed consequence. A successful task outcome does not make an inaccurate explanation correct, and an inaccurate explanation does not change a successful task outcome. This is a local fact about this recorded transition, not a rule about every use of this action in other states or tasks. Earlier decisions are not labelled by this outcome.')

def finalize_with_annotation(base, tokenizer, gamefile, trajectory, *, catalog_root=None, prompt_budget=40000):
    note=terminal_failure_note(trajectory)
    outputs=base(tokenizer,gamefile,trajectory,catalog_root=catalog_root,prompt_budget=prompt_budget)
    if note is None:
        return outputs
    last=outputs[-1];meta=dict(last['meta']);text=note+'\n\n'+meta['reference_text']
    ids=core.insert_reference(tokenizer,trajectory['steps'][-1]['prompt_ids'],text)
    limit=min(int(prompt_budget),40000)
    if len(ids)>limit:
        raise ReferenceBudgetError(f'Complete evidence plus terminal-failure annotation needs {len(ids)} tokens, budget {limit}; no source/event omitted')
    meta.update(reference_text=text,teacher_prompt_tokens=len(ids),terminal_transition_annotation={'protocol':'observed_terminal_failure_v1','applied':True,'student_step':len(outputs)-1,'evidence':'actual_done_true_negative_score_before_decision_limit','final_score':trajectory['final_score'],'note':note,'only_current_final_failure_annotated':True,'source_selection_unchanged':True,'historical_thoughts_added':False})
    outputs[-1]={'teacher_prompt_ids':ids,'meta':meta}
    return outputs


def finalize_trajectory(tokenizer, gamefile, trajectory, *, catalog_root=None, prompt_budget=40000):
    if os.environ.get("SW_TERMINAL_FAILURE_NOTE", "0") == "1":
        return finalize_with_annotation(_finalize_trajectory, tokenizer, gamefile, trajectory,
                                        catalog_root=catalog_root, prompt_budget=prompt_budget)
    return _finalize_trajectory(tokenizer, gamefile, trajectory,
                                catalog_root=catalog_root, prompt_budget=prompt_budget)
