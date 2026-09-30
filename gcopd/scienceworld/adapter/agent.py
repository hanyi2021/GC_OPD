"""Student h5 rollout followed by retrospective physical-graph teacher reference."""
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid

from gcopd.common.io import require_backend
require_backend()
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from scienceworld import ScienceWorldEnv

from gcopd.scienceworld import environment as core

from gcopd.scienceworld.adapter.state_bridge import PhysicalStateTracker
from gcopd.training.context import finalize_teacher_prompts

INVALID = "Invalid response format. Reply with optional <thought>...</thought> followed by exactly one <action>...</action>. This decision round has been consumed; the environment state is unchanged."


class ScienceWorldLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        extra = kwargs.get("extra_info", {})
        gamefile = extra.get("gamefile", kwargs.get("gamefile"))
        trace = kwargs.get("_sw_trajectory_info", {})
        rep = int(trace.get("rollout_n", 0))
        version = int(trace.get("step", 1)) - 1
        is_eval = bool(trace.get("validate", False))
        if is_eval and "eval_rep" in extra:
            rep = int(extra["eval_rep"])
        if extra.get("padding", False):
            return AgentLoopOutput(prompt_ids=[151644], response_ids=[151645], response_mask=[0],
                response_logprobs=[0.], reward_score=0., metrics=AgentLoopMetrics(),
                extra_fields={"sw_turns": [], "sw_padding": True, "sw_gamefile": gamefile,
                              "sw_episode_file": None, "sw_initial_cache": False})
        env = await asyncio.to_thread(ScienceWorldEnv, "", envStepLimit=200)
        turns, history = [], []
        generation_time, started = 0., time.time()
        tracker = None
        try:
            task, variation, simplification = gamefile.split("|")
            await asyncio.to_thread(env.load, task, int(variation), simplification)
            observation, info = await asyncio.to_thread(env.reset)
            score, initial_observation = info["score"], observation
            task_description = env.get_task_description()
            if not is_eval:
                tracker = PhysicalStateTracker(env)
                await asyncio.to_thread(tracker.start, observation, info)
            max_decisions = int(os.environ.get("SW_SMOKE_MAX_DECISIONS", "30"))
            done = False
            for turn in range(max_decisions):
                before_observation = observation
                from gcopd.scienceworld.protocol import build_messages
                user, messages = build_messages(core.prompts, task_description, observation,
                    env.get_possible_actions(), env.get_possible_objects(), history)
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                    add_generation_prompt=True, enable_thinking=False)[-10240:]
                physical_before = tracker.before() if tracker else None
                # No finalizer or teacher reference is called before any student generation.
                seed_text = f"{gamefile}|{rep}|{turn}|0" if is_eval else f"{gamefile}|{rep}|{turn}|{version}|42"
                params = dict(sampling_params)
                params.update(max_tokens=512, seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16) % 2**31)
                begin = time.time()
                output = await self.server_manager.generate(request_id=uuid.uuid4().hex,
                    prompt_ids=prompt, sampling_params=params)
                generation_time += time.time() - begin
                ids, logs = list(output.token_ids), list(output.log_probs)
                assert 0 < len(ids) <= 512 and len(logs) == len(ids)
                raw = self.tokenizer.decode(ids, skip_special_tokens=True)
                action, error = core.parse_format(raw, observation)
                done, rejection, tick = False, None, env.get_num_moves()
                options = core.choices(tracker.last_environment_observation if tracker else before_observation)
                semantic = options.get(action, action) if action is not None else None
                semantic_known = bool(action is not None and (not action.isdigit() or action in options))
                if error:
                    observation = INVALID
                    history_action = "[invalid response format; no environment action]"
                else:
                    observation, _, done, info = await asyncio.to_thread(env.step, action)
                    score = info["score"]
                    rejection = core.rejection_reason(observation)
                    history_action = action
                physical_after, transition = None, None
                if tracker:
                    physical_after, transition = await asyncio.to_thread(tracker.after_decision,
                        observation, info, action or "", env_step_called=not bool(error),
                        rejection=rejection, format_error=error)
                history.append(core.prompts.tcod_format_history(before_observation, turn + 1, history_action))
                turns.append({
                    "student_step": turn, "current_obs": before_observation, "user_text": user,
                    "prompt_ids": list(prompt), "response_ids": ids, "rollout_logprobs": logs,
                    "raw_output": raw, "action": action, "semantic_action": semantic,
                    "semantic_action_known": semantic_known, "feedback": observation,
                    "format_error": error, "env_rejection": rejection,
                    "tick_before": tick, "tick_after": env.get_num_moves(), "score": score,
                    "physical_state_before": physical_before, "physical_state_after": physical_after,
                    "physical_transition": transition,
                })
                if score >= 100 or done:
                    break
            directory = Path(os.environ["SW_RUN_DIR"])
            destination = directory / "episodes"
            destination.mkdir(parents=True, exist_ok=True)
            episode_id = uuid.uuid4().hex
            private_audit = await asyncio.to_thread(tracker.write_private_audit, directory, episode_id,
                gamefile, rep, version) if tracker else None
            record = {
                "model_path": self.config.actor_rollout_ref.model.path, "is_evaluation": is_eval,
                "sample_index": trace.get("sample_index"), "gamefile": gamefile, "rep": rep,
                "policy_version": version, "won": score >= 100, "steps": turns,
                "initial_obs": initial_observation, "task_description": task_description,
                "final_score": score,
                "final_outcome": {"won": score >= 100, "score": score, "done": bool(done),
                                  "decision_count": len(turns),
                                  "decision_limit_reached": len(turns) >= max_decisions},
                "technical_incomplete": False,
                "physical_state_audit": private_audit,
                "physical_capture_seconds": tracker.capture_seconds if tracker else 0.,
                "physical_visit_count": len(tracker.snapshots) if tracker else 0,
                "physical_state_visible_to_student_or_teacher": False,
                "teacher_reference_timing": "after_complete_student_trajectory",
                "student_failure_rollout_retained_for_opd": True,
            }
            episode_file = destination / f"{episode_id}.json"
            begin = time.time()
            try:
                if is_eval:
                    for step in turns:
                        step["teacher_prompt_ids"] = list(step["prompt_ids"])
                        step["graph_reference"] = {"fallback": "evaluation_no_graph"}
                else:
                    await asyncio.to_thread(finalize_teacher_prompts, self.tokenizer, record, environment="scienceworld")
            except Exception:
                record.update(technical_incomplete=True, teacher_finalization_failed=True,
                              teacher_finalization_error=traceback.format_exc(), seconds=time.time() - started)
                episode_file.write_text(json.dumps(record, ensure_ascii=False))
                raise
            record.update(teacher_finalization_seconds=time.time() - begin, seconds=time.time() - started)
            episode_file.write_text(json.dumps(record, ensure_ascii=False))
            last = turns[-1]
            return AgentLoopOutput(prompt_ids=last["prompt_ids"], response_ids=last["response_ids"],
                response_mask=[1] * len(last["response_ids"]), response_logprobs=last["rollout_logprobs"],
                reward_score=float(score >= 100), num_turns=len(turns),
                metrics=AgentLoopMetrics(generate_sequences=generation_time),
                extra_fields={"sw_turns": turns, "sw_padding": False, "sw_gamefile": gamefile,
                              "sw_episode_file": str(episode_file), "sw_initial_cache": False})
        finally:
            await asyncio.to_thread(env.close)
