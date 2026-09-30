"""WebShop student rollout (H-window, no chat memory) followed by retrospective four-case teacher reference. Mirrors sw_adapter/agent.py."""
import asyncio, hashlib, json, os, time, traceback, uuid, urllib.request
from pathlib import Path
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, AgentLoopMetrics
from gcopd.webshop.adapter.prompts import INVALID_FEEDBACK, format_obs, format_avail, build_prompt, parse_action, is_executable, core_state
from gcopd.training.context import finalize_teacher_prompts

ENV_URL = os.environ.get("WS_ENV_URL", "http://127.0.0.1:8391")

def http(url, payload=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode())

class WebShopLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        extra = kwargs.get("extra_info", {}); gamefile = extra.get("gamefile", kwargs.get("gamefile"))
        trace = kwargs.get("_sw_trajectory_info", {}); rep = int(trace.get("rollout_n", 0)); version = int(trace.get("step", 1)) - 1
        is_eval = bool(trace.get("validate", False))
        if is_eval and "eval_rep" in extra: rep = int(extra["eval_rep"])
        if extra.get("padding", False):
            pad = self.tokenizer.pad_token_id
            eos = self.tokenizer.eos_token_id
            assert pad is not None and eos is not None, "Padding requires tokenizer-defined pad/eos IDs"
            return AgentLoopOutput(prompt_ids=[pad], response_ids=[eos], response_mask=[0], response_logprobs=[0.], reward_score=0., metrics=AgentLoopMetrics(),
                                   extra_fields={"sw_turns": [], "sw_padding": True, "sw_gamefile": gamefile, "sw_episode_file": None, "sw_initial_cache": False, "ws_score": 0.0})
        goal = int(extra.get("goal", str(gamefile).replace("goal", "")))
        eid = f"{'eval' if is_eval else 'train'}_{gamefile}_r{rep}_v{version}_{uuid.uuid4().hex[:8]}"
        turns, history, generation_time, started = [], [], 0., time.time()
        max_decisions = int(os.environ.get("WS_MAX_DECISIONS", "15")); max_prompt = int(os.environ.get("WS_MAX_PROMPT_TOKENS", "10240"))
        max_response = int(os.environ.get("WS_MAX_RESPONSE_TOKENS", "2048"))
        try:
            r = await asyncio.to_thread(http, f"{ENV_URL}/reset", {"episode": eid, "goal": goal})
            task, obs, avail, state = r["instruction"], r["obs"], r["available_actions"], r["state"]; initial_obs = obs
            score, done = 0.0, False
            for turn in range(max_decisions):
                before_obs = obs
                obs_fmt = format_obs(obs, task); user = build_prompt(task, obs_fmt, format_avail(avail), history, max_rounds=max_decisions)
                prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
                if len(prompt) > max_prompt:
                    raise ValueError(f'WebShop prompt {len(prompt)} exceeds non-truncating budget {max_prompt}')
                # Freeze the training stream's existing per-task seed namespace.
                # Policy version is metadata, not a change to paired sampling.
                seed_text = f"{goal}|{rep}|{turn}|probe32_paired_v1" if is_eval else f"{gamefile}|{rep}|{turn}|0|42"
                params = dict(sampling_params); params.update(max_tokens=max_response, seed=int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16) % 2**31)
                if 'WS_REPETITION_PENALTY' in os.environ:
                    params.update(repetition_penalty=float(os.environ['WS_REPETITION_PENALTY']), presence_penalty=0.0, frequency_penalty=0.0, min_p=0.0)
                begin = time.time()
                output = await self.server_manager.generate(request_id=uuid.uuid4().hex, prompt_ids=prompt, sampling_params=params)
                generation_time += time.time() - begin
                ids, logs = list(output.token_ids), list(output.log_probs); assert 0 < len(ids) <= max_response and len(logs) == len(ids)
                raw = self.tokenizer.decode(ids, skip_special_tokens=True)
                state_before = core_state(state); rejection = None; reward = 0.0
                action, error = parse_action(raw)
                if error:
                    obs = INVALID_FEEDBACK; history.append((obs_fmt, "[invalid response format; no environment action]")); feedback = INVALID_FEEDBACK
                else:
                    rejection = None if is_executable(action, avail) else "action not in admissible list; page unchanged"
                    s = await asyncio.to_thread(http, f"{ENV_URL}/step", {"episode": eid, "action": action})
                    obs, reward, done, avail, state = s["obs"], s["reward"], s["done"], s["available_actions"], s["state"]; feedback = obs
                    history.append((obs_fmt, action))
                    if done: score = float(reward)
                turns.append({"student_step": turn, "current_obs": before_obs, "user_text": user, "prompt_ids": list(prompt), "response_ids": ids, "rollout_logprobs": logs,
                              "raw_output": raw, "action": action, "semantic_action": action, "semantic_action_known": bool(action and not error and not rejection),
                              "feedback": feedback, "format_error": error, "env_rejection": rejection, "tick_before": turn, "tick_after": turn + 1, "score": score,
                              "physical_state_before": state_before, "physical_state_after": core_state(state), "physical_transition": {"kind": "format_invalid" if error else ("rejection" if rejection else ("state_change" if state_before["key"] != state["strict_key"] else "unchanged")), "decision_cost": 1}})
                if done: break
            won = bool(done and score >= 1.0 - 1e-9)
            directory = Path(os.environ["WS_RUN_DIR"]); dest = directory / "episodes"; dest.mkdir(parents=True, exist_ok=True); episode_id = uuid.uuid4().hex
            record = {"model_path": self.config.actor_rollout_ref.model.path, "is_evaluation": is_eval, "sample_index": trace.get("sample_index"), "gamefile": gamefile, "goal": goal, "rep": rep,
                      "policy_version": version, "won": won, "steps": turns, "initial_obs": initial_obs, "task_description": task, "final_score": score,
                      "final_outcome": {"won": won, "score": score, "done": bool(done), "decision_count": len(turns), "decision_limit_reached": len(turns) >= max_decisions and not done},
                      "technical_incomplete": False, "physical_state_visible_to_student_or_teacher": False, "teacher_reference_timing": "after_complete_student_trajectory", "student_failure_rollout_retained_for_opd": True}
            episode_file = dest / f"{episode_id}.json"; begin = time.time()
            try:
                if is_eval:
                    for step in turns: step["teacher_prompt_ids"] = list(step["prompt_ids"]); step["graph_reference"] = {"fallback": "evaluation_no_graph"}
                else:
                    await asyncio.to_thread(finalize_teacher_prompts, self.tokenizer, record, environment="webshop")
            except Exception:
                record.update(technical_incomplete=True, teacher_finalization_failed=True, teacher_finalization_error=traceback.format_exc(), seconds=time.time() - started)
                episode_file.write_text(json.dumps(record, ensure_ascii=False)); raise
            record.update(teacher_finalization_seconds=time.time() - begin, seconds=time.time() - started); episode_file.write_text(json.dumps(record, ensure_ascii=False))
            last = turns[-1]
            return AgentLoopOutput(prompt_ids=last["prompt_ids"], response_ids=last["response_ids"], response_mask=[1] * len(last["response_ids"]), response_logprobs=last["rollout_logprobs"],
                                   reward_score=float(won), num_turns=len(turns), metrics=AgentLoopMetrics(generate_sequences=generation_time),
                                   extra_fields={"sw_turns": turns, "sw_padding": False, "sw_gamefile": gamefile, "sw_episode_file": str(episode_file), "sw_initial_cache": False, "ws_score": score})
        finally:
            try: await asyncio.to_thread(http, f"{ENV_URL}/close", {"episode": eid})
            except Exception: pass
