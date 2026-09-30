"""Collect WebShop teacher trajectories through JSON model/environment endpoints."""
import argparse, hashlib, json, os, re, time, threading, sys
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from gcopd.webshop.protocol import build_prompt as _canonical_build_prompt, parse_action
MAX_STEPS=15
HISTORY=2
CHAT_MEMORY=False
INVALID_FEEDBACK="Invalid response format. Reply with optional <thought>...</thought> followed by exactly one <action>...</action>. This decision round has been consumed; the environment state is unchanged."


def http(url, payload=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode())

def format_obs(obs, task):
    parts = obs.split(" [SEP] ")
    try:
        i = parts.index(task)
        return " [SEP] ".join(f"'{p}'" for p in parts[i + 1:])
    except ValueError:
        return obs

def format_avail(av):
    acts = []
    if av.get("has_search_bar"):
        acts.append("search[<your query>]")
    for c in av.get("clickables", []):
        acts.append(f"click[{c}]")
    return "\n".join(f"'{a}'," for a in acts)

def build_prompt(task, obs_fmt, avail_fmt, history, max_rounds=None):
    return _canonical_build_prompt(task, obs_fmt, avail_fmt, history, history_length=HISTORY)

def is_executable(action, avail):
    """WebShop 会静默忽略非法动作；这里判断以便记录 env_rejection。"""
    m = re.fullmatch(r"(search|click)\[(.*)\]", action, re.S)
    if not m:
        return False
    kind, arg = m.group(1), m.group(2).strip()
    if kind == "search":
        return bool(avail.get("has_search_bar")) and bool(arg)
    return arg in {c.lower() for c in avail.get("clickables", [])} and arg != "search"

def chat(url, model, prompt, temperature, top_p, top_k, seed, max_tokens, messages=None):
    payload = {"model": model, "messages": (messages + [{"role": "user", "content": prompt}]) if messages is not None else [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": top_p,
               "max_tokens": max_tokens, "seed": seed, "chat_template_kwargs": {"enable_thinking": False}, "logprobs": False}
    if top_k is not None:
        payload["top_k"] = top_k
    r = http(f"{url}/v1/chat/completions", payload)
    c = r["choices"][0]
    return c["message"]["content"], c.get("finish_reason"), r.get("usage", {})

def run_episode(a, env_url, llm_url, goal, rep, tag):
    eid = f"{tag}_g{goal}_r{rep}_{os.getpid()}_{threading.get_ident()}"
    t0 = time.time()
    r = http(f"{env_url}/reset", {"episode": eid, "goal": goal})
    task = r["instruction"]; obs = r["obs"]; avail = r["available_actions"]; state = r["state"]
    steps, history, score, done = [], [], 0.0, False
    memory = [] if CHAT_MEMORY else None
    try:
        for turn in range(MAX_STEPS):
            obs_fmt = format_obs(obs, task)
            prompt = build_prompt(task, obs_fmt, format_avail(avail), history)
            seed = int(hashlib.sha256(f"{goal}|{rep}|{turn}|{tag}".encode()).hexdigest()[:8], 16) % 2**31
            t1 = time.time()
            attempts = []
            for attempt in range(max(1, a.validity_retries + 1)):
                seed_a = seed if attempt == 0 else int(hashlib.sha256(f"{goal}|{rep}|{turn}|{tag}|{attempt}".encode()).hexdigest()[:8], 16) % 2**31
                raw, finish, usage = chat(llm_url, a.model, prompt, a.temperature, a.top_p, a.top_k, seed_a, a.max_tokens, memory)
                action, error = parse_action(raw)
                bad = error
                attempts.append({"attempt": attempt, "seed": seed_a, "raw_output": raw, "action": action, "format_error": error,
                                 "executable": (not error) and is_executable(action, avail), "finish_reason": finish})
                if not bad or a.validity_retries == 0:
                    break
            if error:
                raise RuntimeError("Source format-attempt budget exhausted; source is incomplete")
            gen_s = time.time() - t1
            rec = {"turn": turn, "prompt": prompt, "obs": obs, "obs_fmt": obs_fmt, "available_actions": avail, "state_before": state,
                   "raw_output": raw, "finish_reason": finish, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "action": action, "format_error": error, "gen_seconds": round(gen_s, 3), "seed": seed,
                   "attempts": attempts, "validity_retry_exhausted": bool(a.validity_retries and len(attempts) > a.validity_retries and (attempts[-1]["format_error"] or not attempts[-1]["executable"]))}
            if error:
                # 与 ScienceWorld 适配器一致：不执行环境动作，下一轮的"当前观测"是固定错误文本，页面/可点动作不变，这一轮计数
                feedback = INVALID_FEEDBACK; reward = 0.0; done = False
                rec.update(feedback=feedback, reward=reward, done=False, env_rejection=None, state_after=state, executed=False)
                history.append((obs_fmt, "[invalid response format; no environment action]"))
                obs = INVALID_FEEDBACK
            else:
                rejected = None if is_executable(action, avail) else "action not in admissible list; page unchanged"
                s = http(f"{env_url}/step", {"episode": eid, "action": action})
                obs, reward, done, avail, state = s["obs"], s["reward"], s["done"], s["available_actions"], s["state"]
                rec.update(feedback=obs, reward=reward, done=done, env_rejection=rejected, state_after=state, executed=True)
                history.append((obs_fmt, action))
                score = float(reward) if done else score
            if memory is not None:
                memory.append({"role": "user", "content": prompt}); memory.append({"role": "assistant", "content": raw})
            steps.append(rec)
            if done:
                break
    finally:
        try:
            http(f"{env_url}/close", {"episode": eid})
        except Exception:
            pass
    won = bool(done and score >= 1.0 - 1e-9)
    return {"goal": goal, "rep": rep, "tag": tag, "model": a.model, "instruction": task, "won": won, "score": score, "done": done,
            "decision_count": len(steps), "decision_limit_reached": len(steps) >= MAX_STEPS and not done,
            "n_format_errors": sum(1 for s in steps if s["format_error"]), "n_env_rejections": sum(1 for s in steps if s.get("env_rejection")),
            "protocol": {"chat_memory": CHAT_MEMORY, "prompts_module": os.environ.get("WS_PROMPTS_MODULE", "ws_prompts"), "max_tokens": a.max_tokens, "validity_retries": a.validity_retries, "max_steps": MAX_STEPS, "history": HISTORY, "observation_mode": "text_rich", "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "tag_format": "<thought>/<action>", "enable_thinking": False},
            "env_url": env_url, "llm_url": llm_url, "seconds": round(time.time() - t0, 2), "steps": steps}


def load_goals(path):
    data=json.loads(Path(path).read_text())
    values=data['tasks'] if isinstance(data,dict) else data
    goals=[x['goal'] if isinstance(x,dict) else x for x in values]
    if not goals or any(type(x) is not int or x<0 for x in goals) or len(set(goals))!=len(goals):
        raise ValueError('Supply unique nonnegative integer goal IDs')
    return goals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',required=True)
    p.add_argument('--env-url',required=True)
    p.add_argument('--llm-url',required=True,help='Server base URL, without /v1')
    p.add_argument('--model',required=True,help='Served teacher model name')
    p.add_argument('--output',required=True)
    p.add_argument('--reps',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--format-attempts',type=int,default=64)
    p.add_argument('--tag',default='teacher_k16')
    p.add_argument('--temperature',type=float,default=0.4)
    p.add_argument('--top-p',type=float,default=1.0)
    p.add_argument('--top-k',type=int,default=-1)
    a=p.parse_args()
    if a.reps<1 or a.workers<1 or a.format_attempts<1:raise ValueError('Budgets must be positive')
    a.max_tokens=2048;a.validity_retries=a.format_attempts-1
    a.env_url=a.env_url.rstrip('/');a.llm_url=a.llm_url.rstrip('/')
    goals=load_goals(a.tasks)
    health=http(a.env_url+'/health')
    if not health.get('ok') or health.get('observation_mode')!='text_rich' or health.get('seed')!=42 or health.get('goal_shuffle_seed')!=233:
        raise ValueError('Use the text_rich environment with seed 42 and goal shuffle seed 233')
    if a.model not in [x['id'] for x in http(a.llm_url+'/v1/models')['data']]:
        raise ValueError('Requested teacher is not served by the endpoint')
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    signature={'goals':goals,'reps':a.reps,'model':a.model,'tag':a.tag,'format_attempts':a.format_attempts,
               'temperature':a.temperature,'top_p':a.top_p,'top_k':a.top_k,'history':2,'max_tokens':2048}
    receipt=out/'COLLECTION.json'
    if receipt.exists() and json.loads(receipt.read_text())!=signature:raise ValueError('Collection identity differs')
    receipt.write_text(json.dumps(signature,indent=2))
    def run(job):
        goal,rep=job;path=out/f'g{goal:05d}_r{rep:02d}.json'
        if path.exists():
            d=json.loads(path.read_text())
            if d['goal']!=goal or d['rep']!=rep or d.get('technical_incomplete'):raise ValueError('Invalid completed source')
            return
        d=run_episode(a,a.env_url,a.llm_url,goal,rep,a.tag)
        d.update(source_role='training',technical_incomplete=False)
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False));tmp.replace(path)
    with ThreadPoolExecutor(a.workers) as pool:list(pool.map(run,[(g,r) for g in goals for r in range(a.reps)]))
    (out/'COMPLETE.json').write_text(json.dumps({'tasks':len(goals),'reps':a.reps,'episodes':len(goals)*a.reps}))
    print(out/'COMPLETE.json')


if __name__=='__main__':main()
