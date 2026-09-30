"""Strictly replay WebShop trajectories and construct source-preserving graphs."""
import argparse, gzip, hashlib, json, os, sys, time, urllib.request
from pathlib import Path
from collections import defaultdict
from gcopd.webshop.collect import load_goals


def http(url, payload=None, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers={"Content-Type": "application/json"}, method="POST" if payload is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read().decode())

def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def state_record(st, rep, t, obs):
    keep = {k: st.get(k) for k in ["page_type", "keywords", "page", "asin", "options", "sub_page", "asins_visited", "done", "reward", "url", "n_clickables"]}
    return {"key": st["strict_key"], "node_key": st["strict_key"], "coarse_key": st["coarse_key"], "physical_hash": st["coarse_key"],
            "control": {"page_type": st["page_type"], "asin": st.get("asin"), "options": sorted((str(k), str(v)) for k, v in (st.get("options") or {}).items())},
            "interaction": {"mode": "normal", "options": {}}, "capture_ok": True, "matchable": True, "codec_version": "webshop_url_session_v1",
            "state_t": t, "visit_id": f"r{rep}:t{t}", "observation": obs, "available_actions": st.get("available_actions"), **{"ws_" + k: v for k, v in keep.items()}}

def normalized_source(ep, out_path):
    """写与 ScienceWorld 原始 K16 源文件同构的文档：steps[observation_before/feedback/action/semantic_action/format_error/rejection] + attempts[step,type] + initial_observation + won。"""
    steps, attempts = [], []
    for t, s in enumerate(ep["steps"]):
        steps.append({"action": s["action"], "semantic_action": s["action"], "observation_before": s["obs"], "feedback": s["feedback"],
                      "format_error": s["format_error"], "rejection": s.get("env_rejection"), "reward": s["reward"], "done": s["done"],
                      "page_type_before": s["state_before"]["page_type"], "page_type_after": s["state_after"]["page_type"]})
        for a in (s.get("attempts") or [{"attempt": 0, "raw_output": s["raw_output"], "action": s["action"], "format_error": s["format_error"], "executable": s.get("executed", True)}]):
            last = a is (s.get("attempts") or [a])[-1]
            typ = "accepted" if last and not s["format_error"] else ("format_invalid" if a.get("format_error") else ("not_executable" if not a.get("executable", True) else "accepted"))
            attempts.append({"step": t + 1, "type": typ, "action": a.get("action"), "reason": a.get("format_error") or (None if a.get("executable", True) else "action not in admissible list"), "feedback": s["feedback"] if last else None})
    doc = {"goal": ep["goal"], "gamefile": f"goal{ep['goal']}", "rep": ep["rep"], "won": bool(ep["won"]), "score": ep["score"], "done": ep["done"], "instruction": ep["instruction"],
           "initial_observation": ep["steps"][0]["obs"] if ep["steps"] else None, "steps": steps, "attempts": attempts, "protocol": ep.get("protocol"), "source_kind": ep.get("tag", "webshop_k16"), "raw_episode_path": None}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, ensure_ascii=False)
    return out_path

def trajectory_from_episode(ep, path, env_url):
    rep = ep["rep"]; steps_in = ep["steps"]
    states, steps = [], []
    if not steps_in:
        return None
    states.append(state_record(steps_in[0]["state_before"], rep, 0, steps_in[0]["obs"]))
    for t, s in enumerate(steps_in):
        before_key = states[-1]["key"]
        after = s["state_after"]
        kind = "format_invalid" if s["format_error"] else ("rejection" if s.get("env_rejection") else ("state_change" if after["strict_key"] != before_key else "unchanged"))
        steps.append({"step_index": t, "decision_index": t, "action": s["action"], "semantic_action": s["action"], "executed_action": s["action"] if s.get("executed") else None,
                      "observation_before": s["obs"], "feedback": s["feedback"], "reward": s["reward"], "done": s["done"], "kind": kind,
                      "format_error": s["format_error"], "rejection": s.get("env_rejection"), "env_rejection": s.get("env_rejection"),
                      "decision_cost": 1, "positive_candidate_allowed": not (s["format_error"] or s.get("env_rejection")),
                      "continuous_from_previous": True, "attempts": len(s.get("attempts") or []) or 1, "validity_retry_exhausted": s.get("validity_retry_exhausted", False)})
        states.append(state_record(after, rep, t + 1, s["feedback"]))
    # 可信性回放：只重放真正执行了的动作序列
    executed = [s["action"] for s in steps_in if s.get("executed")]
    replay_ok, replay_note = None, None
    if env_url:
        try:
            r = http(f"{env_url}/replay", {"goal": ep["goal"], "actions": executed})
            exp_keys = [st["key"] for st, s in zip(states[1:], steps_in) if s.get("executed")]
            got_keys = [st["strict_key"] for st in r["states"][1:]]
            replay_ok = (got_keys == exp_keys) and (abs((r["rewards"][-1] if r["rewards"] else 0.0) - (ep["score"] if ep["done"] else 0.0)) < 1e-9 or not ep["done"])
            replay_ok = replay_ok and r['states'][0]['strict_key'] == steps_in[0]['state_before']['strict_key']
            replay_ok = replay_ok and r['obs'][0] == steps_in[0]['obs']
            replay_ok = replay_ok and r['obs'][1:] == [s['feedback'] for s in steps_in if s.get('executed')]
            replay_ok = replay_ok and r['dones'] == [bool(s['done']) for s in steps_in if s.get('executed')]
            if not replay_ok:
                replay_note = f"replay mismatch: {len(got_keys)} vs {len(exp_keys)} keys; first diff at {next((i for i,(x,y) in enumerate(zip(got_keys,exp_keys)) if x!=y), None)}"
        except Exception as e:
            replay_ok, replay_note = False, f"replay error {type(e).__name__}: {str(e)[:120]}"
    trusted = bool(replay_ok) if env_url else False
    return {"rep": rep, "source_path": path, "source_sha256": sha_file(path), "source_won": bool(ep["won"]), "trusted_complete": trusted,
            "replayed_won": bool(ep["won"]) if trusted else False, "trusted_success": bool(ep["won"]) and trusted, "won": bool(ep["won"]) and trusted,
            "score": ep["score"], "states": states, "steps": steps, "source_kind": ep.get("tag", "webshop_k16"), "replay_failures": [] if replay_ok in (None, True) else [replay_note],
            "source_attempt_summary": {"decisions": len(steps_in), "format_errors": ep.get("n_format_errors", 0), "env_rejections": ep.get("n_env_rejections", 0)}}

def build_graph(goal, instruction, trajs):
    nodes, edges = {}, {}
    coarse = defaultdict(set)
    for tr in trajs:
        for t, st in enumerate(tr["states"]):
            k = st["key"]
            if k not in nodes:
                nodes[k] = {"id": f"N{len(nodes)}", "key": k, "coarse_key": st["coarse_key"], "page_type": st["ws_page_type"], "asin": st["ws_asin"], "keywords": st["ws_keywords"], "page": st["ws_page"],
                            "options": st["ws_options"], "sub_page": st["ws_sub_page"], "members": [], "observations": [], "start": t == 0, "won": False, "reps": set(), "min_t": t}
            n = nodes[k]; n["members"].append({"rep": tr["rep"], "state_t": t, "visit_id": st["visit_id"]}); n["reps"].add(tr["rep"]); n["min_t"] = min(n["min_t"], t)
            if len(n["observations"]) < 2 and st["observation"] not in n["observations"]:
                n["observations"].append(st["observation"])
            if tr["won"]:
                n["won"] = True
            coarse[st["coarse_key"]].add(k)
        for t, s in enumerate(tr["steps"]):
            a, b = tr["states"][t]["key"], tr["states"][t + 1]["key"]
            ek = (a, b, s["action"])
            if ek not in edges:
                edges[ek] = {"id": f"E{len(edges)}", "source": a, "target": b, "action": s["action"], "kind": s["kind"], "members": []}
            edges[ek]["members"].append({"rep": tr["rep"], "step_index": t})
    for n in nodes.values():
        n["reps"] = sorted(n["reps"]); n["n_reps"] = len(n["reps"])
    stats = {"nodes": len(nodes), "edges": len(edges), "trajectories": len(trajs), "trusted_success": sum(t["won"] for t in trajs), "source_success": sum(t["source_won"] for t in trajs),
             "untrusted": sum(not t["trusted_complete"] for t in trajs), "state_visits": sum(len(t["states"]) for t in trajs),
             "merged_nodes_2plus_reps": sum(1 for n in nodes.values() if n["n_reps"] >= 2), "coarse_groups_with_multiple_strict": sum(1 for v in coarse.values() if len(v) > 1),
             "mean_steps": sum(len(t["steps"]) for t in trajs) / max(1, len(trajs)), "page_type_hist": dict(defaultdict(int, {pt: sum(1 for n in nodes.values() if n["page_type"] == pt) for pt in set(n["page_type"] for n in nodes.values())}))}
    return {"schema_version": "webshop_graph_v1", "gamefile": f"goal{goal}", "goal": goal, "instruction": instruction, "codec_version": "webshop_url_session_v1",
            "nodes": nodes, "edges": list(edges.values()), "trajectories": trajs, "coarse_index": {k: sorted(v) for k, v in coarse.items()}, "stats": stats,
            "semantics": {"key": "strict: page_type+keywords+page+asin+options+sub_page", "coarse_key": "page_type+asin+options+sub_page", "deterministic_env": True, "replay_verified": True}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',required=True)
    p.add_argument('--episodes',required=True)
    p.add_argument('--env-url',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--reps',type=int,default=16)
    a=p.parse_args();goals=load_goals(a.tasks)
    if a.reps<1:raise ValueError('Repetitions must be positive')
    health=http(a.env_url.rstrip('/')+'/health')
    if not health.get('ok') or health.get('seed')!=42 or health.get('goal_shuffle_seed')!=233 or health.get('observation_mode')!='text_rich':
        raise ValueError('Replay environment must match the collection protocol')
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new or empty output directory')
    for name in ['entries','graphs','sources']:(out/name).mkdir(parents=True,exist_ok=True)
    tasks=[]
    for goal in goals:
        traces=[];instruction=None
        for rep in range(a.reps):
            path=Path(a.episodes)/f'g{goal:05d}_r{rep:02d}.json'
            ep=json.loads(path.read_text())
            if ep['goal']!=goal or ep['rep']!=rep or ep.get('technical_incomplete') or not ep['steps']:
                raise ValueError('Incomplete or mismatched source')
            if instruction is not None and instruction!=ep['instruction']:raise ValueError('Goal instruction changed')
            instruction=ep['instruction']
            relative=f'sources/goal{goal}_r{rep:02d}.json'
            normalized_source(ep,str(out/relative))
            trace=trajectory_from_episode(ep,str(out/relative),a.env_url.rstrip('/'))
            if not trace['trusted_complete']:raise ValueError(f"Replay failed for goal {goal} rep {rep}: {trace['replay_failures']}")
            trace['source_path']=relative;traces.append(trace)
        graph=build_graph(goal,instruction,traces);relative=f'graphs/goal{goal}.json.gz'
        with gzip.open(out/relative,'wt') as f:json.dump(graph,f,ensure_ascii=False)
        gamefile=f'goal{goal}';task_hash=hashlib.sha256(gamefile.encode()).hexdigest()[:16]
        entry={'schema_version':'webshop_catalog_entry_v1','gamefile':gamefile,'task_id':task_hash,'status':'complete',
               'graph':{'path':relative,'sha256':sha_file(out/relative)},'stats':graph['stats']}
        (out/'entries'/f'{task_hash}.json').write_text(json.dumps(entry))
        tasks.append({'gamefile':gamefile,'goal':goal,'instruction':instruction,
                      'source_success_reps':[t['rep'] for t in traces if t['source_won']],
                      'source_files':[{'rep':t['rep'],'path':t['source_path'],'sha256':t['source_sha256'],
                                       'source_won':t['source_won'],'expected_missing':False} for t in traces]})
    (out/'MANIFEST.json').write_text(json.dumps({'schema_version':'webshop_catalog_manifest_v1','tasks':tasks,'K':a.reps}))
    (out/'READY.json').write_text(json.dumps({'tasks':len(tasks),'K':a.reps,'all_sources_replayed':True}))
    print(out/'MANIFEST.json')


if __name__=='__main__':main()
