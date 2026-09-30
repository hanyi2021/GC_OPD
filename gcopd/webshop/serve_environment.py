"""WebShop text-environment state and replay service; no browser automation."""
import argparse, hashlib, json, os, sys, threading, time, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

def log(*a):
    print(f"[ws_env_server {time.strftime('%H:%M:%S')}]", *a, file=sys.stderr, flush=True)
def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
def page_type(url):
    if url is None:
        return "start"
    for p in ("search_results", "item_page", "item_sub_page", "done"):
        if f"/{p}/" in url:
            return p
    return "start"
def describe(env):
    """从 SimServer 会话状态构造规范描述子（不含 goal 真值）。"""
    sid = env.session
    sess = SERVER.user_sessions.get(sid, {})
    url = env.browser.current_url
    pt = page_type(url)
    keywords = list(sess.get("keywords") or [])
    page = sess.get("page")
    asin = sess.get("asin")
    options = dict(sess.get("options") or {})
    sub_page = None
    if pt == "item_sub_page":
        parts = url.split("/")
        sub_page = parts[-2] if len(parts) >= 2 else None
    if pt == "start":
        keywords, page, asin, options = [], None, None, {}
    opts_sorted = sorted((str(k), str(v)) for k, v in options.items())
    strict = [pt, [str(k) for k in keywords], page, asin, opts_sorted, sub_page]
    coarse = [pt, asin, opts_sorted, sub_page]
    try:
        avail = env.get_available_actions()
    except Exception:
        avail = {"has_search_bar": False, "clickables": []}
    return {
        "page_type": pt, "keywords": keywords, "page": page, "asin": asin, "options": options,
        "sub_page": sub_page, "asins_visited": sorted(sess.get("asins") or []),
        "action_counts": dict(sess.get("actions") or {}),
        "done": bool(sess.get("done", False)), "reward": sess.get("reward"),
        "url": url, "strict_key": sha(strict), "coarse_key": sha(coarse),
        "available_actions": avail, "n_clickables": len(avail.get("clickables", [])),
        "capture_ok": True, "matchable": True,
    }
def instruction_of(env):
    sess = SERVER.user_sessions.get(env.session, {})
    return sess.get("goal", {}).get("instruction_text")
def new_env(eid):
    # server= 共享商品库；session_prefix 让同一 goal 的多个 episode 互不串会话
    return WebAgentTextEnv(observation_mode=args.observation_mode, server=SERVER, session_prefix=f"e{eid}_", seed=args.seed)
def do_reset(eid, goal):
    if eid in EPISODES:
        do_close(eid)
    env = new_env(eid)
    obs, _ = env.reset(session=int(goal))
    EPISODES[eid] = {"env": env, "goal": int(goal), "sid": env.session, "steps": 0, "t": time.time()}
    st = describe(env)
    return {"ok": True, "obs": obs, "instruction": instruction_of(env), "available_actions": st["available_actions"], "state": st, "goal": int(goal)}
def do_step(eid, action):
    ep = EPISODES[eid]
    env = ep["env"]
    sid = env.session
    obs, reward, done, _ = env.step(action)
    ep["steps"] += 1
    if done:
        # env.step 在 done 后已内部 reset 到随机新会话；终态从旧会话读
        sess = SERVER.user_sessions.get(sid, {})
        st = {"page_type": "done", "keywords": list(sess.get("keywords") or []), "page": sess.get("page"), "asin": sess.get("asin"),
              "options": dict(sess.get("options") or {}), "sub_page": None, "asins_visited": sorted(sess.get("asins") or []),
              "action_counts": dict(sess.get("actions") or {}), "done": True, "reward": float(reward), "url": None,
              "available_actions": {"has_search_bar": False, "clickables": []}, "n_clickables": 0, "capture_ok": True, "matchable": True}
        opts_sorted = sorted((str(k), str(v)) for k, v in st["options"].items())
        st["strict_key"] = sha(["done", st["asin"], opts_sorted])
        st["coarse_key"] = st["strict_key"]
        st["verbose_info"] = sess.get("verbose_info")
        env.session = sid  # 供 close 清理
    else:
        st = describe(env)
    return {"ok": True, "obs": obs, "reward": float(reward), "done": bool(done), "available_actions": st["available_actions"], "state": st}
def do_close(eid):
    ep = EPISODES.pop(eid, None)
    if ep:
        SERVER.user_sessions.pop(ep["sid"], None)
        # __init__ 时那个随机会话也清掉
        for k in [k for k in SERVER.user_sessions if k.startswith(f"e{eid}_")]:
            SERVER.user_sessions.pop(k, None)
    return {"ok": True}
def do_replay(goal, actions):
    eid = f"replay_{time.time_ns()}"
    out = do_reset(eid, goal)
    states = [out["state"]]; obs = [out["obs"]]; rewards = []; dones = []; avail = [out["available_actions"]]
    try:
        for a in actions:
            r = do_step(eid, a)
            states.append(r["state"]); obs.append(r["obs"]); rewards.append(r["reward"]); dones.append(r["done"]); avail.append(r["available_actions"])
            if r["done"]:
                break
    finally:
        do_close(eid)
    return {"ok": True, "goal": goal, "instruction": out["instruction"], "states": states, "obs": obs, "rewards": rewards, "dones": dones, "available_actions": avail, "n_executed": len(rewards)}
class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a):  # 静音访问日志
        pass
    def _send(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/health":
            with LOCK:
                self._send(200, {"ok": True, "goals": len(SERVER.goals), "products": len(SERVER.all_products), "active_episodes": len(EPISODES), "seed": args.seed, "goal_shuffle_seed": args.goal_shuffle_seed if args.goal_shuffle_seed is not None else args.seed, "observation_mode": args.observation_mode, "num_products": args.num_products, "pid": os.getpid()})
        elif u.path == "/goal":
            idx = int(parse_qs(u.query)["idx"][0])
            g = SERVER.goals[idx]
            self._send(200, {"ok": True, "idx": idx, "instruction_text": g["instruction_text"], "name": SERVER.product_item_dict[g["asin"]]["Title"], "asin": g["asin"], "attributes": g["attributes"], "price_upper": g["price_upper"], "goal_options": g["goal_options"], "category": g.get("category"), "query": g.get("query"), "product_category": g.get("product_category")})
        else:
            self._send(404, {"ok": False, "error": "unknown path"})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n) or b"{}")
        try:
            with LOCK:
                if self.path == "/reset":
                    resp = do_reset(str(req["episode"]), int(req["goal"]))
                elif self.path == "/step":
                    resp = do_step(str(req["episode"]), str(req["action"]))
                elif self.path == "/close":
                    resp = do_close(str(req["episode"]))
                elif self.path == "/replay":
                    resp = do_replay(int(req["goal"]), list(req["actions"]))
                else:
                    resp = {"ok": False, "error": "unknown path"}
            self._send(200, resp)
        except Exception:
            self._send(500, {"ok": False, "error": traceback.format_exc()})
class _Srv(ThreadingHTTPServer):
    # 默认 listen backlog 只有 5，上千并发连接会被内核直接 reset（实测 7–20% 的 episode 报 Connection reset by peer）
    request_queue_size = 4096
    daemon_threads = True
    allow_reuse_address = True

def main():
    global args, SERVER, BASE, LOCK, EPISODES, WebAgentTextEnv
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--webshop-root", required=True)
    ap.add_argument("--host",default="127.0.0.1")
    ap.add_argument("--observation-mode", default="text_rich", choices=["text_rich"])
    ap.add_argument("--num-products", default="all", help="all | 1000 | 100000 (小索引调试)")
    ap.add_argument("--human-goals", type=int, default=1)
    ap.add_argument("--goal-shuffle-seed", type=int, default=233, help="goal 洗牌种子；默认233与官方WebShop一致。复现旧内部实验须显式传42；--seed仅控制其他环境随机性。")
    args = ap.parse_args()
    sys.path.insert(0, args.webshop_root)
    import gym  # noqa
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv  # noqa
    num_products = None if args.num_products == "all" else int(args.num_products)
    if num_products is None:
        fp, ap_ = f"{args.webshop_root}/data/items_shuffle.json", f"{args.webshop_root}/data/items_ins_v2.json"
    else:
        fp, ap_ = f"{args.webshop_root}/data/items_shuffle_1000.json", f"{args.webshop_root}/data/items_ins_v2_1000.json"
    t0 = time.time()
    import web_agent_site.envs.web_agent_text_env as _wte
    if args.goal_shuffle_seed is not None:
        _OrigSim = _wte.SimServer
        class _Sim(_OrigSim):
            def __init__(self, seed, *a, **k):
                # 价格上限采样已由 WebAgentTextEnv.__init__ 的 random.seed(env seed) 决定；这里只让 goal 洗牌用指定种子（233 = 原始 WebShop 顺序）
                super().__init__(args.goal_shuffle_seed, *a, **k)
        _wte.SimServer = _Sim
    BASE = WebAgentTextEnv(observation_mode=args.observation_mode, file_path=fp, attr_path=ap_,
                           num_products=num_products, human_goals=bool(args.human_goals), seed=args.seed)
    SERVER = BASE.server
    log(f"base env ready in {time.time()-t0:.1f}s; goals={len(SERVER.goals)} products={len(SERVER.all_products)} seed={args.seed} mode={args.observation_mode}")
    LOCK = threading.Lock()
    EPISODES = {}  # episode id -> dict(env=..., goal=..., sid=..., steps=int)
    srv = _Srv((args.host, args.port), H)
    log(f"listening on :{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
