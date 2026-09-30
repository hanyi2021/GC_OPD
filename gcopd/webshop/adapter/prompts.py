from gcopd.webshop.protocol import INVALID_FEEDBACK, format_obs, format_avail, build_prompt, parse_action, rejected_by_environment

def is_executable(action, available):
    return not rejected_by_environment(action, available)

def core_state(st):
    """env 服务的描述子 → 四类参考核心需要的 state 记录（key/interaction/capture_ok/matchable）。"""
    return {"key": st["strict_key"], "node_key": st["strict_key"], "coarse_key": st["coarse_key"], "physical_hash": st["coarse_key"],
            "control": {"page_type": st["page_type"], "asin": st.get("asin"), "options": sorted((str(k), str(v)) for k, v in (st.get("options") or {}).items())},
            "interaction": {"mode": "normal", "options": {}}, "capture_ok": True, "matchable": True, "codec_version": "webshop_url_session_v1",
            **{"ws_" + k: st.get(k) for k in ["page_type", "keywords", "page", "asin", "options", "sub_page", "asins_visited", "done", "reward", "url", "n_clickables", "verbose_info"]}}
