"""Vanilla OPD: teacher sees exactly the student's prompt (no retrospective reference)."""
def finalize_trajectory(tokenizer, gamefile, trajectory, *, catalog_root=None, prompt_budget=40000):
    return [{"teacher_prompt_ids": list(step["prompt_ids"]), "meta": {"protocol": "vanilla_no_reference", "case": "vanilla", "student_step": t, "teacher_prompt_tokens": len(step["prompt_ids"]), "reference_text": ""}} for t, step in enumerate(trajectory["steps"])]
