def finalize_trajectory(tokenizer,gamefile,trajectory,*,catalog_root=None,prompt_budget=10240):
    return [{'teacher_prompt_ids':list(step['prompt_ids']),'meta':{'protocol':'plain_opd_no_reference','reference_text':'','teacher_prompt_tokens':len(step['prompt_ids'])}} for step in trajectory['steps']]
