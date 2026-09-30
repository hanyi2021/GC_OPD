"""Attach retrospective teacher prompts without modifying sampled student tokens."""
import importlib
import os


def finalize_teacher_prompts(tokenizer, trajectory, *, environment):
    steps = trajectory["steps"]
    originals = [(list(t["prompt_ids"]), list(t["response_ids"]),
                  list(t["rollout_logprobs"]), t["raw_output"]) for t in steps]
    if environment not in ('scienceworld', 'alfworld', 'webshop'):
        raise ValueError('Unknown environment')
    prefix = 'WS' if environment == 'webshop' else 'SW'
    module = importlib.import_module(os.environ.get(prefix+'_POSTHOC_REFERENCE_MODULE', f'gcopd.{environment}.references.render'))
    prompt_budget = int(os.environ.get(prefix+'_TEACHER_PROMPT_BUDGET', '260095' if environment == 'webshop' else '40000'))
    model_budget = int(os.environ.get(prefix+'_TEACHER_MAX_MODEL_LEN', '262144' if environment == 'webshop' else '40513'))
    references = module.finalize_trajectory(
        tokenizer, trajectory["gamefile"], trajectory,
        catalog_root=os.environ.get(prefix+"_PHYSICAL_GRAPH_ROOT"), prompt_budget=prompt_budget,
    )
    if len(references) != len(steps):
        raise ValueError("Trajectory finalizer must return one teacher reference per student decision")
    prepared = []
    for turn, original, reference in zip(steps, originals, references):
        assert (turn["prompt_ids"], turn["response_ids"], turn["rollout_logprobs"], turn["raw_output"]) == original, "Finalizer modified sampled student data"
        ids = list(reference["teacher_prompt_ids"])
        if not ids or len(ids) > prompt_budget:
            raise ValueError("Teacher prompt exceeds its non-truncating budget")
        if len(ids) + len(turn["response_ids"]) + 1 > model_budget:
            raise ValueError("Teacher prompt plus unchanged response exceeds teacher model context")
        prepared.append((ids, dict(reference["meta"])))
    for turn, (ids, meta) in zip(steps, prepared):
        turn["teacher_prompt_ids"] = ids
        turn["graph_reference"] = meta

