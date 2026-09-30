"""GC-OPD four-case references over replay-verified task-local ALFWorld PDDL visits."""
import hashlib, json
from functools import lru_cache
from pathlib import Path

@lru_cache(maxsize=128)
def catalog(root,gamefile):
    p=Path(root)/(hashlib.sha256(gamefile.encode()).hexdigest()+'.json')
    d=json.loads(p.read_text())
    assert d['gamefile']==gamefile and d['trusted'] and d['K']>=1
    assert all(s['trusted_complete'] for s in d['sources'])
    if d.get('planner_augmented'):
        assert d.get('teacher_K') == 16 and d.get('planner_max_success_steps') == 30
        assert len(d['sources']) == 16 + d.get('planner_sources', 0)
        assert d.get('planner_sources') in (0, 1)
        assert all(s['won'] and 1 <= len(s['steps']) <= 30 and s.get('source_kind') == 'planner'
                   for s in d['sources'][16:])
    assert len(d['sources']) == d['K']
    assert [s['rep'] for s in d['sources']]==list(range(d['K']))
    return d

from gcopd.alfworld.references.selection import key, positive, candidates, anchored_success, failed_source, select, ordered_references

def render(trajectory,t,meta,refs,compact=False):
    observations={}
    def obs(value):
        value=str(value)
        if not compact:return value
        if value in observations:return '[same observation as O'+str(observations[value])+']'
        n=len(observations)+1;observations[value]=n;return f'[O{n}] '+value
    def trajectory_lines(steps):
        out=[]
        for i,s in enumerate(steps):
            before=s.get('observation_before',s.get('current_obs',''))
            action=s.get('action') or '[no environment action: invalid format]'
            out.append(f"Step {i+1}:\nObservation: {obs(before)}\nAction: {action}\nFeedback: {obs(s['feedback'])}")
        return '\n'.join(out)
    lines=['<four_case_reference>','Training-only retrospective evidence. Recorded executions are references, not proof that a historical continuation can be executed from the current state.',f"Current scored decision: {t+1}. Retrieval case: {meta['case']}.",'Complete student execution:',trajectory_lines(trajectory['steps']),f"Student final outcome: {'SUCCESS' if trajectory['won'] else 'FAILURE'}. This outcome does not imply every action was good or bad."]
    for r in ordered_references(refs):
        s=r['source'];origin='PLANNER' if s.get('source_kind')=='planner' else 'teacher';lines.append(f"Recorded {origin} source rep={s['rep']}; complete verified outcome: {'SUCCESS' if s['won'] else 'FAILURE'}.")
        if r['aligned']:
            lines.append(f"Shared historical state: before student decision {r['student_anchor']+1}, before source decision {r['source_anchor']+1}. The full source prefix and suffix follow. This is a historical anchor, not an instruction to rewind the environment.")
        else:lines.append('Same-task historical reference, but no reliable shared state was found. Do not assume it connects to the current state.')
        lines.append(trajectory_lines(s['steps']))
        if not s['won']:lines.append('This complete source execution FAILED. Its actions are not automatically recommended.')
    if meta['case']=='matched_no_recorded_success':lines.append('No trusted successful continuation was recorded here; this does not prove the state is unsolvable.')
    lines.append('</four_case_reference>');return '\n\n'.join(lines)

def finalize_trajectory(tokenizer,gamefile,trajectory,*,catalog_root=None,prompt_budget=40000):
    if not catalog_root:raise ValueError('Missing frozen ALFWorld catalog')
    cat=catalog(catalog_root,gamefile);result=[]
    for t,turn in enumerate(trajectory['steps']):
        meta,refs=select(cat,trajectory,t)
        original=list(turn['prompt_ids']);end_token=tokenizer.convert_tokens_to_ids('<|im_end|>')
        positions=[i for i,x in enumerate(original) if x==end_token]
        if not positions:raise ValueError('Missing original user-message boundary')
        at=positions[-1]
        tail=tokenizer.decode(original[at:],skip_special_tokens=False)
        if not tail.startswith('<|im_end|>\n<|im_start|>assistant'):raise ValueError('Unexpected assistant header')
        for compact in [False,True]:
            text=render(trajectory,t,meta,refs,compact=compact)
            addition=tokenizer.encode('\n\n'+text+'\n',add_special_tokens=False)
            ids=original[:at]+addition+original[at:]
            if len(ids)<=prompt_budget:break
        else:raise ValueError(f'Full graph evidence exceeds {prompt_budget} tokens')
        result.append({'teacher_prompt_ids':ids,'meta':{**meta,'reference_text':text,'source_reps':[r['source']['rep'] for r in refs],'teacher_prompt_tokens':len(ids),'exact_repeat_compression':compact,'protocol':'alf_pddl_four_case_v1','selection_rule':'alf_reference_retention_v2'}})
    return result
