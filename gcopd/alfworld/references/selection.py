"""Shared ALF GC reference selection, decision 008 (CPU-only).

Catalog validation and source rendering remain specific to each payload.
"""

def key(turn):
    state=turn.get('physical_state_before') or {}
    return state.get('key') if state.get('capture_ok') else None

def positive(step):
    return bool(step['env_step_called']) and not step['format_error'] and not step['rejection']

def candidates(cat,k):
    if not k:return []
    result=[]
    for v in cat['visits'].get(k,[]):
        src=cat['sources'][v['rep']];i=v['index']
        if src['won'] and i<len(src['steps']) and positive(src['steps'][i]):
            result.append((len(src['steps'])-i,src['rep'],i,src))
    return sorted(result,key=lambda x:x[:3])

def anchored_success(cat,turns,t):
    for j in range(t,-1,-1):
        found=candidates(cat,key(turns[j]))
        if found:return found[0],j,True
    full=[(len(s['steps']),s['rep'],0,s) for s in cat['sources'] if s['won']]
    return (min(full,key=lambda x:x[:3]),None,False) if full else (None,None,False)

def failed_source(cat,turns,t):
    for j in range(t,-1,-1):
        options=[]
        for v in cat['visits'].get(key(turns[j]),[]):
            s=cat['sources'][v['rep']]
            if not s['won']:
                options.append((abs(v['index']-j),len(s['steps']),s['rep'],v['index'],s))
        if options:
            best=min(options,key=lambda x:x[:4]);return best[-1],j,best[3]
    failed=[s for s in cat['sources'] if not s['won']]
    return (min(failed,key=lambda s:(len(s['steps']),s['rep'])),None,None) if failed else (None,None,None)

def select(cat,trajectory,t):
    turns=trajectory['steps'];matched=key(turns[t]) in cat['visits'];won=bool(trajectory['won'])
    success=candidates(cat,key(turns[t]));meta={'current_turn':t+1,'matched':matched,'student_final_won':won};refs=[]
    if not matched and won:
        meta['case']='outside_self_success';return meta,refs
    # A successful student only competes when a current external success exists.
    # Otherwise retain available historical/same-task successes and failures.
    if matched and success:
        meta['case']='matched_success_suffix'
        self_cost=len(turns)-t
        if success and (not won or success[0][0]<=self_cost):
            cost,rep,idx,src=success[0];refs.append({'kind':'success','source':src,'source_anchor':idx,'student_anchor':t,'aligned':True});meta['selected_remaining_cost']=cost
        else:meta['selected_source']='student_self';meta['selected_remaining_cost']=self_cost
        return meta,refs
    meta['case']='matched_no_recorded_success' if matched else 'outside_student_failure'
    fail,j,i=failed_source(cat,turns,t)
    if fail:refs.append({'kind':'failure','source':fail,'source_anchor':i,'student_anchor':j,'aligned':j is not None})
    success,j,aligned=anchored_success(cat,turns,t)
    if success:
        cost,rep,i,src=success;refs.append({'kind':'success','source':src,'source_anchor':i,'student_anchor':j,'aligned':aligned})
    return meta,refs


def ordered_references(refs):
    """Render complete successes before failures without mutating selection.

    The renderer emits the complete student execution before these records.
    Stable ordering preserves source identity and ordering within each kind.
    """
    return sorted(refs, key=lambda ref: ref['kind'] != 'success')
