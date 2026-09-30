"""Generate and replay WebShop training-only oracle trajectories for graph augmentation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import uuid
from gcopd.common.io import new_output, write_json, digest
from gcopd.webshop.collect import load_goals
from gcopd.webshop import planner as p


ORACLE_KINDS = {
    'privileged_goal_oracle_v1', 'privileged_goal_oracle_query_search_v2',
    'privileged_goal_oracle_equivalent_product_v3', 'privileged_goal_oracle_broad_search_v4',
    'privileged_full_catalog_oracle_v5', 'privileged_catalog_query_filter_oracle_v6',
}


def verify_episode(ep, env):
    steps = ep['steps']
    if not steps or len(steps)>15 or ep.get('model') is not None or not ep.get('complete'):
        raise ValueError('Expected a completed oracle execution of 1–15 decisions')
    if ep.get('mode') != 'train' or ep.get('tag') not in ORACLE_KINDS:
        raise ValueError('Expected a training oracle with a recognized source kind')
    if any(s.get('executed') is not True or s.get('format_error') for s in steps):
        raise ValueError('Oracle sources must contain only actually executed actions')
    replay = p.http(env+'/replay', {'goal':ep['goal'], 'actions':[s['action'] for s in steps]})
    if (replay.get('ok') is not True or replay.get('instruction') != ep['instruction'] or
        replay['obs'] != [steps[0]['obs']]+[s['feedback'] for s in steps] or
        [s['strict_key'] for s in replay['states']] !=
            [steps[0]['state_before']['strict_key']]+[s['state_after']['strict_key'] for s in steps] or
        replay['dones'] != [s['done'] for s in steps] or
        len(replay['rewards']) != len(steps) or
        any(not math.isclose(a, b['reward'], rel_tol=0, abs_tol=1e-9) for a,b in zip(replay['rewards'],steps))):
        raise ValueError('Oracle replay differs from the recorded execution')
    score = steps[-1]['reward'] if steps[-1]['done'] else 0.
    if (not math.isfinite(score) or ep['done'] != steps[-1]['done'] or
        not math.isclose(ep['score'], score, rel_tol=0, abs_tol=1e-9) or
        ep['won'] != (bool(ep['done']) and score>=1-1e-9)):
        raise ValueError('Inconsistent oracle outcome')
    ep.update(replay_verified=True, technical_incomplete=False, source_role='training',
              source_kind=ep['tag'])
    return ep


def refinement_goals(goals, initial_outcomes, requested=None):
    selected = set(goals) if requested is None else set(requested)
    if not selected.issubset(goals):
        raise ValueError('Refinement tasks must be a subset of the training task list')
    return [goal for goal in goals if goal in selected and not initial_outcomes[goal]]


def clean_queries(values):
    return list(dict.fromkeys(' '.join(str(q).replace('[',' ').replace(']',' ').split()).lower()
                             for q in values if q))


def query_candidates(truth):
    title=' '.join(truth['name'].replace('[',' ').replace(']',' ').split())
    words=title.split();stop={'for','with','the','a','an','in','of','and','to','by','set','pack'}
    core=[x for x in words if x.lower() not in stop]
    values=[truth['asin'],f'"{title}"',truth.get('query',''),' '.join(words[:4]),
        ' '.join(words[:8]),' '.join(words[-8:]),' '.join(core[:10]),
        ' '.join(sorted(core,key=len,reverse=True)[:6]),' '.join(words[:2]+words[-4:])]
    return list(dict.fromkeys(p.norm(q) for q in values if q))


def run_refinements(goal, truth, env, save):
    for index, query in enumerate(query_candidates(truth)):
        ep = p.query_attempt(goal,truth,query,env,index)
        save('query',ep)
        if ep['won']:
            return ep
        if ep['done']:
            break
    for stage, discover, attempt in [('equivalent',p.discover_equivalent,p.equivalent_attempt),
                                     ('broad',p.discover_broad,p.broad_attempt)]:
        if stage=='equivalent':
            queries=clean_queries([truth['name'],truth.get('query',''),' '.join(truth['attributes'])])
        else:
            raw=truth['instruction_text'].split(', and price lower than')[0]
            options=' '.join(truth.get('goal_options',[]));attrs=' '.join(truth['attributes']);q=truth.get('query','')
            queries=clean_queries([raw,q+' '+options,q+' '+attrs+' '+options,
                ' '.join(truth['name'].split()[:6])+' '+options]+
                ['<a> '+x for x in truth['attributes']]+['<q> '+q])
        seen=set()
        for query in queries:
            for asin,page in discover(goal,truth,env,query):
                if asin in seen:
                    continue
                seen.add(asin)
                ep=attempt(goal,truth,env,query,asin,page,len(seen))
                save(stage,ep)
                if ep and ep['won']:
                    return ep
    return None


def run_catalog(goal, candidates, env, save):
    by_asin={c['asin'].lower():c for c in candidates}
    queries=list(dict.fromkeys(p.normalized(c['name'].replace('[',' ').replace(']',' '))
                 for c in sorted(candidates,key=lambda x:x['asin'])))
    tried=set()
    for query in queries:
        eid='catalog_probe_'+uuid.uuid4().hex
        found=[]
        try:
            p.http(env+'/reset',{'episode':eid,'goal':goal})
            state=p.http(env+'/step',{'episode':eid,'action':f'search[{query}]'})
            for page in range(1,4):
                found.extend((by_asin[x],page) for x in state['available_actions']['clickables']
                             if x in by_asin and x not in tried)
                if page==3 or 'next >' not in state['available_actions']['clickables']:
                    break
                state=p.http(env+'/step',{'episode':eid,'action':'click[next >]'})
        finally:
            p.http(env+'/close',{'episode':eid})
        for candidate,page in found:
            if candidate['asin'].lower() in tried:
                continue
            tried.add(candidate['asin'].lower())
            ep=p.catalog_attempt(goal,candidate,query,page,env,len(tried))
            save('catalog',ep)
            if ep and ep['won']:
                return ep
    return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tasks',required=True)
    ap.add_argument('--catalog',required=True,help='Replay-verified K16 teacher catalog')
    ap.add_argument('--env-url',required=True)
    ap.add_argument('--webshop-root',required=True,help='Official WebShop checkout and full data')
    ap.add_argument('--output',required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--refinement-tasks',help='Optional training-goal JSON list restricting v2-v6 refinement')
    ap.add_argument('--query-filter-tasks',help='Optional training-goal JSON list restricting v6 query filtering')
    a=ap.parse_args()
    goals=load_goals(a.tasks);env=a.env_url.rstrip('/')
    if a.workers<1:
        raise ValueError('workers must be positive')
    base=Path(a.catalog).resolve();manifest=json.loads((base/'MANIFEST.json').read_text())
    tasks={t['goal']:t for t in manifest['tasks']}
    if len(tasks)!=len(manifest['tasks']) or set(tasks)!=set(goals) or not (base/'READY.json').is_file():
        raise ValueError('Use the complete teacher catalog for exactly the supplied training goals')
    if any([s['rep'] for s in t['source_files']]!=list(range(16)) for t in tasks.values()):
        raise ValueError('Expected an unaugmented K16 teacher catalog')
    health=p.http(env+'/health')
    if not health.get('ok') or health.get('seed')!=42 or health.get('goal_shuffle_seed')!=233 or health.get('observation_mode')!='text_rich':
        raise ValueError('Use the seed-42, goal-shuffle-233 text_rich service')
    requested = load_goals(a.refinement_tasks) if a.refinement_tasks else None
    query_filter = set(load_goals(a.query_filter_tasks)) if a.query_filter_tasks else set(goals)
    if not query_filter.issubset(goals) or (requested is not None and not set(requested).issubset(goals)):
        raise ValueError('Optional strategy task lists must be subsets of training tasks')
    p.load_dependencies(a.webshop_root)
    out=new_output(a.output)
    for n in ['episodes','attempts','results','candidates']:(out/n).mkdir()
    write_json(out/'COLLECTION.json',{'goals':goals,'catalog_sha256':digest(base/'MANIFEST.json'),
        'protocol':'webshop_historical_oracle_v1_to_v6','max_steps':15,'teacher_K':16,
        'refinement_tasks':requested,'query_filter_tasks':sorted(query_filter),
        'refinement_scope':'Failed original oracle executions, independent of teacher outcomes'})
    truths={g:{**p.http(env+f'/goal?idx={g}'),'goal':g} for g in goals}
    if any(not t.get('name') for t in truths.values()):
        raise ValueError('Restart the environment service from this release to expose product titles')
    winners={};counters={g:0 for g in goals}
    def save_for(goal):
        def save(stage,ep):
            counters[goal]+=1
            write_json(out/'attempts'/f'g{goal:05d}_{counters[goal]:05d}_{stage}.json',
                       ep if ep is not None else {'goal':goal,'stage':stage,'no_feasible_purchase':True})
        return save
    def accept(goal,ep):
        if not ep['won']:
            raise ValueError('Only successful refinements may be admitted')
        verify_episode(ep,env)
        ep['rep']=17
        write_json(out/'episodes'/f'g{goal:05d}_r17.json',ep)
        winners[goal]=ep['tag']
    initial_outcomes={}
    def generate(goal):
        initial=verify_episode(p.initial(goal,truths[goal],env),env)
        write_json(out/'episodes'/f'g{goal:05d}_r16.json',initial)
        initial_outcomes[goal]=initial['won']
    with ThreadPoolExecutor(a.workers) as pool:
        list(pool.map(generate,goals))
    refine=refinement_goals(goals,initial_outcomes,requested)
    def refine_one(goal):
        ep=run_refinements(goal,truths[goal],env,save_for(goal))
        if ep:accept(goal,ep)
        return not bool(ep)
    with ThreadPoolExecutor(a.workers) as pool:
        missing=[g for g,needed in zip(refine,pool.map(refine_one,refine)) if needed]
    if missing:
        candidates,count=p.catalog_candidates(a.webshop_root,[truths[g] for g in missing])
        if count!=health['products']:
            raise ValueError('Local product catalog differs from the environment service')
        for goal in missing:
            write_json(out/'candidates'/f'g{goal:05d}_optimistic.json',candidates[goal])
            ep=run_catalog(goal,candidates[goal],env,save_for(goal))
            if ep:accept(goal,ep)
        remaining=[g for g in missing if g not in winners and g in query_filter]
        if remaining:
            exact,count=p.catalog_candidates(a.webshop_root,[truths[g] for g in remaining],exact=True)
            if count!=health['products']:
                raise ValueError('Local product catalog differs from the environment service')
            for goal in remaining:
                write_json(out/'candidates'/f'g{goal:05d}_exact.json',exact[goal])
                for query in sorted({c['query'] for c in exact[goal]}):
                    ep=p.query_filter_attempt(goal,exact[goal],query,env)
                    save_for(goal)('query_filter',ep)
                    if ep['won']:
                        accept(goal,ep)
                        break
    entries=[]
    for goal in goals:
        paths=[out/'episodes'/f'g{goal:05d}_r16.json']
        if goal in winners:paths.append(out/'episodes'/f'g{goal:05d}_r17.json')
        entries.append({'goal':goal,'sources':[{'path':str(x.relative_to(out)),'sha256':digest(x)} for x in paths]})
        write_json(out/'results'/f'g{goal:05d}.json',{'goal':goal,'refinement':winners.get(goal),
            'attempts':counters[goal]})
    write_json(out/'COMPLETE.json',{'tasks':entries,'catalog_sha256':digest(base/'MANIFEST.json'),
        'all_admitted_sources_replayed':True,'teacher_K':16,'refinements':len(winners)})
    print(out/'COMPLETE.json')


if __name__=='__main__':
    main()
