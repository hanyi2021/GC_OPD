"""Execute planner actions and replay their recorded observations and outcomes."""
PLANNER_KIND="oracle_planner_new_replay"
PLANNER_REP=1000000

def execute_saved_plan(env, gamefile, primary, save_snapshot, codec, compare, on_transition=None):
    """No planning API, action edits, menu rebinding, free retries, or 30-step cut."""
    task, variation, simplification = gamefile.split('|')
    env.load(task, int(variation), simplification, generateGoldPath=False)
    observation, info = env.reset()
    focus = None
    current = codec.capture_state(env, focus, observation, info['score'])
    if not current.get('capture_ok'):
        raise ValueError('Planner replay initial state capture failed')
    current['verified'] = True
    states = [save_snapshot(current, 'planner_initial')]
    steps = []
    initial_observation = observation
    first_success = None
    done = False
    status = 'plan_exhausted_without_success'
    for i, action in enumerate(primary['planned_actions'][:200]):
        before_observation, before_tick = observation, env.get_num_moves()
        observation, _, done, info = env.step(action)
        if on_transition:
            on_transition({'decision': i+1, 'action': action, 'observation_before': before_observation,
                           'feedback': observation, 'score': info['score'], 'done': bool(done),
                           'tick_before': before_tick, 'tick_after': env.get_num_moves()})
        focus = codec.update_focus(focus, observation)
        after = codec.capture_state(env, focus, observation, info['score'], previous_interaction=current['interaction'])
        if not after.get('capture_ok'):
            raise ValueError('Planner replay successor state capture failed at decision ' + str(i+1))
        after['verified'] = True
        step = {'step': i+1, 'decision': i+1, 'step_index': i, 'decision_index': i,
                'action': action, 'executed_action': action, 'semantic_action': codec.parse_menu(before_observation).get(action, action),
                'observation_before': before_observation, 'feedback': observation, 'score': info['score'],
                'source_score': info['score'], 'done': bool(done), 'tick_before': before_tick, 'tick_after': env.get_num_moves(),
                'source_terminal_failure': bool(done and info['score'] < 100), 'env_step_called': True, 'decision_cost': 1,
                'effects': codec.meaningful_changes(current['objects'], after['objects'])}
        steps.append(step)
        states.append(save_snapshot(after, 'planner_after'))
        current = after
        if info['score'] >= 100:
            first_success = i+1
            status = 'success_within_30' if first_success <= 30 else 'success_after_30'
            break
        if done:
            status = 'environment_terminated_without_success'
            break
    else:
        if len(steps) >= 200:
            status = 'execution_capped_at_200_decisions'
    mismatches = []
    for i in range(max(len(steps), len(primary['steps']))):
        actual = steps[i] if i < len(steps) else None
        expected = primary['steps'][i] if i < len(primary['steps']) else None
        if actual is None or expected is None:
            mismatches.append({'decision': i+1, 'kind': 'executed_length_difference'})
            continue
        checks = {'action': actual['action'] == expected['action'],
                  'observation_before_exact': actual['observation_before'] == expected['observation_before'],
                  'feedback_exact': actual['feedback'] == expected['feedback'],
                  'feedback_equivalent': compare(actual['feedback'], expected['feedback'])['equal'],
                  'score': actual['score'] == expected['score'], 'done': actual['done'] == expected['done'],
                  'tick_before': actual['tick_before'] == expected['tick_before'], 'tick_after': actual['tick_after'] == expected['tick_after']}
        if not all(checks.values()):
            mismatches.append({'decision': i+1, 'checks': checks})
    primary_comparison = {'primary_first_success_decision': primary['first_success_decision'],
        'replay_first_success_decision': first_success,
        'initial_observation_exact': initial_observation == primary['initial_observation'],
        'initial_physical_key_equal': states[0]['key'] == primary['initial_physical_key'],
        'primary_decisions': len(primary['steps']), 'replay_decisions': len(steps),
        'step_mismatches': mismatches,
        'all_recorded_actions_feedback_scores_ticks_done_equal': not mismatches and initial_observation == primary['initial_observation'],
        'per_step_physical_equality_to_primary': 'unavailable_without_primary_state_snapshots',
        'always_registered_as_new_actual_replay': True}
    return {'gamefile': gamefile, 'source_kind': PLANNER_KIND, 'rep': PLANNER_REP,
            'initial_observation': initial_observation, 'initial_physical_key': states[0]['key'],
            'planned_actions': primary['planned_actions'], 'executed_actions': [s['action'] for s in steps],
            'states': states, 'steps': steps, 'attempt_branches': [], 'replay_failures': [],
            'trusted_complete': True, 'won': first_success is not None, 'source_won': first_success is not None,
            'final_score': states[-1]['score'], 'first_success_decision': first_success,
            'success_within_30': first_success is not None and first_success <= 30,
            'status': status, 'length': len(steps), 'primary_comparison': primary_comparison,
            'no_model_thought': True, 'new_planning_calls': 0, 'new_model_calls': 0,
            'unexecuted_plan_tail_is_not_transition_evidence': True}
