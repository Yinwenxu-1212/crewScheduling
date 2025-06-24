# environment.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import collections
import random

import config
from utils import RuleChecker, DataHandler, identify_duties_and_cycles, calculate_final_score

class CrewRosteringEnv:
    def __init__(self, data_handler: DataHandler):
        self.dh = data_handler
        self.rule_checker = RuleChecker(self.dh)
        self.crews = self.dh.data['crews'].to_dict('records')
        self.unified_tasks_df = self.dh.data['unified_tasks']
        self.planning_start_dt = pd.to_datetime(config.PLANNING_START_DATE)

    def reset(self):
        self.roster_plan = collections.defaultdict(list)
        self.crew_states = {c['crewId']: self._get_initial_crew_state(c) for c in self.crews}
        self.unassigned_flight_ids = set(self.dh.data['flights']['id'])
        for crew in self.crews:
            ground_duties = self.dh.crew_ground_duties.get(crew['crewId'], [])
            if ground_duties: self.roster_plan[crew['crewId']].extend(ground_duties)
        self.total_steps = 0
        self.decision_priority_queue = self._build_priority_queue()
        observation, info = self._get_observation()
        return observation, info

    def _get_initial_crew_state(self, crew):
        last_task = max(self.dh.crew_ground_duties.get(crew['crewId'], []), key=lambda x: x['endTime'], default=None)
        return {'last_task_end_time': last_task['endTime'] if last_task else self.planning_start_dt, 'last_location': last_task['airport'] if last_task else crew['stayStation']}

    def _build_priority_queue(self):
        priorities = []
        unassigned_departures = self.dh.data['flights'][self.dh.data['flights']['id'].isin(self.unassigned_flight_ids)].groupby('depaAirport')['id'].count()
        for i, crew in enumerate(self.crews):
            crew_state = self.crew_states[crew['crewId']]
            score = unassigned_departures.get(crew_state['last_location'], 0)
            score -= (crew_state['last_task_end_time'] - self.planning_start_dt).total_seconds() / (3600 * 24)
            priorities.append((score, i))
        priorities.sort(key=lambda x: x[0], reverse=True)
        return collections.deque([idx for score, idx in priorities])

    def _get_observation(self):
        if not self.decision_priority_queue: return None, {'valid_actions': []}
        self.current_crew_idx = self.decision_priority_queue[0]
        crew_info = self.crews[self.current_crew_idx]
        crew_state = self.crew_states[crew_info['crewId']]
        state_vector = self._build_state_vector(crew_state, crew_info)
        valid_actions, relaxed_actions = self._get_valid_actions(crew_info, crew_state)
        actions_to_consider = valid_actions if valid_actions else relaxed_actions
        action_features = np.zeros((config.MAX_CANDIDATE_ACTIONS, config.ACTION_DIM))
        action_mask = np.zeros(config.MAX_CANDIDATE_ACTIONS, dtype=np.uint8)
        for i, action in enumerate(actions_to_consider):
            action_mask[i] = 1
            action_features[i, :] = self._task_to_feature_vector(action, crew_state, crew_info)
        observation = {'state': state_vector, 'action_features': action_features, 'action_mask': action_mask}
        info = {'valid_actions': actions_to_consider}
        return observation, info
        
    def _build_state_vector(self, crew_state, crew_info):
        vec = np.zeros(config.STATE_DIM)
        now = crew_state['last_task_end_time']
        vec[0] = (now - self.planning_start_dt).total_seconds() / (3600*24*7)
        vec[1], vec[2] = now.hour / 24.0, now.weekday() / 7.0
        vec[3] = 1 if crew_state['last_location'] == crew_info['base'] else 0
        total_flights = len(self.dh.data['flights'])
        vec[4] = len(self.unassigned_flight_ids) / total_flights if total_flights > 0 else 0
        return vec

    def _get_valid_actions(self, crew_info, crew_state):
        last_loc, last_time = crew_state['last_location'], crew_state['last_task_end_time']
        potential_tasks_df = self.unified_tasks_df[(self.unified_tasks_df['startTime'] > last_time) & (self.unified_tasks_df['depaAirport'] == last_loc)].copy()
        all_actions, current_roster = [], self.roster_plan[crew_info['crewId']]
        for _, task_series in potential_tasks_df.iterrows():
            task_id = task_series['taskId']
            if task_series['type'] == 'flight':
                if task_id in self.unassigned_flight_ids and task_id in self.dh.crew_leg_map.get(crew_info['crewId'], set()):
                    all_actions.append({**task_series.to_dict(), 'type': 'flight'})
                all_actions.append({**task_series.to_dict(), 'type': 'positioning_flight'})
            else: all_actions.append(task_series.to_dict())
        valid_actions, relaxed_actions = [], []
        for action in all_actions:
            is_valid, violated_rules = self._is_incrementally_valid(current_roster, action, crew_info, crew_state)
            if is_valid: valid_actions.append(action)
            elif any('minor' in r for r in violated_rules):
                action['violation_penalty'] = len(violated_rules)
                relaxed_actions.append(action)
        
        valid_actions.sort(key=lambda x: x.get('flyTime',0), reverse=True)
        relaxed_actions.sort(key=lambda x: (-x.get('violation_penalty', 0), x.get('flyTime',0)), reverse=True)
        if valid_actions: return valid_actions[:config.MAX_CANDIDATE_ACTIONS], []
        return [], relaxed_actions[:config.MAX_CANDIDATE_ACTIONS]

    def _is_incrementally_valid(self, roster, new_task, crew_info, crew_state):
        violated_rules = set()
        if new_task.get('type') == 'flight' and new_task['taskId'] not in self.dh.crew_leg_map.get(crew_info['crewId'], set()): return False, {'hard_violation_qualification'}
        for task in roster:
            if not (new_task['endTime'] <= task.get('startTime', config.PLANNING_END_DATE) or new_task['startTime'] >= task.get('endTime', config.PLANNING_START_DATE)): return False, {'hard_violation_overlap'}
        temp_roster = sorted(roster + [new_task], key=lambda x: x['startTime'])
        ground_duties = self.dh.crew_ground_duties.get(crew_info['crewId'], [])
        temp_duties, _ = identify_duties_and_cycles(temp_roster, ground_duties)
        if temp_duties:
            last_duty = temp_duties[-1]
            flight_duty_tasks = [t for t in last_duty if t.get('type') in ['flight', 'positioning_flight', 'positioning_bus']]
            if flight_duty_tasks:
                if sum(t.get('flyTime', 0) for t in flight_duty_tasks) / 60.0 > 8: violated_rules.add('minor_violation_fly_time')
                duty_start = flight_duty_tasks[0]['startTime']
                flight_tasks_in_duty = [t for t in flight_duty_tasks if 'flight' in t.get('type','')]
                if flight_tasks_in_duty:
                    duty_end = flight_tasks_in_duty[-1]['endTime']
                    if (duty_end - duty_start) > timedelta(hours=12): violated_rules.add('minor_violation_duty_duration')
        return len(violated_rules) == 0, violated_rules
    
    def _task_to_feature_vector(self, task, crew_state, crew_info):
        vec = np.zeros(config.ACTION_DIM)
        vec[0] = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
        vec[1] = task.get('flyTime', 0) / 60.0
        vec[2] = 1 if task.get('type') == 'flight' else 0
        vec[3] = 1 if 'positioning' in task.get('type', '') else 0
        vec[4] = 1 if task.get('arriAirport') == crew_info['base'] else 0
        vec[5] = -task.get('violation_penalty', 0)
        next_location = task.get('arriAirport')
        unassigned_departures_next = self.dh.data['flights'][(self.dh.data['flights']['id'].isin(self.unassigned_flight_ids)) & (self.dh.data['flights']['depaAirport'] == next_location)].shape[0]
        vec[6] = unassigned_departures_next / 10.0
        return vec

    def step(self, action_idx):
        reward = 0
        if not self.decision_priority_queue: return None, 0, True, False, {'valid_actions': []}
        
        self.current_crew_idx = self.decision_priority_queue.popleft()
        crew_info = self.crews[self.current_crew_idx]
        crew_id = crew_info['crewId']
        
        _, info = self._get_observation()
        actions_to_consider = info['valid_actions']
        
        if action_idx < 0 or action_idx >= len(actions_to_consider):
            penalty_multiplier = 1 + len(actions_to_consider) / 10.0
            reward -= 1.0 * penalty_multiplier
        else:
            task = actions_to_consider[action_idx]
            self.roster_plan[crew_id].append(task)
            task_type = task.get('type', '')
            if task_type == 'flight':
                if task['taskId'] in self.unassigned_flight_ids: self.unassigned_flight_ids.remove(task['taskId'])
            self.crew_states[crew_id]['last_task_end_time'] = task['endTime']
            self.crew_states[crew_id]['last_location'] = task['arriAirport']
            reward += self._calculate_immediate_reward(task, crew_info)
            reward -= task.get('violation_penalty', 0) * config.PENALTY_RULE_VIOLATION * 0.2

        self.total_steps += 1
        
        if self.total_steps % len(self.crews) == 0: self.decision_priority_queue = self._build_priority_queue()
        
        terminated = not self.unassigned_flight_ids
        truncated = not self.decision_priority_queue or self.total_steps >= (len(self.crews) * config.MAX_STEPS_PER_CREW * 1.5)
        
        if terminated or truncated:
            reward += calculate_final_score(self.roster_plan, self.dh)
            observation, info = None, {'valid_actions': []}
        else:
            observation, info = self._get_observation()
        
        return observation, reward, terminated, truncated, info

    def _calculate_immediate_reward(self, task, crew_info):
        reward = 0
        task_type = task.get('type', '')
        if task_type == 'flight':
            reward += config.IMMEDIATE_COVERAGE_REWARD
            reward += (task.get('flyTime', 0) / 60.0) * 0.5
        elif 'positioning' in task_type:
            reward += config.PENALTY_POSITIONING
        if task.get('arriAirport') == crew_info['base']:
            reward += 0.5
        return reward