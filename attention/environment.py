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
        return {
            'last_task_end_time': last_task['endTime'] if last_task else self.planning_start_dt,
            'last_location': last_task['airport'] if last_task else crew['stayStation'],
            'duty_start_time': None,  # 当前值勤开始时间
            'duty_flight_time': 0,    # 值勤内飞行时间
            'duty_flight_count': 0,   # 值勤内航班数
            'duty_task_count': 0,     # 值勤内任务数
            'total_flight_hours': 0,  # 总飞行时间
            'total_positioning': 0,   # 总调机次数
            'total_away_overnights': 0,  # 总外站过夜
            'total_calendar_days': set(),  # 总日历天数
            'cost': 0                 # 累计成本
        }

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
        """
        构建状态向量，与attention_guided_subproblem_solver.py中的_extract_state_features保持一致
        """
        features = np.zeros(config.STATE_DIM)
        
        # 时间特征
        current_time = crew_state['last_task_end_time']
        features[0] = current_time.weekday()  # 星期几
        features[1] = current_time.hour  # 小时
        features[2] = current_time.day  # 日期
        
        # 位置特征（机场哈希）
        if crew_state['last_location']:
            features[3] = hash(crew_state['last_location']) % 1000
        
        # 值勤状态特征 - 从crew_state中获取
        if crew_state.get('duty_start_time'):
            duty_duration = (current_time - crew_state['duty_start_time']).total_seconds() / 3600
            features[4] = min(duty_duration, 24)  # 当前值勤时长（小时）
            features[5] = crew_state.get('duty_flight_time', 0)  # 值勤内飞行时间
            features[6] = crew_state.get('duty_flight_count', 0)  # 值勤内航班数
            features[7] = crew_state.get('duty_task_count', 0)  # 值勤内任务数
        
        # 累计资源特征
        features[8] = crew_state.get('total_flight_hours', 0)  # 总飞行时间
        features[9] = crew_state.get('total_positioning', 0)  # 总调机次数
        features[10] = crew_state.get('total_away_overnights', 0)  # 总外站过夜
        features[11] = len(crew_state.get('total_calendar_days', set()))  # 总日历天数
        
        # 成本特征
        features[12] = crew_state.get('cost', 0) / 1000.0  # 归一化成本
        
        # 机组基地特征
        features[13] = 1 if crew_state['last_location'] == crew_info['base'] else 0
        
        return features

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
        """
        将任务转换为特征向量，用于强化学习的动作表示。
        与attention_guided_subproblem_solver.py中的_extract_task_features保持一致
        """
        features = np.zeros(config.ACTION_DIM)
        
        # 连接时间
        connection_time = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
        features[0] = min(connection_time, 48)  # 限制在48小时内
        
        # 任务类型特征
        if task['type'] == 'flight':
            features[1] = 1
            features[2] = task.get('flyTime', 0) / 60.0  # 飞行时间（小时）
        elif 'positioning' in task.get('type', ''):
            features[3] = 1
            if 'bus' in task.get('type', ''):
                features[4] = 1  # 巴士调机
            else:
                features[5] = 1  # 飞行调机
        elif task['type'] == 'ground_duty':
            features[6] = 1  # 占位任务特征
        
        # 机场特征
        features[7] = hash(task['depaAirport']) % 1000
        features[8] = hash(task['arriAirport']) % 1000
        
        # 时间特征
        features[9] = task['startTime'].weekday()
        features[10] = task['startTime'].hour
        features[11] = task['endTime'].hour
        
        # 任务持续时间
        duration = (task['endTime'] - task['startTime']).total_seconds() / 3600
        features[12] = min(duration, 24)
        
        return features

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
            
            # 更新crew_state中的所有状态信息
            crew_state = self.crew_states[crew_id]
            
            # 计算连接时间（基于之前的任务结束时间）
            connection_time = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
            
            # 更新基本状态
            crew_state['last_task_end_time'] = task['endTime']
            crew_state['last_location'] = task['arriAirport']
            
            # 更新值勤状态
            if connection_time >= 12 or crew_state['duty_start_time'] is None:  # 新值勤开始
                crew_state['duty_start_time'] = task['startTime']
                crew_state['duty_flight_time'] = 0
                crew_state['duty_flight_count'] = 0
                crew_state['duty_task_count'] = 0
            
            crew_state['duty_task_count'] += 1
            if task_type == 'flight':
                crew_state['duty_flight_count'] += 1
                crew_state['duty_flight_time'] += task.get('flyTime', 0) / 60.0
                crew_state['total_flight_hours'] += task.get('flyTime', 0) / 60.0
            elif 'positioning' in task_type:
                crew_state['total_positioning'] += 1
            
            # 更新外站过夜
            if task['arriAirport'] != crew_info['base']:
                crew_state['total_away_overnights'] += 1
            
            # 更新日历天数
            crew_state['total_calendar_days'].add(task['startTime'].date())
            crew_state['total_calendar_days'].add(task['endTime'].date())
            
            # 更新成本（简化版）
            crew_state['cost'] += task.get('flyTime', 0) * 0.1  # 简化的成本计算
            
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