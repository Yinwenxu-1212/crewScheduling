# environment.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import collections

from . import config
from .utils import RuleChecker, calculate_final_score, DataHandler

class CrewRosteringEnv:
    """
    机组排班问题的强化学习环境。
    智能体（Agent）将轮流为每个机长选择下一个要执行的任务。
    遵循类似Gymnasium的接口设计。
    """
    def __init__(self, data_handler: DataHandler):
        self.dh = data_handler
        self.rule_checker = RuleChecker(self.dh)
        
        self.crews = self.dh.data['crews'].to_dict('records')
        self.unified_tasks_df = self.dh.data['unified_tasks']
        self.planning_start_dt = pd.to_datetime(config.PLANNING_START_DATE)
        self.planning_end_dt = pd.to_datetime(config.PLANNING_END_DATE)

    def reset(self):
        """
        开始一个新回合，重置所有状态，并返回第一个观测。
        返回:
        - observation (dict): 初始观测。
        - info (dict): 附加信息，这里是有效的动作列表。
        """
        # 每个机长的排班计划
        self.roster_plan = collections.defaultdict(list)
        # 记录每个机组的动态状态
        self.crew_states = {crew['crewId']: self._get_initial_crew_state(crew) for crew in self.crews}
        # 未分配的任务ID集合
        self.unassigned_task_ids = set(self.unified_tasks_df[self.unified_tasks_df['type'] == 'flight']['taskId'])
        
        # 将固有占位任务预先加入排班表
        for crew in self.crews:
            crew_id = crew['crewId']
            ground_duties = self.dh.crew_ground_duties.get(crew_id, [])
            if ground_duties:
                self.roster_plan[crew_id].extend(ground_duties)
                # 更新状态
                last_duty = max(ground_duties, key=lambda x: x['endTime'])
                self.crew_states[crew_id]['last_task_end_time'] = last_duty['endTime']
                self.crew_states[crew_id]['last_location'] = last_duty['airport']

        # 当前轮到哪个机长做决策
        self.current_crew_idx = 0
        self.total_steps = 0
        
        observation, valid_actions = self._get_observation()
        info = {'valid_actions': valid_actions}
        
        return observation, info

    def _get_initial_crew_state(self, crew):
        """获取机长的初始状态。"""
        return {
            'last_task_end_time': self.planning_start_dt,
            'last_location': crew['stayStation'],
            'total_flight_duty_time': 0,
            # 在此可以添加更多需要追踪的状态，如飞行周期相关状态
        }

    def _get_observation(self):
        """
        核心函数：为当前机长构建观测，包括状态和有效动作。
        """
        if self.current_crew_idx >= len(self.crews):
            # 这种情况理论上不应该发生，除非所有机组都已处理完毕
            return None, []

        crew_info = self.crews[self.current_crew_idx]
        crew_id = crew_info['crewId']
        crew_state = self.crew_states[crew_id]
        current_roster = sorted(self.roster_plan[crew_id], key=lambda x: x.get('startTime', self.planning_start_dt))

        # --- 1. 构建状态向量 (State Vector) ---
        state_vector = self._build_state_vector(crew_state)
        
        # --- 2. 筛选有效候选动作 (Action Masking) ---
        valid_actions = self._get_valid_actions(crew_info, crew_state, current_roster)
        
        # --- 3. 构建动作特征向量和掩码 ---
        action_features = np.zeros((config.MAX_CANDIDATE_ACTIONS, config.ACTION_DIM))
        action_mask = np.zeros(config.MAX_CANDIDATE_ACTIONS, dtype=np.uint8)

        for i, action in enumerate(valid_actions):
            action_mask[i] = 1
            action_features[i, :] = self._task_to_feature_vector(action, crew_state)
            
        observation = {
            'state': state_vector,
            'action_features': action_features,
            'action_mask': action_mask
        }
        return observation, valid_actions
        
    def _build_state_vector(self, crew_state):
        """特征工程: 构建状态向量"""
        state_vector = np.zeros(config.STATE_DIM)
        time_since_start = (crew_state['last_task_end_time'] - self.planning_start_dt).total_seconds() / (3600 * 24 * 7)
        time_of_day = crew_state['last_task_end_time'].hour / 24.0
        day_of_week = crew_state['last_task_end_time'].weekday() / 7.0
        
        state_vector[0] = time_since_start
        state_vector[1] = time_of_day
        state_vector[2] = day_of_week
        state_vector[3] = crew_state['total_flight_duty_time'] / 60.0
        # 可以在此添加更多状态特征，如机场的one-hot编码等
        return state_vector

    def _get_valid_actions(self, crew_info, crew_state, current_roster):
        """根据硬约束筛选有效动作。"""
        valid_actions = []
        crew_id = crew_info['crewId']
        
        # 预筛选
        potential_tasks_df = self.unified_tasks_df[
            (self.unified_tasks_df['taskId'].isin(self.unassigned_task_ids)) &
            (self.unified_tasks_df['startTime'] > crew_state['last_task_end_time'] + timedelta(hours=1)) &
            (self.unified_tasks_df['depaAirport'] == crew_state['last_location'])
        ].sort_values('startTime').head(config.MAX_CANDIDATE_ACTIONS * 2)

        for _, task_series in potential_tasks_df.iterrows():
            task_dict = task_series.to_dict()
            
            # 使用增量规则检查器（这里是简化的逻辑，需要你根据utils.py完善）
            # Rule 10: Qualification
            if task_dict['type'] == 'flight' and task_dict['taskId'] not in self.dh.crew_leg_map.get(crew_id, set()):
                continue
            
            # Rule 11: No Overlap with existing roster (including ground duties)
            is_overlap = False
            for existing_task in current_roster:
                if not (task_dict['endTime'] <= existing_task.get('startTime', self.planning_end_dt) or \
                        task_dict['startTime'] >= existing_task.get('endTime', self.planning_start_dt)):
                    is_overlap = True
                    break
            if is_overlap:
                continue
            
            # ... 在此添加更多增量检查 ...
            
            valid_actions.append(task_dict)
            if len(valid_actions) >= config.MAX_CANDIDATE_ACTIONS:
                break
        return valid_actions

    def _task_to_feature_vector(self, task, crew_state):
        """特征工程: 将任务转换为特征向量"""
        vec = np.zeros(config.ACTION_DIM)
        vec[0] = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600 # Connection time
        vec[1] = task.get('flyTime', 0) / 60.0 # Fly time
        vec[2] = 1 if task['type'] == 'flight' else 0
        vec[3] = 1 if 'positioning' in task.get('type', '') else 0
        # ... 可以在此添加更多动作特征
        return vec

    def step(self, action_idx, valid_actions):
        """
        执行一步动作，即为当前机长选择一个任务。
        返回:
        - observation (dict): 下一个观测。
        - reward (float): 该动作获得的奖励。
        - terminated (bool): 回合是否因为正常结束而终止。
        - truncated (bool): 回合是否因为达到步数上限而终止。
        - info (dict): 附加信息。
        """
        reward = 0
        crew_info = self.crews[self.current_crew_idx]
        crew_id = crew_info['crewId']

        # 如果没有有效动作，或者智能体选择“不安排”(action_idx 超出范围)
        if action_idx >= len(valid_actions):
            reward -= 0.1  # 小的负奖励，鼓励模型尽可能安排任务
        else:
            task = valid_actions[action_idx]
            
            # 更新排班和状态
            self.roster_plan[crew_id].append(task)
            if task['type'] == 'flight':
                 self.unassigned_task_ids.remove(task['taskId'])
            
            self.crew_states[crew_id]['last_task_end_time'] = task['endTime']
            self.crew_states[crew_id]['last_location'] = task['arriAirport']
            # TODO: 更新更复杂的累计状态，如值勤时间
            
            # 计算即时奖励
            reward += self._calculate_immediate_reward(task, crew_info)

        # 切换到下一个机长进行决策 (轮询)
        self.current_crew_idx = (self.current_crew_idx + 1) % len(self.crews)
        self.total_steps += 1
        
        # 检查回合是否结束
        terminated = not self.unassigned_task_ids
        truncated = self.total_steps >= (len(self.crews) * config.MAX_STEPS_PER_CREW)
        done = terminated or truncated
        
        if done:
            final_score = calculate_final_score(self.roster_plan, self.dh)
            reward += final_score
            observation, next_valid_actions = None, []
        else:
            observation, next_valid_actions = self._get_observation()
            
        info = {'valid_actions': next_valid_actions, 'is_success': terminated}
        
        return observation, reward, terminated, truncated, info

    def _calculate_immediate_reward(self, task, crew_info):
        """奖励塑造: 设计即时奖励来引导模型学习"""
        reward = 0
        if task.get('type') == 'flight':
            reward += (task.get('flyTime', 0) / 60.0) * 0.5  # 飞行时间奖励
        elif 'positioning' in task.get('type'):
            reward += config.PENALTY_POSITIONING * 0.5  # 置位惩罚
            
        # 惩罚外站过夜（简化判断）
        if task['arriAirport'] != crew_info['base']:
            if task['endTime'].hour >= 20:
                reward += config.PENALTY_OVERNIGHT_STAY * 0.1
        
        return reward