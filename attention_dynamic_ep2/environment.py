# environment.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import collections
import random
import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from . import config
    from .utils import RuleChecker, DataHandler, identify_duties_and_cycles, calculate_final_score
except ImportError:
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
        # 正确初始化未分配占位任务的跟踪
        self.unassigned_ground_duty_ids = set(self.dh.data['ground_duties']['id'])
        
        # 占位任务应该通过动作选择机制进行分配，而不是预分配
        # 这样可以让模型学习如何平衡航班任务和占位任务
        self.total_steps = 0
        self.decision_priority_queue = self._build_priority_queue()
        observation, info = self._get_observation()
        return observation, info

    def _get_initial_crew_state(self, crew):
        # 机组初始位置应该始终是stayStation，不受占位任务影响
        # 占位任务是预分配的，不影响机组的实际位置
        return {
            'last_task_end_time': self.planning_start_dt,
            'last_location': crew['stayStation'],  # 机组初始位置始终是stayStation
            'duty_start_time': None,  # 当前值勤开始时间
            'duty_flight_time': 0,    # 值勤内飞行时间
            'duty_flight_count': 0,   # 值勤内航班数
            'duty_task_count': 0,     # 值勤内任务数
            'total_flight_hours': 0,  # 总飞行时间
            'total_positioning': 0,   # 总调机次数
            'duty_positioning': 0,    # 当前值勤内置位次数
            'total_away_overnights': 0,  # 总外站过夜
            'total_calendar_days': set(),  # 总日历天数
            'cost': 0                 # 累计成本
        }

    def _build_priority_queue(self):
        priorities = []
        unassigned_departures = self.dh.data['flights'][self.dh.data['flights']['id'].isin(self.unassigned_flight_ids)].groupby('depaAirport')['id'].count()
        for i, crew in enumerate(self.crews):
            crew_state = self.crew_states[crew['crewId']]
            # 确保score始终是数值类型，避免字典和整数比较错误
            departure_count = unassigned_departures.get(crew_state['last_location'], 0)
            if hasattr(departure_count, 'iloc'):  # 如果是pandas Series
                departure_count = departure_count.iloc[0] if len(departure_count) > 0 else 0
            score = float(departure_count)  # 确保是数值类型
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
        构建增强的状态向量，包含全局信息和更丰富的特征
        """
        features = np.zeros(config.STATE_DIM)
        
        # 时间特征（归一化）
        current_time = crew_state['last_task_end_time']
        features[0] = current_time.weekday() / 6.0  # 星期几（归一化）
        features[1] = current_time.hour / 23.0  # 小时（归一化）
        features[2] = (current_time - self.planning_start_dt).total_seconds() / (7 * 24 * 3600)  # 规划进度
        
        # 位置特征（改进的编码方式）
        if crew_state['last_location']:
            # 使用更稳定的机场编码
            airport_code = crew_state['last_location']
            features[3] = len(airport_code) / 10.0  # 机场代码长度特征
            features[4] = 1 if airport_code == crew_info['base'] else 0  # 是否在基地
        
        # 值勤状态特征（归一化）
        if crew_state.get('duty_start_time'):
            duty_duration = (current_time - crew_state['duty_start_time']).total_seconds() / 3600
            features[5] = min(duty_duration / 12.0, 1.0)  # 当前值勤时长比例
            features[6] = min(crew_state.get('duty_flight_time', 0) / 8.0, 1.0)  # 值勤内飞行时间比例
            features[7] = min(crew_state.get('duty_flight_count', 0) / 4.0, 1.0)  # 值勤内航班数比例
            features[8] = min(crew_state.get('duty_task_count', 0) / 6.0, 1.0)  # 值勤内任务数比例
        
        # 累计资源特征（归一化）
        features[9] = min(crew_state.get('total_flight_hours', 0) / 60.0, 1.0)  # 总飞行时间比例
        features[10] = min(crew_state.get('total_positioning', 0) / 10.0, 1.0)  # 总调机次数比例
        features[11] = min(crew_state.get('total_away_overnights', 0) / 7.0, 1.0)  # 总外站过夜比例
        features[12] = min(len(crew_state.get('total_calendar_days', set())) / 7.0, 1.0)  # 总日历天数比例
        
        # 全局状态特征
        features[13] = self._get_global_coverage_rate()  # 全局航班覆盖率
        features[14] = self._get_crew_workload_balance()  # 机组工作负载平衡度
        features[15] = self._get_time_pressure()  # 时间压力指标
        
        # 约束风险特征
        features[16] = self._get_constraint_risk_score(crew_state)  # 约束违规风险评分
        
        # 添加辅助方法的实现
        return features
    
    def _calculate_crew_capability_bonus(self, crew_info):
        """计算机组能力匹配奖励"""
        # 基于机组的基地位置、资质等计算能力匹配度
        base_airport = crew_info.get('base', '')
        stay_station = crew_info.get('stayStation', '')
        
        # 如果机组在主要基地，给予奖励
        if base_airport in ['PEK', 'SHA', 'CAN', 'SZX']:  # 主要枢纽
            return 0.3
        elif base_airport == stay_station:
            return 0.2
        else:
            return 0.1
    
    def _calculate_time_window_pressure(self, task):
        """计算时间窗口压力"""
        # 检查任务时间窗口的紧迫性
        time_to_task = (task['startTime'] - self.planning_start_dt).total_seconds() / 3600
        
        if time_to_task < 12:
            return 0.8  # 高压力
        elif time_to_task < 24:
            return 0.6  # 中高压力
        elif time_to_task < 48:
            return 0.4  # 中等压力
        else:
            return 0.2  # 低压力
    
    def _calculate_airport_coverage_pressure(self, task):
        """计算机场覆盖压力"""
        depa_airport = task['depaAirport']
        
        # 计算该机场未分配航班的比例
        airport_flights = self.dh.data['flights'][self.dh.data['flights']['depaAirport'] == depa_airport]
        unassigned_airport_flights = [f for f in airport_flights['id'] if f in self.unassigned_flight_ids]
        
        if len(airport_flights) == 0:
            return 0.0
        
        unassigned_ratio = len(unassigned_airport_flights) / len(airport_flights)
        return unassigned_ratio
    
    def _calculate_ground_duty_pressure(self, task):
        """计算占位任务的覆盖压力"""
        # 基于时间段的重要性
        start_hour = task['startTime'].hour
        is_weekday = task['startTime'].weekday() < 5
        
        if is_weekday and 8 <= start_hour <= 18:
            return 0.7  # 工作日白天，高压力
        elif is_weekday and (6 <= start_hour < 8 or 18 < start_hour <= 22):
            return 0.5  # 工作日早晚，中等压力
        else:
            return 0.3  # 其他时间，低压力
    
    def _calculate_positioning_pressure(self, task):
        """计算置位任务的压力"""
        # 基于置位的战略价值
        arrival_airport = task['arriAirport']
        
        # 检查目标机场的未分配航班数量
        future_flights = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task['endTime']) &
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids))
        ]
        
        if len(future_flights) > 0:
            return min(len(future_flights) * 0.2, 0.8)  # 有后续航班，压力较高
        else:
            return 0.1  # 无后续航班，压力很低
    
    def _calculate_consecutive_days_risk(self, task, crew_state):
        """计算连续工作天数风险"""
        # 简化实现：基于已工作的日历天数
        total_days = len(crew_state.get('total_calendar_days', set()))
        
        if total_days >= 6:
            return 0.3  # 高风险
        elif total_days >= 4:
            return 0.2  # 中等风险
        elif total_days >= 2:
            return 0.1  # 低风险
        else:
            return 0.0  # 无风险
    
    def _calculate_overnight_risk(self, task, crew_state):
        """计算外站过夜风险"""
        total_overnights = crew_state.get('total_away_overnights', 0)
        
        if total_overnights >= 5:
            return 0.2  # 高风险
        elif total_overnights >= 3:
            return 0.1  # 中等风险
        else:
            return 0.0  # 低风险
    
    def _get_global_coverage_rate(self):
        """计算全局航班覆盖率"""
        total_flights = len(self.dh.data['flights'])
        covered_flights = total_flights - len(self.unassigned_flight_ids)
        return covered_flights / total_flights if total_flights > 0 else 0
    
    def _get_crew_workload_balance(self):
        """计算机组工作负载平衡度"""
        workloads = []
        for crew in self.crews:
            crew_tasks = len(self.roster_plan.get(crew['crewId'], []))
            workloads.append(crew_tasks)
        
        if len(workloads) > 1:
            mean_workload = np.mean(workloads)
            std_workload = np.std(workloads)
            return 1.0 - min(std_workload / (mean_workload + 1e-6), 1.0)
        return 1.0
    
    def _get_time_pressure(self):
        """计算时间压力指标"""
        if not self.unassigned_flight_ids:
            return 0.0
        
        # 计算未分配航班的平均剩余时间
        current_time = pd.Timestamp.now()
        urgent_flights = 0
        
        for flight_id in self.unassigned_flight_ids:
            flight_row = self.dh.data['flights'][self.dh.data['flights']['id'] == flight_id]
            if not flight_row.empty:
                flight_time = pd.to_datetime(flight_row.iloc[0]['std'])
                time_to_flight = (flight_time - current_time).total_seconds() / 3600
                if time_to_flight < 48:  # 48小时内的航班算紧急
                    urgent_flights += 1
        
        return urgent_flights / len(self.unassigned_flight_ids) if self.unassigned_flight_ids else 0
    
    def _get_constraint_risk_score(self, crew_state):
        """计算约束违规风险评分"""
        risk_score = 0.0
        
        # 飞行时间风险
        flight_time_ratio = crew_state.get('duty_flight_time', 0) / 8.0
        if flight_time_ratio > 0.8:
            risk_score += 0.3
        
        # 值勤时长风险
        if crew_state.get('duty_start_time'):
            current_time = crew_state['last_task_end_time']
            duty_duration = (current_time - crew_state['duty_start_time']).total_seconds() / 3600
            duty_ratio = duty_duration / 12.0
            if duty_ratio > 0.8:
                risk_score += 0.3
        
        # 总飞行时间风险
        total_flight_ratio = crew_state.get('total_flight_hours', 0) / 60.0
        if total_flight_ratio > 0.8:
            risk_score += 0.4
        
        return min(risk_score, 1.0)

    def _get_valid_actions(self, crew_info, crew_state):
        last_loc, last_time = crew_state['last_location'], crew_state['last_task_end_time']
        potential_tasks_df = self.unified_tasks_df[(self.unified_tasks_df['startTime'] > last_time) & (self.unified_tasks_df['depaAirport'] == last_loc)].copy()
        all_actions, current_roster = [], self.roster_plan[crew_info['crewId']]
        
        # 分类收集任务：优先级为 占位任务 > 飞行任务 > 置位任务
        ground_duties = []
        flight_tasks = []
        positioning_tasks = []
        
        for _, task_series in potential_tasks_df.iterrows():
            task_id = task_series['taskId']
            if task_series['type'] == 'flight':
                # 正常飞行任务（如果机组有资格）
                if task_id in self.unassigned_flight_ids and task_id in self.dh.crew_leg_map.get(crew_info['crewId'], set()):
                    flight_tasks.append({**task_series.to_dict(), 'type': 'flight'})
                # 置位任务：所有已被其他机组执飞的航班都可用于置位（不需要检查机组资格）
                elif task_id not in self.unassigned_flight_ids and self._should_consider_positioning(task_series, crew_info, crew_state):
                    positioning_tasks.append({**task_series.to_dict(), 'type': 'positioning_flight'})
            elif task_series['type'] == 'ground_duty':
                # 包含所有未分配的占位任务
                if task_id in self.unassigned_ground_duty_ids:
                    ground_duties.append(task_series.to_dict())
            elif task_series['type'] == 'positioning_bus':
                # 大巴置位任务：与飞行置位任务同等处理
                if self._should_consider_positioning(task_series, crew_info, crew_state):
                    positioning_tasks.append({**task_series.to_dict(), 'type': 'positioning_bus'})
            else:
                # 其他类型任务（如果有的话）
                all_actions.append(task_series.to_dict())
        
        # 按优先级排序：占位任务优先
        # 1. 占位任务逻辑：严格按照crewId分配，不允许跨机组执行
        crew_specific_ground_duties = [gd for gd in ground_duties if gd.get('crewId') == crew_info['crewId']]
        # 占位任务必须严格分配给指定机组，不允许其他机组执行
        other_ground_duties = []  # 移除跨机组占位任务分配
        
        # 改进的置位任务筛选逻辑
        valuable_positioning_tasks = self._filter_valuable_positioning_tasks(positioning_tasks, crew_info, crew_state)
        
        # 智能任务选择：基于时间敏感性和任务特性的动态优先级
        all_candidate_tasks = []
        all_candidate_tasks.extend(flight_tasks)
        all_candidate_tasks.extend(crew_specific_ground_duties)
        # 占位任务严格按照crewId分配，不添加其他机组的占位任务
        
        # 更积极地考虑置位任务，特别是在覆盖率低或有战略价值时
        coverage_rate = self._calculate_current_coverage_rate()
        if (not flight_tasks or  # 没有直接可执行的航班时
            len(valuable_positioning_tasks) > 0 or  # 有价值的置位任务
            coverage_rate < 0.7):  # 覆盖率较低时也考虑置位
            all_candidate_tasks.extend(valuable_positioning_tasks)
        
        valid_actions, relaxed_actions = [], []
        for action in all_candidate_tasks:
            # 预筛选：检查置位任务限制
            if self._should_filter_positioning_action(action, crew_state):
                continue  # 跳过会导致超出置位限制的动作
                
            is_valid, violated_rules = self._is_incrementally_valid(current_roster, action, crew_info, crew_state)
            if is_valid: 
                # 计算动态优先级分数
                action['dynamic_priority'] = self._calculate_dynamic_priority(action, crew_state, crew_info)
                valid_actions.append(action)
            elif any('minor' in r for r in violated_rules):
                action['violation_penalty'] = len(violated_rules)
                action['dynamic_priority'] = self._calculate_dynamic_priority(action, crew_state, crew_info)
                relaxed_actions.append(action)
        
        # 基于动态优先级排序（分数越高优先级越高）
        valid_actions.sort(key=lambda x: -x['dynamic_priority'])
        relaxed_actions.sort(key=lambda x: (-x.get('violation_penalty', 0), -x['dynamic_priority']))
        

        
        if valid_actions: return valid_actions[:config.MAX_CANDIDATE_ACTIONS], []
        return [], relaxed_actions[:config.MAX_CANDIDATE_ACTIONS]
    
    def _calculate_dynamic_priority(self, task, crew_state, crew_info):
        """计算任务的动态优先级分数，综合考虑时间敏感性、任务特性和机组匹配度"""
        from config import PRIORITY_WEIGHTS
        priority_score = 0.0
        
        # 1. 基础任务类型权重（使用配置化权重）
        if task['type'] == 'flight':
            priority_score += PRIORITY_WEIGHTS['flight_base']
        elif task['type'] == 'ground_duty':
            # 改进的占位任务优先级计算
            base_ground_score = PRIORITY_WEIGHTS['ground_duty_base']
            
            # 专属占位任务获得最高优先级（硬约束）
            if task.get('crewId') == crew_info['crewId']:
                base_ground_score += PRIORITY_WEIGHTS['ground_duty_exclusive']
                
                # 任务完成风险评估
                completion_risk = self._evaluate_task_completion_risk(task, crew_state)
                if completion_risk > 0.7:  # 高风险任务
                    base_ground_score += PRIORITY_WEIGHTS['ground_duty_risk_bonus']
            else:
                # 非专属占位任务不应该出现在候选列表中，如果出现则给予极低优先级
                base_ground_score -= 1000.0  # 极低优先级，基本不会被选择
            
            # 根据机组工作负载平衡调整
            current_flight_count = sum(1 for t in self.roster_plan.get(crew_info['crewId'], []) if t.get('type') == 'flight')
            if current_flight_count >= 3:
                base_ground_score += PRIORITY_WEIGHTS['ground_duty_workload_balance']
            elif current_flight_count >= 2:
                base_ground_score += 10.0
            
            priority_score += base_ground_score
        else:  # positioning tasks
            # 置位任务基础分数：与航班任务平衡竞争
            base_positioning_score = PRIORITY_WEIGHTS['positioning_base']
            
            # 大巴置位比飞行置位稍微优先一些
            if task.get('type') == 'positioning_bus':
                base_positioning_score += 10.0  # 大巴置位额外奖励（稍高于飞行置位）
            elif task.get('type') == 'positioning_flight':
                base_positioning_score += 8.0  # 飞行置位额外奖励
            
            # 评估置位的战略价值（主要决定因素）
            strategic_value = self._calculate_positioning_strategic_value(task, crew_info)
            
            # 根据战略价值调整优先级（降低门槛）
            if strategic_value >= 15.0:  # 高战略价值置位（从20.0降低到15.0）
                base_positioning_score += PRIORITY_WEIGHTS['positioning_high_value']
            elif strategic_value >= 7.0:  # 中等战略价值置位（从10.0降低到7.0）
                base_positioning_score += PRIORITY_WEIGHTS['positioning_medium_value']
            elif strategic_value >= 3.0:  # 低战略价值置位（从5.0降低到3.0）
                base_positioning_score += PRIORITY_WEIGHTS['positioning_low_value']
            else:  # 无明显战略价值
                base_positioning_score += 5.0  # 给予基础奖励（从0.0提高到5.0）
            
            # 根据当前覆盖率微调（次要因素）
            coverage_rate = self._calculate_current_coverage_rate()
            if coverage_rate < 0.5:  # 覆盖率极低时
                base_positioning_score += 20.0  # 增加奖励（从15.0提高到20.0）
            elif coverage_rate < 0.7:  # 覆盖率低时（从0.6提高到0.7）
                base_positioning_score += 15.0  # 增加奖励（从10.0提高到15.0）
            elif coverage_rate > 0.9:  # 覆盖率很高时才降低置位优先级（从0.85提高到0.9）
                base_positioning_score -= 10.0  # 减少惩罚（从-15.0提高到-10.0）
            
            priority_score += base_positioning_score
        
        # 2. 时间紧迫性加权（占位任务在时间窗口紧迫时优先级提升）
        time_urgency = self._calculate_time_urgency(task)
        if task['type'] == 'ground_duty':
            # 占位任务的时间窗口越紧迫，优先级越高
            time_window_hours = (task['endTime'] - task['startTime']).total_seconds() / 3600
            if time_window_hours <= 12:  # 短时间窗口的占位任务
                priority_score += time_urgency * 50.0  # 大幅提升优先级
            else:
                priority_score += time_urgency * 20.0
        else:
            priority_score += time_urgency * PRIORITY_WEIGHTS['time_urgency_multiplier']
        
        # 3. 机组匹配度加权
        if task['type'] == 'ground_duty' and task.get('crewId') == crew_info['crewId']:
            priority_score += 60.0  # 专属占位任务加分（从40.0提升至60.0，确保硬约束放松后的优先级）
        elif task['type'] == 'flight':
            # 航班任务的稀缺性加分
            scarcity = self._calculate_task_scarcity(task, crew_info)
            priority_score += scarcity * PRIORITY_WEIGHTS['scarcity_multiplier']
        
        # 4. 连接性和效率加权
        connection_time = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
        if 1 <= connection_time <= 6:  # 理想连接时间
            priority_score += PRIORITY_WEIGHTS['connection_bonus']
        elif connection_time > 24:  # 长时间等待的惩罚
            priority_score -= 10.0
        
        # 5. 当前时间点的战略考虑
        # 如果当前机组已经有较多航班任务，适当提升占位任务优先级以平衡工作负载
        current_flight_count = sum(1 for t in self.roster_plan.get(crew_info['crewId'], []) if t.get('type') == 'flight')
        if task['type'] == 'ground_duty' and current_flight_count >= 3:
            priority_score += 30.0  # 工作负载平衡加分（从20.0提升至30.0）
        
        return priority_score

    def _is_incrementally_valid(self, roster, new_task, crew_info, crew_state):
        violated_rules = set()
        if new_task.get('type') == 'flight' and new_task['taskId'] not in self.dh.crew_leg_map.get(crew_info['crewId'], set()): return False, {'hard_violation_qualification'}
        for task in roster:
            if not (new_task['endTime'] <= task.get('startTime', config.PLANNING_END_DATE) or new_task['startTime'] >= task.get('endTime', config.PLANNING_START_DATE)): return False, {'hard_violation_overlap'}
        
        # 检查连接时间约束
        temp_roster = sorted(roster + [new_task], key=lambda x: x['startTime'])
        for i in range(len(temp_roster) - 1):
            curr_task = temp_roster[i]
            next_task = temp_roster[i + 1]
            connection_time = (next_task['startTime'] - curr_task['endTime']).total_seconds() / 3600
            
            # 获取最小连接时间要求
            min_connection_hours = self._get_min_connection_time_hours(curr_task, next_task)
            if connection_time < min_connection_hours:
                return False, {'hard_violation_connection_time'}
        
        ground_duties = self.dh.crew_ground_duties.get(crew_info['crewId'], [])
        temp_duties, _ = identify_duties_and_cycles(temp_roster, ground_duties)
        if temp_duties:
            last_duty = temp_duties[-1]
            flight_duty_tasks = [t for t in last_duty if t.get('type') in ['flight', 'positioning_flight', 'positioning_bus']]
            if flight_duty_tasks:
                # 计算飞行时间：只包括实际飞行任务的飞行时间，不包括飞行置位
                total_flight_time = sum(t.get('flyTime', 0) for t in flight_duty_tasks if t.get('type') == 'flight' and not t.get('is_positioning', False)) / 60.0
                if total_flight_time > 8: violated_rules.add('minor_violation_fly_time')
                duty_start = flight_duty_tasks[0]['startTime']
                flight_tasks_in_duty = [t for t in flight_duty_tasks if 'flight' in t.get('type','')]
                if flight_tasks_in_duty:
                    duty_end = flight_tasks_in_duty[-1]['endTime']
                    if (duty_end - duty_start) > timedelta(hours=12): violated_rules.add('minor_violation_duty_duration')
        return len(violated_rules) == 0, violated_rules
    
    def _should_filter_positioning_action(self, action, crew_state):
        """检查是否应该过滤掉置位动作以避免超出限制"""
        if 'positioning' not in action.get('type', ''):
            return False
        
        # 占位任务（ground_duty）不过滤，优先分配
        if action.get('type') == 'ground_duty':
            return False
            
        # 对置位任务进行严格的规则检查
        # 1. 检查是否超出置位次数限制
        if crew_state.get('total_positioning', 0) >= config.MAX_POSITIONING_TOTAL:
            return True  # 过滤掉
            
        # 2. 检查是否在值勤日中间进行置位（违反规则）
        if self._is_middle_positioning(action, crew_state):
            return True  # 过滤掉
            
        # 3. 检查值勤内置位次数限制
        if crew_state.get('duty_positioning', 0) >= config.MAX_POSITIONING_PER_DUTY:
            return True  # 过滤掉
            
        return False  # 不过滤
    
    def _get_min_connection_time_hours(self, prev_task, next_task):
        """获取两个任务之间的最小连接时间（小时）"""
        # 飞行值勤日内相邻任务的最小连接时间限制
        if prev_task.get('type') in ['flight', 'positioning_flight'] and next_task.get('type') in ['flight', 'positioning_flight']:
            # 检查是否为同一架飞机
            prev_aircraft = prev_task.get('aircraftNo')
            next_aircraft = next_task.get('aircraftNo')
            if prev_aircraft and next_aircraft and prev_aircraft == next_aircraft:
                return 0.5  # 同飞机30分钟
            else:
                return 3.0  # 不同飞机3小时
        elif prev_task.get('type') == 'positioning_bus' or next_task.get('type') == 'positioning_bus':
            # 大巴置位与相邻飞行任务及飞行置位任务之间最小间隔时间为2小时
            return 2.0
        else:
            return 0.5  # 默认最小连接时间30分钟
    
    def _calculate_task_scarcity(self, task, crew_info):
        """计算任务稀缺性评分（模拟对偶价格中的稀缺性）"""
        if task['type'] == 'flight':
            # 计算能执行此航班的机组数量
            eligible_crews = 0
            total_workload = 0
            
            for crew in self.crews:
                if task['taskId'] in self.dh.crew_leg_map.get(crew['crewId'], set()):
                    eligible_crews += 1
                    # 考虑机组当前工作负载
                    current_load = len(self.roster_plan.get(crew['crewId'], []))
                    total_workload += current_load
            
            if eligible_crews == 0:
                return 1.0  # 最高稀缺性
            
            # 综合考虑可用机组数量和平均负载
            avg_load = total_workload / eligible_crews if eligible_crews > 0 else 0
            load_factor = 1.0 + avg_load * 0.1  # 负载越高，稀缺性越高
            
            # 稀缺性与可用机组数量成反比，与负载成正比
            scarcity = load_factor / max(eligible_crews, 1)
            return min(scarcity * 5.0, 1.0)  # 归一化到[0,1]
        
        elif task['type'] == 'ground_duty':
            # 占位任务的稀缺性基于时间窗口和竞争程度
            time_window_hours = (task['endTime'] - task['startTime']).total_seconds() / 3600
            # 时间窗口越短，稀缺性越高
            time_scarcity = max(0.2, 1.0 - time_window_hours / 24.0)
            return min(time_scarcity, 0.8)  # 中等到高稀缺性
        
        else:  # positioning tasks
            # 置位任务的稀缺性基于目标机场的后续任务密度
            arrival_airport = task['arriAirport']
            future_tasks_count = len(self.unified_tasks_df[
                (self.unified_tasks_df['depaAirport'] == arrival_airport) &
                (self.unified_tasks_df['startTime'] > task['endTime']) &
                (self.unified_tasks_df['startTime'] <= task['endTime'] + pd.Timedelta(hours=24))
            ])
            # 后续任务越多，置位的战略价值越高
            strategic_scarcity = min(future_tasks_count * 0.1, 0.5)
            return strategic_scarcity
    
    def _calculate_time_urgency(self, task):
        """计算时间紧迫性评分"""
        # 使用规划开始时间作为参考点，而不是当前系统时间
        reference_time = self.planning_start_dt + pd.Timedelta(hours=self.total_steps * 0.1)  # 模拟时间推进
        time_to_task = (task['startTime'] - reference_time).total_seconds() / 3600
        
        urgency = 0.0
        if time_to_task < 0:
            urgency = 1.0  # 已过期，最高紧迫性
        elif time_to_task < 6:
            urgency = 0.95  # 6小时内，极紧急
        elif time_to_task < 12:
            urgency = 0.8  # 12小时内，很紧急
        elif time_to_task < 24:
            urgency = 0.6  # 24小时内，紧急
        elif time_to_task < 48:
            urgency = 0.4  # 48小时内，中等
        elif time_to_task < 72:
            urgency = 0.2  # 72小时内，较低
        else:
            urgency = 0.1  # 72小时以上，很低
        
        # 对于航班任务，额外考虑连接紧迫性
        if task['type'] == 'flight':
            # 检查是否有紧急的后续连接
            connecting_flights = self.unified_tasks_df[
                (self.unified_tasks_df['depaAirport'] == task['arriAirport']) &
                (self.unified_tasks_df['startTime'] > task['endTime']) &
                (self.unified_tasks_df['startTime'] <= task['endTime'] + pd.Timedelta(hours=4))
            ]
            if len(connecting_flights) > 0:
                urgency += 0.1  # 连接紧迫性加成
        
        return min(urgency, 1.0)
    
    def _calculate_immediate_value(self, task, crew_state, crew_info):
        """计算执行任务的立即价值（模拟reduced cost贡献）"""
        value = 0.0
        
        if task['type'] == 'flight':
            # 航班覆盖的基础价值
            value += 50.0
            
            # 飞行时间价值
            fly_time_hours = task.get('flyTime', 0) / 60.0
            value += fly_time_hours * 10.0
            
            # 连接效率奖励
            connection_time = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
            if 2 <= connection_time <= 6:  # 理想连接时间
                value += 20.0
            
        elif task['type'] == 'ground_duty':
            # 占位任务的价值
            value += 15.0
            
        elif 'positioning' in task.get('type', ''):
            # 置位任务的负价值
            value -= 30.0
            
            # 但如果能连接到高价值航班，则有战略价值
            strategic_value = self._calculate_positioning_strategic_value(task, crew_info)
            value += strategic_value
        
        return np.tanh(value / 100.0)  # 归一化到[-1,1]
    
    def _calculate_positioning_strategic_value(self, task, crew_info):
        """计算置位任务的战略价值"""
        arrival_airport = task['arriAirport']
        
        # 检查置位后能执行的高价值航班
        future_flights = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task['endTime']) &
            (self.unified_tasks_df['startTime'] <= task['endTime'] + pd.Timedelta(hours=24)) &
            (self.unified_tasks_df['type'] == 'flight')
        ]
        
        if len(future_flights) > 0:
            # 有可连接的航班，战略价值较高
            return min(len(future_flights) * 15.0, 50.0)
        
        return 0.0
    
    def _calculate_crew_load_balance(self, crew_info):
        """计算机组负载平衡评分（模拟机组对偶价格）"""
        crew_id = crew_info['crewId']
        current_workload = len(self.roster_plan.get(crew_id, []))
        
        # 计算加权工作负载（考虑任务类型和飞行时间）
        weighted_workload = 0
        for task in self.roster_plan.get(crew_id, []):
            if task.get('type') == 'flight':
                weighted_workload += 1.0 + task.get('flyTime', 0) / 300.0  # 飞行任务权重更高
            elif 'positioning' in task.get('type', ''):
                weighted_workload += 0.8  # 置位任务中等权重
            else:
                weighted_workload += 0.5  # 其他任务较低权重
        
        # 计算所有机组的加权平均工作负载
        all_weighted_workloads = []
        for crew in self.crews:
            crew_weighted_load = 0
            for task in self.roster_plan.get(crew['crewId'], []):
                if task.get('type') == 'flight':
                    crew_weighted_load += 1.0 + task.get('flyTime', 0) / 300.0
                elif 'positioning' in task.get('type', ''):
                    crew_weighted_load += 0.8
                else:
                    crew_weighted_load += 0.5
            all_weighted_workloads.append(crew_weighted_load)
        
        avg_weighted_workload = np.mean(all_weighted_workloads) if all_weighted_workloads else 0
        
        # 负载差异（负值表示低于平均，正值表示高于平均）
        load_diff = weighted_workload - avg_weighted_workload
        
        # 转换为平衡评分：负载越低，评分越高（更适合分配任务）
        balance_score = np.tanh(-load_diff / 3.0)  # 归一化到[-1,1]
        
        # 额外考虑机组能力匹配度
        capability_bonus = self._calculate_crew_capability_bonus(crew_info)
        
        return balance_score + capability_bonus * 0.2
    
    def _calculate_coverage_pressure(self, task):
        """计算全局覆盖压力"""
        if task['type'] == 'flight':
            # 基于未分配航班数量的压力
            total_flights = len(self.dh.data['flights'])
            unassigned_count = len(self.unassigned_flight_ids)
            coverage_rate = 1.0 - (unassigned_count / total_flights) if total_flights > 0 else 1.0
            
            # 覆盖率越低，压力越大
            base_pressure = 1.0 - coverage_rate
            
            # 额外考虑时间窗口压力
            time_pressure = self._calculate_time_window_pressure(task)
            
            # 考虑机场覆盖压力
            airport_pressure = self._calculate_airport_coverage_pressure(task)
            
            # 综合压力评分
            total_pressure = base_pressure * 0.5 + time_pressure * 0.3 + airport_pressure * 0.2
            return min(total_pressure, 1.0)
        
        elif task['type'] == 'ground_duty':
            # 占位任务的覆盖压力基于时间段重要性
            return self._calculate_ground_duty_pressure(task)
        
        else:  # positioning tasks
            # 置位任务的压力基于战略价值
            return self._calculate_positioning_pressure(task)
    
    def _calculate_connection_efficiency(self, task, crew_state):
        """计算连接效率评分（后续任务连接潜力）"""
        arrival_airport = task['arriAirport']
        task_end_time = task['endTime']
        
        # 查找从到达机场出发的后续任务
        future_tasks = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task_end_time) &
            (self.unified_tasks_df['startTime'] <= task_end_time + pd.Timedelta(hours=48))
        ]
        
        if len(future_tasks) == 0:
            return 0.1  # 没有后续连接，效率很低
        
        # 计算连接效率：后续任务数量和时间窗口
        efficiency = 0.0
        for _, future_task in future_tasks.iterrows():
            connection_time = (future_task['startTime'] - task_end_time).total_seconds() / 3600
            if 2 <= connection_time <= 12:  # 理想连接时间窗口
                efficiency += 0.3
            elif connection_time <= 24:  # 可接受的连接时间
                efficiency += 0.1
        
        return min(efficiency, 1.0)
    
    def _estimate_constraint_risk(self, task, crew_state):
        """估计执行任务后的约束违规风险"""
        risk = 0.0
        
        # 飞行时间约束风险（更精细的评估）
        if task['type'] == 'flight':
            projected_duty_flight_time = crew_state.get('duty_flight_time', 0) + task.get('flyTime', 0) / 60.0
            flight_time_ratio = projected_duty_flight_time / 8.0
            
            if flight_time_ratio > 1.0:  # 超出限制
                risk += 0.8
            elif flight_time_ratio > 0.9:  # 接近限制
                risk += 0.5
            elif flight_time_ratio > 0.8:  # 较高风险
                risk += 0.3
            elif flight_time_ratio > 0.7:  # 中等风险
                risk += 0.1
        
        # 值勤时长约束风险（更精细的评估）
        if crew_state.get('duty_start_time'):
            projected_duty_duration = (task['endTime'] - crew_state['duty_start_time']).total_seconds() / 3600
            duty_ratio = projected_duty_duration / 12.0
            
            if duty_ratio > 1.0:  # 超出限制
                risk += 0.8
            elif duty_ratio > 0.9:  # 接近限制
                risk += 0.5
            elif duty_ratio > 0.8:  # 较高风险
                risk += 0.3
            elif duty_ratio > 0.7:  # 中等风险
                risk += 0.1
        
        # 置位次数约束风险
        if 'positioning' in task.get('type', ''):
            current_duty_positioning = crew_state.get('duty_positioning', 0)
            total_positioning = crew_state.get('total_positioning', 0)
            
            # 值勤内置位风险
            duty_positioning_ratio = (current_duty_positioning + 1) / config.MAX_POSITIONING_PER_DUTY
            if duty_positioning_ratio > 1.0:
                risk += 0.6
            elif duty_positioning_ratio > 0.8:
                risk += 0.4
            elif duty_positioning_ratio > 0.6:
                risk += 0.2
            
            # 总置位风险
            total_positioning_ratio = (total_positioning + 1) / config.MAX_POSITIONING_TOTAL
            if total_positioning_ratio > 1.0:
                risk += 0.6
            elif total_positioning_ratio > 0.8:
                risk += 0.4
            elif total_positioning_ratio > 0.6:
                risk += 0.2
        
        # 连续工作天数风险
        consecutive_days_risk = self._calculate_consecutive_days_risk(task, crew_state)
        risk += consecutive_days_risk
        
        # 外站过夜风险
        overnight_risk = self._calculate_overnight_risk(task, crew_state)
        risk += overnight_risk
        
        return min(risk, 1.0)  # 非置位任务不过滤
        
        current_duty_positioning = crew_state.get('duty_positioning', 0)
        total_positioning = crew_state.get('total_positioning', 0)
        
        # 如果值勤内已达到置位限制，过滤掉
        if current_duty_positioning >= config.MAX_POSITIONING_PER_DUTY:
            return True
        
        # 如果总置位数已达到限制，过滤掉
        if total_positioning >= config.MAX_POSITIONING_TOTAL:
            return True
        
        return False
    
    def _should_consider_positioning(self, task_series, crew_info, crew_state):
        """判断是否应该考虑置位任务（放宽约束以提高占位任务覆盖率）"""
        # 注意：占位任务(ground_duty)不是置位任务，应该优先分配
        task_type = task_series.get('type', '')
        

        
        # 占位任务(ground_duty)应该优先分配，不受置位限制
        if task_type == 'ground_duty':
            return True
        
        # 以下逻辑仅适用于真正的置位任务(positioning_flight, positioning_bus)
        if 'positioning' not in task_type:
            return True  # 非置位任务直接允许
        
        # 置位任务的限制检查（进一步放宽，特别是大巴置位）
        current_duty_positioning = crew_state.get('duty_positioning', 0)
        total_positioning = crew_state.get('total_positioning', 0)
        
        # 大巴置位和飞行置位同等考虑，使用相同的放宽限制
        if task_type in ['positioning_bus', 'positioning_flight']:
            # 置位任务的限制放宽（大巴置位和飞行置位同等对待）
            if current_duty_positioning >= config.MAX_POSITIONING_PER_DUTY + 1:  # 允许多一次
                return False
            if total_positioning >= config.MAX_POSITIONING_TOTAL + 2:  # 允许多两次
                return False
        
        # 进一步放宽置位任务的时机限制
        duty_task_count = crew_state.get('duty_task_count', 0)
        duty_start_time = crew_state.get('duty_start_time')
        
        # 大巴置位和飞行置位享受相同的宽松时机限制
        if duty_start_time is not None and duty_task_count > 0:
            task_end_time = task_series['endTime']
            remaining_duty_time = 12 - (task_end_time - duty_start_time).total_seconds() / 3600
            
            # 大巴置位和飞行置位都使用相同的宽松时间限制
            min_remaining_time = 0.5  # 统一使用0.5小时的宽松限制
            if remaining_duty_time < min_remaining_time:
                return False
        
        # 放宽：不强制要求当前位置没有飞行任务才能置位
        # 允许在有飞行任务的情况下也进行置位，以提高灵活性
        
        # 检查置位后是否能连接到有价值的航班（保留此检查以确保置位有意义）
        arrival_airport = task_series['arriAirport']
        future_flights = self.unified_tasks_df[
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task_series['endTime']) &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids)) &
            (self.unified_tasks_df['taskId'].isin(self.dh.crew_leg_map.get(crew_info['crewId'], set())))
        ]
        
        # 放宽：即使没有后续航班，也允许置位（可能用于回基地等）
        # 但如果有后续航班，给予更高优先级
        
        return True  # 总是允许置位，让奖励机制来引导选择
    
    def _task_to_feature_vector(self, task, crew_state, crew_info):
        """
        将任务转换为特征向量，用于强化学习的动作表示。
        扩展版本：包含原有13维特征 + 7维伪对偶价格特征
        """
        features = np.zeros(config.ACTION_DIM)
        
        # 原有特征（0-12）
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
        
        # 新增伪对偶价格特征（13-19）
        # 13. 任务稀缺性评分（基于可执行机组数）
        features[13] = self._calculate_task_scarcity(task, crew_info)
        
        # 14. 时间紧迫性评分（基于起飞时间）
        features[14] = self._calculate_time_urgency(task)
        
        # 15. 立即价值评分（执行此任务的即时收益估算）
        features[15] = self._calculate_immediate_value(task, crew_state, crew_info)
        
        # 16. 机组负载平衡评分（模拟机组对偶价格）
        features[16] = self._calculate_crew_load_balance(crew_info)
        
        # 17. 全局覆盖压力（基于剩余未分配任务）
        features[17] = self._calculate_coverage_pressure(task)
        
        # 18. 连接效率评分（后续任务连接潜力）
        features[18] = self._calculate_connection_efficiency(task, crew_state)
        
        # 19. 约束风险评分（执行后的约束违规风险）
        features[19] = self._estimate_constraint_risk(task, crew_state)
        
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
            logger.warning(f"无效动作选择 - 机组: {crew_id}, 动作索引: {action_idx}, 可用动作数: {len(actions_to_consider)}")
        else:
            task = actions_to_consider[action_idx]
            self.roster_plan[crew_id].append(task)
            task_type = task.get('type', '')
            
            if task_type == 'flight':
                if task['taskId'] in self.unassigned_flight_ids: 
                    self.unassigned_flight_ids.remove(task['taskId'])
            elif task_type == 'ground_duty':
                if task['taskId'] in self.unassigned_ground_duty_ids: 
                    self.unassigned_ground_duty_ids.remove(task['taskId'])
            
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
                crew_state['duty_positioning'] = 0  # 重置值勤内置位计数
            
            crew_state['duty_task_count'] += 1
            if task_type == 'flight' and not task.get('is_positioning', False):
                crew_state['duty_flight_count'] += 1
                crew_state['duty_flight_time'] += task.get('flyTime', 0) / 60.0
                crew_state['total_flight_hours'] += task.get('flyTime', 0) / 60.0
            elif 'positioning' in task_type:
                crew_state['total_positioning'] += 1
                crew_state['duty_positioning'] += 1  # 更新值勤内置位计数
                # 飞行置位的飞行时间不计入总飞行时间（根据用户要求）
            
            # 更新外站过夜
            if task['arriAirport'] != crew_info['base']:
                crew_state['total_away_overnights'] += 1
            
            # 更新日历天数
            crew_state['total_calendar_days'].add(task['startTime'].date())
            crew_state['total_calendar_days'].add(task['endTime'].date())
            
            # 更新成本（简化版）
            crew_state['cost'] += task.get('flyTime', 0) * 0.1  # 简化的成本计算
            
            immediate_reward = self._calculate_immediate_reward(task, crew_info)
            violation_penalty = task.get('violation_penalty', 0) * config.PENALTY_RULE_VIOLATION * 0.2
            reward += immediate_reward
            reward -= violation_penalty
            


        self.total_steps += 1
        
        if self.total_steps % len(self.crews) == 0: self.decision_priority_queue = self._build_priority_queue()
        
        terminated = not self.unassigned_flight_ids and not self.unassigned_ground_duty_ids
        # 优化的截断条件：添加总步数硬限制，适配大规模机组
        max_steps_limit = min(
            len(self.crews) * config.MAX_STEPS_PER_CREW * 1.5,
            config.MAX_TOTAL_STEPS  # 总步数硬限制
        )
        truncated = not self.decision_priority_queue or self.total_steps >= max_steps_limit
        
        if terminated or truncated:
            reward += calculate_final_score(self.roster_plan, self.dh)
            observation, info = None, {'valid_actions': []}
        else:
            observation, info = self._get_observation()
        
        return observation, reward, terminated, truncated, info

    def _calculate_immediate_reward(self, task, crew_info):
        """改进的奖励函数，考虑多维度因素"""
        reward = 0
        task_type = task.get('type', '')
        crew_state = self.crew_states[crew_info['crewId']]
        
        # 基础任务奖励
        if task_type == 'flight':
            reward += config.IMMEDIATE_COVERAGE_REWARD
            # 飞行时间奖励（鼓励高利用率）
            fly_time_hours = task.get('flyTime', 0) / 60.0
            reward += fly_time_hours * 0.8
            
            # 航班紧迫性奖励（基于剩余未分配航班数）
            urgency_bonus = self._calculate_flight_urgency(task)
            reward += urgency_bonus
            
        elif 'positioning' in task_type:
            # 动态置位奖励/惩罚机制
            positioning_reward = self._calculate_dynamic_positioning_reward(task, crew_info, crew_state)
            reward += positioning_reward
        
        elif task_type == 'ground_duty':
            # 占位任务覆盖奖励（提高优先级）
            reward += config.GROUND_DUTY_COVERAGE_REWARD
            
            # 占位任务优先级奖励
            reward += config.GROUND_DUTY_PRIORITY_BONUS
            
            # 额外奖励：如果覆盖了关键时段的占位任务
            if self._is_critical_ground_duty(task):
                reward += config.CRITICAL_GROUND_DUTY_BONUS
            
            # 如果是分配给当前机组的占位任务，给予更高奖励
            if task.get('crewId') == crew_info['crewId']:
                reward += 5.0  # 专属占位任务奖励
        
        # 资源利用效率奖励
        efficiency_bonus = self._calculate_resource_efficiency(task, crew_state, crew_info)
        reward += efficiency_bonus
        
        # 回基地奖励
        if task.get('arriAirport') == crew_info['base']:
            reward += 1.0
        
        # 约束违规惩罚预测
        violation_risk = self._estimate_violation_risk(task, crew_state)
        reward -= violation_risk * 0.5
        
        return reward
    
    def _calculate_flight_urgency(self, task):
        """计算航班的紧迫性，基于时间窗口和可用机组数"""
        # 简化实现：基于起飞时间的紧迫性
        time_to_departure = (task['startTime'] - self.planning_start_dt).total_seconds() / 3600
        if time_to_departure < 24:  # 24小时内起飞
            return 2.0
        elif time_to_departure < 48:  # 48小时内起飞
            return 1.0
        return 0.5
    
    def _calculate_positioning_value(self, task, crew_info):
        """计算调机的战略价值（增强版）"""
        strategic_value = 0.0
        arrival_airport = task['arriAirport']
        crew_id = crew_info['crewId']
        
        # 1. 置位-执行链价值评估
        chain_value = self._evaluate_positioning_chain(task, crew_info)
        strategic_value += chain_value
        
        # 2. 覆盖难点区域价值
        coverage_value = self._evaluate_coverage_value(arrival_airport)
        strategic_value += coverage_value
        
        # 3. 占位任务帮助价值（新增）
        ground_duty_help_value = self._evaluate_ground_duty_help(task, crew_info)
        strategic_value += ground_duty_help_value
        
        # 4. 航班连接价值（增强）
        connection_value = self._evaluate_enhanced_connection_value(task, crew_info)
        strategic_value += connection_value
        
        # 5. 稀缺性权重（新增）
        scarcity_value = self._evaluate_positioning_scarcity(task, crew_info)
        strategic_value += scarcity_value
        
        # 6. 时间窗口紧迫性（新增）
        urgency_value = self._evaluate_time_urgency(task)
        strategic_value += urgency_value
        
        # 7. 回基地价值
        if arrival_airport == crew_info['base']:
            strategic_value += 3.0
        
        # 8. 机场航班密度价值
        density_value = self._evaluate_airport_density(arrival_airport)
        strategic_value += density_value
        
        return strategic_value
    
    def _evaluate_positioning_chain(self, task, crew_info):
        """评估置位-执行链的价值"""
        arrival_airport = task['arriAirport']
        crew_id = crew_info['crewId']
        
        # 查找置位后可执行的航班
        future_flights = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task['endTime']) &
            (self.unified_tasks_df['startTime'] <= task['endTime'] + pd.Timedelta(hours=48)) &  # 48小时内
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids)) &
            (self.unified_tasks_df['taskId'].isin(self.dh.crew_leg_map.get(crew_id, set())))
        ]
        
        flight_count = len(future_flights)
        
        if flight_count == 0:
            return 0.0
        elif flight_count == 1:
            return config.POSITIONING_STRATEGIC_BONUS * 0.5  # 单个航班
        elif flight_count >= 2:
            return config.POSITIONING_CHAIN_BONUS  # 多个航班，给予链式奖励
        
        return 0.0
    
    def _evaluate_coverage_value(self, airport):
        """评估置位到该机场的覆盖价值"""
        # 统计该机场未覆盖的航班数量
        uncovered_flights = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == airport) &
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids))
        ]
        
        uncovered_count = len(uncovered_flights)
        
        # 如果该机场有大量未覆盖航班，置位价值更高
        if uncovered_count >= 5:
            return config.POSITIONING_COVERAGE_BONUS
        elif uncovered_count >= 3:
            return config.POSITIONING_COVERAGE_BONUS * 0.6
        elif uncovered_count >= 1:
            return config.POSITIONING_COVERAGE_BONUS * 0.3
        
        return 0.0
    
    def _evaluate_airport_density(self, airport):
        """评估机场的航班密度价值"""
        # 计算该机场在未来24小时内的航班密度
        current_time = pd.Timestamp.now()  # 使用当前规划时间
        
        airport_flights = self.unified_tasks_df[
            ((self.unified_tasks_df['depaAirport'] == airport) | 
             (self.unified_tasks_df['arriAirport'] == airport)) &
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['startTime'] >= current_time) &
            (self.unified_tasks_df['startTime'] <= current_time + pd.Timedelta(hours=24))
        ]
        
        density = len(airport_flights)
        
        # 高密度机场给予更高价值
        if density >= 10:
            return 2.0
        elif density >= 5:
            return 1.0
        elif density >= 2:
            return 0.5
        
        return 0.0
    
    def _evaluate_ground_duty_help(self, task, crew_info):
        """评估置位是否帮助完成占位任务"""
        arrival_airport = task['arriAirport']
        crew_id = crew_info['crewId']
        
        # 查找该机组在目标机场的未完成占位任务
        crew_ground_duties = self.dh.crew_ground_duties.get(crew_id, [])
        
        for ground_duty in crew_ground_duties:
            if (ground_duty.get('location') == arrival_airport and 
                ground_duty.get('id') in self.unassigned_ground_duty_ids and
                task['endTime'] <= ground_duty.get('startTime', pd.Timestamp.max)):
                return 30.0  # 高价值：帮助完成占位任务
        
        return 0.0
    
    def _evaluate_enhanced_connection_value(self, task, crew_info):
        """增强的航班连接价值评估"""
        arrival_airport = task['arriAirport']
        crew_id = crew_info['crewId']
        
        # 查找置位后的航班连接
        future_flights = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task['endTime']) &
            (self.unified_tasks_df['startTime'] <= task['endTime'] + pd.Timedelta(hours=72)) &
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids)) &
            (self.unified_tasks_df['taskId'].isin(self.dh.crew_leg_map.get(crew_id, set())))
        ]
        
        connection_value = 0.0
        flight_count = len(future_flights)
        
        if flight_count >= 3:
            connection_value += 25.0  # 多航班连接链
        elif flight_count == 2:
            connection_value += 15.0  # 双航班连接
        elif flight_count == 1:
            connection_value += 8.0   # 单航班连接
        
        # 考虑连接时间的合理性
        if flight_count > 0:
            min_connection_time = (future_flights.iloc[0]['startTime'] - task['endTime']).total_seconds() / 3600
            if 2 <= min_connection_time <= 6:  # 理想连接时间
                connection_value += 5.0
        
        return connection_value
    
    def _evaluate_positioning_scarcity(self, task, crew_info):
        """评估置位任务的稀缺性权重"""
        arrival_airport = task['arriAirport']
        
        # 计算能到达该机场的其他置位选项数量
        alternative_positioning = self.unified_tasks_df[
            (self.unified_tasks_df['arriAirport'] == arrival_airport) &
            (self.unified_tasks_df['type'].str.contains('positioning', na=False)) &
            (self.unified_tasks_df['startTime'] >= task['startTime'] - pd.Timedelta(hours=12)) &
            (self.unified_tasks_df['startTime'] <= task['startTime'] + pd.Timedelta(hours=12))
        ]
        
        alternative_count = len(alternative_positioning)
        
        if alternative_count <= 1:  # 唯一或极少选择
            return 15.0
        elif alternative_count <= 3:  # 较少选择
            return 8.0
        elif alternative_count <= 5:  # 中等选择
            return 3.0
        else:  # 选择较多
            return 0.0
    
    def _evaluate_time_urgency(self, task):
        """评估时间窗口紧迫性"""
        current_time = pd.Timestamp.now()
        time_to_start = (task['startTime'] - current_time).total_seconds() / 3600
        
        if time_to_start <= 6:  # 6小时内开始
            return 12.0  # 高紧迫性
        elif time_to_start <= 12:  # 12小时内开始
            return 6.0   # 中等紧迫性
        elif time_to_start <= 24:  # 24小时内开始
            return 2.0   # 低紧迫性
        else:
            return 0.0   # 无紧迫性
    
    def _evaluate_task_completion_risk(self, task, crew_state):
        """评估任务完成的风险和紧迫性"""
        risk_score = 0.0
        
        # 时间窗口风险
        if task['type'] == 'ground_duty':
            time_window = (task['endTime'] - task['startTime']).total_seconds() / 3600
            if time_window <= 2:  # 极短时间窗口
                risk_score += 0.5
            elif time_window <= 6:  # 短时间窗口
                risk_score += 0.3
        
        # 时间紧迫性风险
        current_time = pd.Timestamp.now()
        time_to_deadline = (task['endTime'] - current_time).total_seconds() / 3600
        if time_to_deadline <= 12:  # 12小时内必须完成
            risk_score += 0.4
        elif time_to_deadline <= 24:  # 24小时内必须完成
            risk_score += 0.2
        
        # 机组位置风险（是否需要长距离移动）
        if crew_state.get('last_location') != task.get('depaAirport'):
            risk_score += 0.2
        
        # 工作负载风险
        current_task_count = len(self.roster_plan.get(crew_state.get('crew_id', ''), []))
        if current_task_count == 0:  # 机组还没有任务，风险较高
            risk_score += 0.3
        
        return min(risk_score, 1.0)  # 限制在0-1之间
    
    def _calculate_current_coverage_rate(self):
        """计算当前的航班覆盖率"""
        total_flights = len(self.unified_tasks_df[self.unified_tasks_df['type'] == 'flight'])
        covered_flights = total_flights - len(self.unassigned_flight_ids)
        
        if total_flights == 0:
            return 1.0
        
        return covered_flights / total_flights
    
    def _calculate_positioning_strategic_value(self, task, crew_info):
        """计算置位任务的战略价值（用于优先级计算）"""
        strategic_value = 0.0
        arrival_airport = task['arriAirport']
        crew_id = crew_info['crewId']
        
        # 1. 后续航班连接价值
        future_flights = self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['startTime'] > task['endTime']) &
            (self.unified_tasks_df['startTime'] <= task['endTime'] + pd.Timedelta(hours=48)) &
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids)) &
            (self.unified_tasks_df['taskId'].isin(self.dh.crew_leg_map.get(crew_id, set())))
        ]
        
        flight_count = len(future_flights)
        if flight_count >= 2:
            strategic_value += 15.0  # 多航班连接
        elif flight_count == 1:
            strategic_value += 8.0   # 单航班连接
        
        # 2. 回基地价值
        if arrival_airport == crew_info['base']:
            strategic_value += 10.0
        
        # 3. 难点区域覆盖价值
        uncovered_flights_at_airport = len(self.unified_tasks_df[
            (self.unified_tasks_df['depaAirport'] == arrival_airport) &
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids))
        ])
        
        if uncovered_flights_at_airport >= 5:
            strategic_value += 12.0
        elif uncovered_flights_at_airport >= 3:
            strategic_value += 6.0
        elif uncovered_flights_at_airport >= 1:
            strategic_value += 3.0
        
        return strategic_value
    
    def _calculate_dynamic_positioning_reward(self, task, crew_info, crew_state):
        """动态置位奖励/惩罚机制：奖励有效置位，惩罚无效置位"""
        reward = 0.0
        
        # 1. 评估置位的战略价值
        strategic_value = self._calculate_positioning_value(task, crew_info)
        
        # 2. 根据战略价值决定奖励/惩罚策略
        if strategic_value >= config.POSITIONING_CHAIN_BONUS:  # 高价值置位（能连接多个航班）
            # 关键置位：明确奖励
            reward += strategic_value
            reward += 5.0  # 额外的关键置位奖励
            
        elif strategic_value >= config.POSITIONING_STRATEGIC_BONUS * 0.5:  # 中等价值置位
            # 有效置位：轻罚甚至奖励
            reward += strategic_value
            reward += config.PENALTY_POSITIONING * 0.3  # 轻微惩罚
            
        else:  # 低价值或无效置位
            # 无效置位：重罚
            reward += config.PENALTY_POSITIONING  # 基础惩罚
            
            # 检查是否有航班可走时进行置位
            if self._has_available_flights(crew_info, crew_state):
                reward += config.PENALTY_NO_FLIGHT_POSITIONING
        
        # 3. 置位数量限制惩罚（所有置位都要检查）
        current_duty_positioning = crew_state.get('duty_positioning', 0)
        total_positioning = crew_state.get('total_positioning', 0)
        
        # 值勤内置位限制
        if current_duty_positioning >= config.MAX_POSITIONING_PER_DUTY:
            reward += config.PENALTY_EXCESS_DUTY_POSITIONING
        
        # 总置位数量限制
        if total_positioning >= config.MAX_POSITIONING_TOTAL:
            reward += config.PENALTY_EXCESS_TOTAL_POSITIONING
        elif total_positioning >= (config.MAX_POSITIONING_TOTAL - 1):
            reward += config.PENALTY_SECOND_POSITIONING
        
        # 4. 值勤日中间置位的额外惩罚
        if self._is_middle_positioning(task, crew_state):
            reward += config.PENALTY_POSITIONING_MIDDLE
        
        # 5. 覆盖率动态调整
        coverage_rate = self._calculate_current_coverage_rate()
        if coverage_rate < 0.6:  # 覆盖率低时，减少置位惩罚
            reward += abs(config.PENALTY_POSITIONING) * 0.5  # 减少50%的基础惩罚
        elif coverage_rate > 0.9:  # 覆盖率高时，增加置位惩罚
            reward += config.PENALTY_POSITIONING * 0.5  # 增加50%的惩罚
        
        return reward
    
    def _filter_valuable_positioning_tasks(self, positioning_tasks, crew_info, crew_state):
        """筛选有价值的置位任务，优先考虑能带来后续航班的置位"""
        if not positioning_tasks:
            return []
        
        valuable_tasks = []
        
        for task in positioning_tasks:
            # 1. 检查基本限制
            if self._should_filter_positioning_action(task, crew_state):
                continue
            
            # 2. 评估置位价值
            strategic_value = self._calculate_positioning_value(task, crew_info)
            

            
            # 3. 分类置位任务（降低门槛，更积极地考虑置位）
            if strategic_value >= config.POSITIONING_CHAIN_BONUS:  # 高价值置位
                task['positioning_priority'] = 'high'
                valuable_tasks.append(task)
            elif strategic_value >= config.POSITIONING_STRATEGIC_BONUS * 0.3:  # 中等价值置位（降低门槛）
                task['positioning_priority'] = 'medium'
                valuable_tasks.append(task)
            else:
                # 低价值置位：更宽松的条件
                coverage_rate = self._calculate_current_coverage_rate()
                if (coverage_rate < 0.7 or  # 覆盖率较低时（从0.5提高到0.7）
                    task.get('type') == 'positioning_bus' or  # 大巴置位更容易被接受
                    strategic_value >= 2.0):  # 有一定战略价值的置位
                    task['positioning_priority'] = 'low'
                    valuable_tasks.append(task)
        
        # 4. 按价值排序
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        valuable_tasks.sort(key=lambda x: -priority_order.get(x.get('positioning_priority', 'low'), 0))
        

        
        # 5. 限制置位任务数量，避免过多选择
        # 但要确保大巴置位任务有机会被选中
        max_positioning_candidates = min(8, len(valuable_tasks))  # 最多考虑8个置位任务（从6增加到8）
        
        # 所有置位任务平等对待，按价值排序后选择
        final_tasks = valuable_tasks[:max_positioning_candidates]
        

        
        return final_tasks
    
    def _calculate_resource_efficiency(self, task, crew_state, crew_info):
        """计算资源利用效率"""
        efficiency = 0
        
        # 连接时间效率（避免过长等待）
        connection_time = (task['startTime'] - crew_state['last_task_end_time']).total_seconds() / 3600
        if 2 <= connection_time <= 6:  # 理想连接时间
            efficiency += 1.0
        elif connection_time > 12:  # 过长等待
            efficiency -= 0.5
        
        # 值勤内任务密度奖励
        if crew_state.get('duty_start_time'):
            duty_duration = (task['endTime'] - crew_state['duty_start_time']).total_seconds() / 3600
            task_density = (crew_state.get('duty_task_count', 0) + 1) / max(duty_duration, 1)
            if 0.3 <= task_density <= 0.6:  # 理想任务密度
                efficiency += 0.8
        
        return efficiency
    
    def _estimate_violation_risk(self, task, crew_state):
        """估计添加任务后的约束违规风险"""
        risk = 0
        
        # 飞行时间风险
        if task.get('type') == 'flight':
            projected_duty_flight_time = crew_state.get('duty_flight_time', 0) + task.get('flyTime', 0) / 60.0
            if projected_duty_flight_time > 7:  # 接近8小时限制
                risk += 1.0
        
        # 值勤时长风险
        if crew_state.get('duty_start_time'):
            projected_duty_duration = (task['endTime'] - crew_state['duty_start_time']).total_seconds() / 3600
            if projected_duty_duration > 10:  # 接近12小时限制
                risk += 1.0
        
        return risk
    
    def _is_critical_ground_duty(self, task):
        """判断是否为关键时段的占位任务"""
        # 关键时段：工作日的白天时段 (8:00-18:00)
        start_hour = task['startTime'].hour
        is_weekday = task['startTime'].weekday() < 5  # 周一到周五
        
        if is_weekday and 8 <= start_hour <= 18:
            return True
        
        # 或者是节假日/周末的重要时段
        if not is_weekday and 10 <= start_hour <= 16:
            return True
            
        return False
    
    def _is_middle_positioning(self, task, crew_state):
        """判断置位任务是否在值勤日中间（严格版本）"""
        # 如果没有值勤开始时间，说明是值勤日的第一个任务，允许置位
        if not crew_state.get('duty_start_time'):
            return False
        
        # 获取当前值勤日的任务信息
        duty_task_count = crew_state.get('duty_task_count', 0)
        
        # 如果值勤日还没有任务，这是第一个任务，允许置位
        if duty_task_count == 0:
            return False
        
        # 检查当前值勤日是否已经有飞行任务
        crew_id = crew_state.get('crew_id')
        current_roster = self.roster_plan.get(crew_id, [])
        duty_start_time = crew_state.get('duty_start_time')
        
        # 分析当前值勤日的任务组成
        duty_has_flight = False
        duty_positioning_count = 0
        
        for roster_task in current_roster:
            # 检查任务是否在当前值勤日内
            if (roster_task.get('startTime') and duty_start_time and 
                roster_task['startTime'] >= duty_start_time):
                
                if roster_task.get('type') == 'flight':
                    duty_has_flight = True
                elif 'positioning' in roster_task.get('type', ''):
                    duty_positioning_count += 1
        
        # 严格的置位规则：
        # 1. 如果值勤日已经有飞行任务，不允许再添加置位任务（置位只能在首末）
        if duty_has_flight:
            return True
        
        # 2. 如果值勤日已经有置位任务，不允许再添加置位任务（一个值勤日最多一个置位）
        if duty_positioning_count >= 1:
            return True
        
        # 3. 如果值勤日已经有多个任务（不管什么类型），认为是中间置位
        if duty_task_count >= 2:
            return True
        
        return False
    
    def _has_available_flights(self, crew_info, crew_state):
        """检查当前位置是否有可执行的航班任务"""
        current_location = crew_state['last_location']
        current_time = crew_state['last_task_end_time']
        crew_id = crew_info['crewId']
        
        # 查找当前位置的可执行飞行任务
        available_flights = self.unified_tasks_df[
            (self.unified_tasks_df['type'] == 'flight') &
            (self.unified_tasks_df['depaAirport'] == current_location) &
            (self.unified_tasks_df['startTime'] > current_time) &
            (self.unified_tasks_df['startTime'] <= current_time + pd.Timedelta(hours=24)) &  # 24小时内
            (self.unified_tasks_df['taskId'].isin(self.unassigned_flight_ids)) &
            (self.unified_tasks_df['taskId'].isin(self.dh.crew_leg_map.get(crew_id, set())))
        ]
        
        return len(available_flights) > 0
