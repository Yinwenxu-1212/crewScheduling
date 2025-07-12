# optimized_reward_function.py
"""
优化的奖励函数，智能评估置位价值
"""
import numpy as np
from datetime import timedelta
from optimized_unified_config import OptimizedUnifiedConfig
from flight_cycle_constraints import FlightCycleConstraints

class OptimizedRewardFunction:
    """优化的奖励函数"""
    
    def __init__(self, data_handler, precompute_manager):
        self.dh = data_handler
        self.precompute = precompute_manager
        self.config = OptimizedUnifiedConfig
        
        # 初始化飞行周期约束检查器
        self.cycle_constraints = FlightCycleConstraints(data_handler)
        
        # 预计算机场需求
        self._compute_airport_demand()
        
    def _compute_airport_demand(self):
        """预计算各机场的航班需求"""
        self.airport_demand = {}
        
        for _, flight in self.dh.data['flights'].iterrows():
            depa = flight['depaAirport']
            self.airport_demand[depa] = self.airport_demand.get(depa, 0) + 1
            
        # 归一化
        max_demand = max(self.airport_demand.values()) if self.airport_demand else 1
        for airport in self.airport_demand:
            self.airport_demand[airport] /= max_demand
            
    def calculate_reward(self, action, crew_state, env_state):
        """
        计算动作奖励
        
        Args:
            action: 执行的动作
            crew_state: 机组状态
            env_state: 环境状态（包含覆盖率等全局信息）
        """
        reward = 0.0
        
        task_type = action.get('type')
        task = action.get('task')
        
        # 1. 基础奖励/惩罚
        if task_type == 'flight':
            # 飞行奖励
            fly_hours = task.get('flyTime', 0) / 60.0
            reward += fly_hours * self.config.FLIGHT_TIME_REWARD
            
            # 额外奖励：如果这是一个难以覆盖的航班
            if self._is_hard_to_cover_flight(task, env_state):
                reward += 50  # 额外奖励
                
        elif 'positioning' in task_type:
            # 智能置位评估
            positioning_info = self._evaluate_positioning(task, crew_state, env_state)
            positioning_penalty = self.config.get_dynamic_positioning_penalty(
                env_state.get('coverage_rate', 0),
                positioning_info
            )
            reward -= positioning_penalty
            
        elif task_type == 'ground_duty':
            # 占位任务
            if task.get('isDuty'):
                reward -= 10  # 轻微惩罚值勤占位
            else:
                reward += 5   # 轻微奖励休息占位
                
        # 2. 连接质量奖励
        connection_reward = self._evaluate_connection_quality(action, crew_state)
        reward += connection_reward
        
        # 3. 全局平衡奖励
        balance_reward = self._evaluate_global_balance(action, env_state)
        reward += balance_reward
        
        # 4. 约束违规惩罚
        constraint_violations = self._check_all_constraints(action, crew_state)
        if constraint_violations['has_violations']:
            reward -= self.config.VIOLATION_PENALTY * constraint_violations['violation_count']
            
        # 5. 飞行周期奖励/惩罚
        cycle_reward = self._evaluate_flight_cycle_compliance(action, crew_state)
        reward += cycle_reward
            
        return reward
        
    def _evaluate_positioning(self, task, crew_state, env_state):
        """评估置位的价值"""
        positioning_info = {
            'leads_to_flights': 0,
            'returns_to_base': False,
            'to_high_demand_airport': False
        }
        
        # 检查置位后可执行的航班
        if task.get('type') == 'positioning_flight':
            arrival_airport = task.get('arriAirport')
            arrival_time = task.get('sta')
        else:  # positioning_bus
            arrival_airport = task.get('arriAirport')
            arrival_time = task.get('ta')
            
        # 查找置位后可执行的航班
        potential_flights = self._find_flights_after_positioning(
            arrival_airport, arrival_time, crew_state
        )
        positioning_info['leads_to_flights'] = len(potential_flights)
        
        # 检查是否返回基地
        crew_base = crew_state.get('base')
        if arrival_airport == crew_base:
            positioning_info['returns_to_base'] = True
            
        # 检查是否到高需求机场
        if self.airport_demand.get(arrival_airport, 0) > 0.7:
            positioning_info['to_high_demand_airport'] = True
            
        return positioning_info
        
    def _find_flights_after_positioning(self, airport, arrival_time, crew_state):
        """查找置位后可执行的航班"""
        potential_flights = []
        max_wait_time = timedelta(hours=6)
        
        # 使用预计算的航班索引
        airport_flights = self.precompute.features_cache.get('tasks_by_airport', {}).get(airport, [])
        
        for task_type, flight in airport_flights:
            if task_type != 'flight':
                continue
                
            if flight['std'] > arrival_time and flight['std'] - arrival_time <= max_wait_time:
                # 检查机组是否有资质执行
                if self.precompute.is_compatible(crew_state['crewId'], f"flight_{flight['id']}"):
                    potential_flights.append(flight)
                    
        return potential_flights
        
    def _is_hard_to_cover_flight(self, flight, env_state):
        """判断是否是难以覆盖的航班"""
        # 获取可以执行此航班的机组数
        compatible_crews = 0
        for crew_id in self.dh.data['crews']['crewId']:
            if self.precompute.is_compatible(crew_id, f"flight_{flight['id']}"):
                compatible_crews += 1
                
        # 如果可执行机组少，或者时间紧迫，认为是难覆盖航班
        if compatible_crews < 5:
            return True
            
        # 检查时间紧迫性
        time_to_departure = (flight['std'] - env_state.get('current_time')).total_seconds() / 3600
        if time_to_departure < 12:  # 12小时内起飞
            return True
            
        return False
        
    def _evaluate_connection_quality(self, action, crew_state):
        """评估连接质量"""
        reward = 0.0
        
        if not crew_state.get('last_task_end_time'):
            return reward
            
        # 计算连接时间
        task = action['task']
        if action['type'] == 'flight':
            connection_time = (task['std'] - crew_state['last_task_end_time']).total_seconds() / 3600
        else:
            connection_time = (task.get('td', task.get('startTime')) - crew_state['last_task_end_time']).total_seconds() / 3600
            
        # 理想连接时间：2-4小时
        if 2 <= connection_time <= 4:
            reward += 10
        elif connection_time < 2:
            reward -= 20  # 太紧张
        elif connection_time > 8:
            reward -= 10  # 等待太久
            
        # 同机型奖励
        if (crew_state.get('last_aircraft_no') and 
            task.get('aircraftNo') == crew_state['last_aircraft_no']):
            reward += 15
            
        return reward
        
    def _evaluate_global_balance(self, action, env_state):
        """评估全局平衡"""
        reward = 0.0
        
        # 机组工作负载平衡
        crew_workload = env_state.get('crew_workloads', {})
        avg_workload = np.mean(list(crew_workload.values())) if crew_workload else 0
        current_crew_workload = crew_workload.get(action.get('crew_id'), 0)
        
        # 如果当前机组工作量低于平均，给予奖励
        if current_crew_workload < avg_workload * 0.8:
            reward += 20
        elif current_crew_workload > avg_workload * 1.2:
            reward -= 20
            
        return reward
        
    def _check_all_constraints(self, action, crew_state):
        """检查所有约束条件"""
        violations = []
        task = action.get('task')
        
        # 1. 检查最小休息时间约束
        if not self.cycle_constraints.check_min_rest_before_duty(crew_state, task):
            violations.append('min_rest_violation')
            
        # 2. 检查最大值勤时间约束
        if not self.cycle_constraints.check_max_duty_time(crew_state, task):
            violations.append('max_duty_time_violation')
            
        # 3. 检查飞行周期开始条件（如果适用）
        if crew_state.get('is_cycle_start_candidate', False):
            if not self.cycle_constraints.check_flight_cycle_start(crew_state, action.get('crew_info', {})):
                violations.append('cycle_start_violation')
                
        return {
            'has_violations': len(violations) > 0,
            'violation_count': len(violations),
            'violations': violations
        }
        
    def _evaluate_flight_cycle_compliance(self, action, crew_state):
        """评估飞行周期合规性并给予相应奖励/惩罚"""
        reward = 0.0
        task = action.get('task')
        
        # 1. 飞行周期结束奖励
        if action.get('type') == 'flight':
            cycle_end_info = self.cycle_constraints.check_flight_cycle_end(crew_state, task)
            
            if cycle_end_info['is_valid_end']:
                reward += 50  # 有效结束飞行周期的奖励
                
                if cycle_end_info['returns_to_base']:
                    reward += 30  # 直接回到基地的额外奖励
                elif cycle_end_info['can_return_via_positioning']:
                    reward += 15  # 可通过置位回基地的奖励
                    
        # 2. 置位回基地的战略奖励
        elif 'positioning' in action.get('type', ''):
            crew_base = crew_state.get('base')
            arrival_airport = task.get('arriAirport')
            
            if arrival_airport == crew_base:
                # 检查是否有助于结束飞行周期
                if crew_state.get('has_flight_tasks_in_cycle', False):
                    reward += 25  # 置位回基地结束周期的奖励
                    
        # 3. 飞行周期时间管理奖励
        cycle_time = self.cycle_constraints.calculate_flight_cycle_time(crew_state, task)
        if cycle_time > 0:
            # 合理的飞行周期时间（72-168小时，即3-7天）
            if 72 <= cycle_time <= 168:
                reward += 10
            elif cycle_time > 168:
                reward -= 5  # 周期过长的轻微惩罚
                
        return reward