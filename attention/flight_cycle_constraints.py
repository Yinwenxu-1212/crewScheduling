# flight_cycle_constraints.py
"""
飞行周期约束检查模块
实现用户要求的三个关键约束：
1. 飞行周期首尾判断
2. 飞行值勤日开始前最小休息时间限制
3. 飞行值勤日最大飞行值勤时间限制
"""
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from unified_config import UnifiedConfig

class FlightCycleConstraints:
    """飞行周期约束检查器"""
    
    def __init__(self, data_handler):
        self.dh = data_handler
        self.MIN_REST_HOURS = UnifiedConfig.MIN_REST_HOURS  # 飞行值勤日开始前最小休息时间
        self.MAX_DUTY_HOURS = UnifiedConfig.MAX_DUTY_DAY_HOURS  # 飞行值勤日最大值勤时间
        self.MIN_CYCLE_REST_DAYS = UnifiedConfig.MIN_CYCLE_REST_DAYS  # 飞行周期开始前需要完整休息的日历日数
        
    def check_flight_cycle_start(self, crew_state, crew_info):
        """
        检查飞行周期开始条件：
        1. 开始之前在基地并且休息两个完整日历日
        
        Args:
            crew_state: 机组当前状态
            crew_info: 机组基本信息
            
        Returns:
            bool: 是否满足飞行周期开始条件
        """
        # 获取机组基地
        crew_base = crew_info.get('base')
        current_location = crew_state.get('last_location')
        
        # 检查是否在基地
        if current_location != crew_base:
            return False
            
        # 检查是否休息了两个完整日历日
        last_duty_end = crew_state.get('last_duty_end_time')
        current_time = crew_state.get('current_time')
        
        if last_duty_end and current_time:
            # 计算休息的完整日历日数
            rest_days = self._calculate_complete_rest_days(last_duty_end, current_time)
            if rest_days < self.MIN_CYCLE_REST_DAYS:
                return False
                
        return True
        
    def check_flight_cycle_end(self, crew_state, proposed_task):
        """
        检查飞行周期结束条件：
        1. 结束必须是飞行值勤日（有飞行航班任务）
        2. 最好是回到基地，但也可以通过置位回到基地
        
        Args:
            crew_state: 机组当前状态
            proposed_task: 提议的任务
            
        Returns:
            dict: 包含是否满足结束条件和相关信息
        """
        result = {
            'is_valid_end': False,
            'has_flight_task': False,
            'returns_to_base': False,
            'can_return_via_positioning': False
        }
        
        # 检查是否包含飞行任务
        if proposed_task.get('type') == 'flight':
            result['has_flight_task'] = True
            
        # 检查是否回到基地
        crew_base = crew_state.get('base')
        task_end_location = proposed_task.get('arriAirport')
        
        if task_end_location == crew_base:
            result['returns_to_base'] = True
            
        # 检查是否可以通过置位回到基地
        if not result['returns_to_base']:
            positioning_options = self._find_positioning_to_base(
                task_end_location, 
                proposed_task.get('sta', proposed_task.get('ta')),
                crew_base
            )
            if positioning_options:
                result['can_return_via_positioning'] = True
                
        # 飞行周期结束条件：必须有飞行任务，且能回到基地
        result['is_valid_end'] = (
            result['has_flight_task'] and 
            (result['returns_to_base'] or result['can_return_via_positioning'])
        )
        
        return result
        
    def check_min_rest_before_duty(self, crew_state, proposed_task):
        """
        检查飞行值勤日开始前最小休息时间限制：
        值勤日结束后休息开始时间定为最后一个任务的结束时间；
        值勤日开始前休息结束时间定为第一个任务的开始时间。
        飞行值勤日开始前最小休息时间设为12小时。
        
        Args:
            crew_state: 机组当前状态
            proposed_task: 提议的任务
            
        Returns:
            bool: 是否满足最小休息时间要求
        """
        last_task_end = crew_state.get('last_task_end_time')
        
        if not last_task_end:
            return True  # 如果没有前一个任务，认为满足条件
            
        # 获取提议任务的开始时间
        if proposed_task.get('type') == 'flight':
            task_start = proposed_task.get('std')
        else:
            task_start = proposed_task.get('td', proposed_task.get('startTime'))
            
        if not task_start:
            return False
            
        # 计算休息时间
        rest_duration = (task_start - last_task_end).total_seconds() / 3600.0
        
        return rest_duration >= self.MIN_REST_HOURS
        
    def check_max_duty_time(self, crew_state, proposed_task):
        """
        检查飞行值勤日最大飞行值勤时间限制：
        对于每个飞行值勤日而言，飞行值勤开始时间为第一个任务的开始时间，
        飞行值勤结束时间为最后一个飞行任务的到达时间。
        飞行值勤日值勤时间不超过12小时。
        
        Args:
            crew_state: 机组当前状态
            proposed_task: 提议的任务
            
        Returns:
            bool: 是否满足最大值勤时间要求
        """
        duty_start = crew_state.get('duty_start_time')
        
        # 如果是新的值勤日开始
        if not duty_start:
            return True  # 新值勤日总是满足条件
            
        # 获取提议任务的结束时间
        if proposed_task.get('type') == 'flight':
            task_end = proposed_task.get('sta')
        else:
            task_end = proposed_task.get('ta', proposed_task.get('endTime'))
            
        if not task_end:
            return False
            
        # 计算值勤时间
        duty_duration = (task_end - duty_start).total_seconds() / 3600.0
        
        return duty_duration <= self.MAX_DUTY_HOURS
        
    def calculate_flight_cycle_time(self, crew_state, proposed_task):
        """
        计算飞行周期的时间：停止休息——最后一个飞行任务
        
        Args:
            crew_state: 机组当前状态
            proposed_task: 提议的任务
            
        Returns:
            float: 飞行周期时间（小时）
        """
        cycle_start = crew_state.get('cycle_start_time')
        
        if not cycle_start:
            return 0.0
            
        # 如果提议任务是飞行任务，使用其结束时间
        if proposed_task.get('type') == 'flight':
            cycle_end = proposed_task.get('sta')
        else:
            # 如果不是飞行任务，使用当前最后一个飞行任务的时间
            cycle_end = crew_state.get('last_flight_end_time')
            
        if not cycle_end:
            return 0.0
            
        cycle_duration = (cycle_end - cycle_start).total_seconds() / 3600.0
        return max(0.0, cycle_duration)
        
    def _calculate_complete_rest_days(self, start_time, end_time):
        """
        计算两个时间点之间的完整日历日数
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            int: 完整日历日数
        """
        if not start_time or not end_time:
            return 0
            
        # 获取开始日期的下一天和结束日期
        start_date = (start_time + timedelta(days=1)).date()
        end_date = end_time.date()
        
        # 计算完整日历日数
        if end_date > start_date:
            return (end_date - start_date).days
        else:
            return 0
            
    def _find_positioning_to_base(self, current_airport, current_time, base_airport):
        """
        查找从当前机场到基地的置位选项
        
        Args:
            current_airport: 当前机场
            current_time: 当前时间
            base_airport: 基地机场
            
        Returns:
            list: 可用的置位选项
        """
        positioning_options = []
        max_search_hours = 24  # 最多搜索24小时内的置位
        
        # 搜索大巴置位
        for _, bus in self.dh.data['bus_info'].iterrows():
            if (bus['depaAirport'] == current_airport and 
                bus['arriAirport'] == base_airport and
                bus['td'] >= current_time and
                bus['td'] <= current_time + timedelta(hours=max_search_hours)):
                positioning_options.append({
                    'type': 'bus_positioning',
                    'task': bus,
                    'departure_time': bus['td'],
                    'arrival_time': bus['ta']
                })
                
        # 搜索飞行置位（如果有相关数据）
        # 这里可以根据实际数据结构添加飞行置位的搜索逻辑
        
        return positioning_options
        
    def validate_complete_schedule(self, crew_schedule):
        """
        验证完整的机组排班是否满足所有约束
        
        Args:
            crew_schedule: 机组完整排班
            
        Returns:
            dict: 验证结果
        """
        violations = []
        warnings = []
        
        # 检查每个值勤日的约束
        duty_days = self._group_tasks_by_duty_day(crew_schedule)
        
        for duty_day, tasks in duty_days.items():
            # 检查值勤时间约束
            if not self._validate_duty_day_constraints(tasks):
                violations.append(f"值勤日 {duty_day} 违反时间约束")
                
            # 检查休息时间约束
            if not self._validate_rest_constraints(tasks):
                violations.append(f"值勤日 {duty_day} 违反休息时间约束")
                
        # 检查飞行周期约束
        cycle_validation = self._validate_flight_cycle_constraints(crew_schedule)
        violations.extend(cycle_validation.get('violations', []))
        warnings.extend(cycle_validation.get('warnings', []))
        
        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'warnings': warnings
        }
        
    def _group_tasks_by_duty_day(self, crew_schedule):
        """
        按值勤日分组任务
        
        Args:
            crew_schedule: 机组排班
            
        Returns:
            dict: 按值勤日分组的任务
        """
        duty_days = defaultdict(list)
        
        for task in crew_schedule:
            # 根据任务开始时间确定值勤日
            if task.get('type') == 'flight':
                task_date = task['std'].date()
            else:
                task_date = task.get('td', task.get('startTime')).date()
                
            duty_days[task_date].append(task)
            
        return duty_days
        
    def _validate_duty_day_constraints(self, tasks):
        """
        验证单个值勤日的约束
        
        Args:
            tasks: 值勤日内的任务列表
            
        Returns:
            bool: 是否满足约束
        """
        if not tasks:
            return True
            
        # 按时间排序任务
        sorted_tasks = sorted(tasks, key=lambda x: x.get('std', x.get('td', x.get('startTime'))))
        
        # 获取第一个和最后一个任务时间
        first_task = sorted_tasks[0]
        last_task = sorted_tasks[-1]
        
        if first_task.get('type') == 'flight':
            duty_start = first_task['std']
        else:
            duty_start = first_task.get('td', first_task.get('startTime'))
            
        # 找到最后一个飞行任务
        last_flight_task = None
        for task in reversed(sorted_tasks):
            if task.get('type') == 'flight':
                last_flight_task = task
                break
                
        if last_flight_task:
            duty_end = last_flight_task['sta']
        else:
            # 如果没有飞行任务，使用最后一个任务的结束时间
            if last_task.get('type') == 'flight':
                duty_end = last_task['sta']
            else:
                duty_end = last_task.get('ta', last_task.get('endTime'))
                
        # 检查值勤时间
        if duty_start and duty_end:
            duty_duration = (duty_end - duty_start).total_seconds() / 3600.0
            return duty_duration <= self.MAX_DUTY_HOURS
            
        return True
        
    def _validate_rest_constraints(self, tasks):
        """
        验证休息时间约束
        
        Args:
            tasks: 任务列表
            
        Returns:
            bool: 是否满足休息时间约束
        """
        # 这里可以添加更详细的休息时间验证逻辑
        return True
        
    def _validate_flight_cycle_constraints(self, crew_schedule):
        """
        验证飞行周期约束
        
        Args:
            crew_schedule: 机组完整排班
            
        Returns:
            dict: 验证结果
        """
        violations = []
        warnings = []
        
        # 这里可以添加飞行周期的详细验证逻辑
        
        return {
            'violations': violations,
            'warnings': warnings
        }