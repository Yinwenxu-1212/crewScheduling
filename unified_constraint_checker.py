#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一约束检查模块
将scoring_system中的约束检查逻辑提取出来，供生成器和求解器共同使用
确保生成时和评分时使用相同的约束检查逻辑
"""

from datetime import datetime, timedelta
from typing import List, Dict, Set, Optional, Tuple
from data_models import Flight, Crew, GroundDuty, BusInfo, DutyDay, FlightDutyPeriod, Label

class UnifiedConstraintChecker:
    """统一的约束检查器，确保生成和评分使用相同的逻辑"""
    
    def __init__(self, layover_stations_set: Set[str]):
        self.layover_stations_set = layover_stations_set
        
        # 约束参数
        self.MAX_DUTY_DAY_HOURS = 12.0  # 修正：飞行值勤日最大值勤时间不超过12小时
        self.MIN_REST_HOURS = 12.0
        self.MAX_FLIGHTS_IN_DUTY = 4
        self.MAX_TASKS_IN_DUTY = 6
        self.MAX_FLIGHT_TIME_IN_DUTY_HOURS = 8.0
        self.MAX_TOTAL_FLIGHT_HOURS = 60.0  # 修正：总飞行值勤时间不超过60小时
        self.MAX_FLIGHT_CYCLE_DAYS = 4
        self.MIN_CYCLE_REST_DAYS = 2
        self.MAX_CONSECUTIVE_DUTY_DAYS = 4
        
        # 连接时间参数
        self.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT = timedelta(minutes=30)
        self.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT = timedelta(hours=3)  # 修正：不同飞机间隔不小于3小时
        self.MIN_CONNECTION_TIME_BUS = timedelta(hours=2)  # 修正：大巴置位与飞行任务间隔2小时
        self.DEFAULT_MIN_CONNECTION_TIME = timedelta(hours=1)
        
    def organize_tasks_into_duty_days(self, tasks: List) -> List[DutyDay]:
        """
        将任务组织为值勤日
        使用与scoring_system相同的逻辑
        """
        if not tasks:
            return []
            
        # 按时间排序
        sorted_tasks = sorted(tasks, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        duty_days = []
        current_day = DutyDay()
        
        for i, task in enumerate(sorted_tasks):
            task_start = getattr(task, 'std', getattr(task, 'startTime', None))
            
            if not task_start:
                continue
                
            # 如果是第一个任务，直接加入当前值勤日
            if i == 0:
                current_day.add_task(task)
                continue
                
            # 检查与前一个任务的时间间隔
            prev_task = sorted_tasks[i-1]
            prev_end = getattr(prev_task, 'sta', getattr(prev_task, 'endTime', None))
            
            if prev_end and task_start:
                rest_interval = task_start - prev_end
                
                # 判断是否需要开始新的值勤日
                should_start_new_duty_day = False
                
                # 1. 休息时间超过12小时
                if rest_interval >= timedelta(hours=self.MIN_REST_HOURS):
                    should_start_new_duty_day = True
                    
                # 2. 当前值勤日已经超过24小时
                elif current_day.start_time and (task_start - current_day.start_time) > timedelta(hours=24):
                    should_start_new_duty_day = True
                    
                # 3. 跨越了太多日历日（超过2个日历日）
                elif (current_day.start_time and 
                      (task_start.date() - current_day.start_time.date()).days > 1):
                    should_start_new_duty_day = True
                
                if should_start_new_duty_day:
                    # 结束当前值勤日，开始新的值勤日
                    if current_day.tasks:
                        duty_days.append(current_day)
                    current_day = DutyDay()
            
            current_day.add_task(task)
        
        # 添加最后一个值勤日
        if current_day.tasks:
            duty_days.append(current_day)
        
        # 更新所有值勤日的飞行值勤日状态
        for duty_day in duty_days:
            duty_day.set_layover_stations(self.layover_stations_set)
        
        return duty_days
    
    def can_assign_task_to_label(self, current_label: Label, task: Dict, crew: Crew) -> bool:
        """
        检查是否可以将任务分配给当前标签
        使用新的DutyDay结构进行约束检查
        """
        # 1. 基本时间顺序检查
        if current_label.node and task['startTime'] < current_label.node.time:
            return False
        
        # 2. 地点衔接检查
        if current_label.node and current_label.node.airport != task['depaAirport']:
            return False
        
        # 3. 资格检查（飞行任务）
        if task['type'] == 'flight':
            # 这里需要传入crew_leg_matches_set进行检查
            # 暂时简化处理
            pass
        
        # 4. 连接时间检查
        if current_label.node and current_label.node.time:
            connection_time = task['startTime'] - current_label.node.time
            min_connection = self._get_min_connection_time(current_label, task)
            
            # 如果连接时间足够长，可以开始新值勤日
            if connection_time >= timedelta(hours=self.MIN_REST_HOURS):
                # 足够休息，可以开始新值勤日
                return self._check_new_duty_day_constraints(current_label, task, crew)
            elif connection_time >= min_connection:
                # 连接时间足够，继续当前值勤日
                return self._check_continue_duty_day_constraints(current_label, task, crew)
            else:
                # 连接时间不足
                return False
        
        # 5. 第一个任务的检查
        return self._check_new_duty_day_constraints(current_label, task, crew)
    
    def _get_min_connection_time(self, current_label: Label, task: Dict) -> timedelta:
        """获取最小连接时间"""
        if task['type'] == 'flight':
            # 检查是否为同一架飞机
            last_task = current_label.path[-1] if current_label.path else None
            if (last_task and hasattr(last_task, 'aircraftNo') and 
                hasattr(task, 'aircraftNo') and 
                last_task.aircraftNo == task.get('aircraftNo')):
                return self.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
            else:
                return self.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT
        elif task['type'] == 'bus':
            return self.MIN_CONNECTION_TIME_BUS
        else:
            return self.DEFAULT_MIN_CONNECTION_TIME
    
    def _check_new_duty_day_constraints(self, current_label: Label, task: Dict, crew: Crew) -> bool:
        """检查开始新值勤日的约束"""
        # 1. 检查飞行周期约束
        if not self._check_flight_cycle_constraints(current_label, task, crew, is_new_duty=True):
            return False
        
        # 2. 检查工作休息模式约束
        if not self._check_work_rest_pattern(current_label, task, crew, is_new_duty=True):
            return False
        
        # 3. 检查总飞行时间约束
        if not self._check_total_flight_time_constraint(current_label, task):
            return False
        
        return True
    
    def _check_continue_duty_day_constraints(self, current_label: Label, task: Dict, crew: Crew) -> bool:
        """检查继续当前值勤日的约束"""
        # 1. 检查值勤时间限制
        if current_label.duty_start_time:
            potential_duty_end = task['endTime']
            duty_duration = (potential_duty_end - current_label.duty_start_time).total_seconds() / 3600
            if duty_duration > self.MAX_DUTY_DAY_HOURS:
                return False
        
        # 2. 检查任务数量限制
        if current_label.duty_task_count >= self.MAX_TASKS_IN_DUTY:
            return False
        
        # 3. 检查航班数量限制
        if task['type'] == 'flight' and current_label.duty_flight_count >= self.MAX_FLIGHTS_IN_DUTY:
            return False
        
        # 4. 检查值勤内飞行时间限制
        if task['type'] == 'flight':
            potential_duty_flight_time = current_label.duty_flight_time + task.get('flyTime', 0) / 60.0
            if potential_duty_flight_time > self.MAX_FLIGHT_TIME_IN_DUTY_HOURS:
                return False
        
        # 5. 检查总飞行时间约束
        if not self._check_total_flight_time_constraint(current_label, task):
            return False
        
        return True
    
    def _check_flight_cycle_constraints(self, current_label: Label, task: Dict, crew: Crew, is_new_duty: bool = False) -> bool:
        """检查飞行周期约束"""
        task_date = task['startTime'].date()
        
        # 如果任务结束在基地，飞行周期结束
        if task['arriAirport'] == crew.base:
            return True
        
        # 检查飞行周期长度限制
        if current_label.current_cycle_start is not None:
            potential_cycle_days = (task_date - current_label.current_cycle_start).days + 1
            if potential_cycle_days > self.MAX_FLIGHT_CYCLE_DAYS:
                return False
        
        # 检查周期间休息
        if (is_new_duty and current_label.last_base_return is not None and 
            current_label.current_cycle_start is None):
            days_since_base = (task_date - current_label.last_base_return).days
            if days_since_base < self.MIN_CYCLE_REST_DAYS:
                return False
        
        return True
    
    def _check_work_rest_pattern(self, current_label: Label, task: Dict, crew: Crew, is_new_duty: bool = False) -> bool:
        """检查值四修二工作模式约束"""
        if not hasattr(current_label, 'duty_days_count'):
            return True
        
        # 如果是新值勤日且已经连续工作4天
        if (is_new_duty and current_label.duty_days_count >= self.MAX_CONSECUTIVE_DUTY_DAYS and 
            task['type'] in ['flight']):
            
            # 检查是否有足够的休息时间
            if current_label.node and current_label.node.time:
                time_gap = task['startTime'] - current_label.node.time
                if time_gap.total_seconds() < 48 * 3600:  # 少于48小时休息
                    return False
        
        return True
    
    def _check_total_flight_time_constraint(self, current_label: Label, task: Dict) -> bool:
        """检查总飞行时间约束"""
        if task['type'] == 'flight':
            potential_total_flight_hours = current_label.total_flight_hours + task.get('flyTime', 0) / 60.0
            if potential_total_flight_hours > self.MAX_TOTAL_FLIGHT_HOURS:
                return False
        return True
    
    def validate_duty_day(self, duty_day: DutyDay) -> List[str]:
        """验证单个值勤日的约束"""
        violations = []
        
        # 1. 检查值勤时间限制
        if duty_day.get_duration_hours() > self.MAX_DUTY_DAY_HOURS:
            violations.append(f"值勤时间超限: {duty_day.get_duration_hours():.1f}小时 > {self.MAX_DUTY_DAY_HOURS}小时")
        
        # 2. 检查任务数量限制
        if len(duty_day.tasks) > self.MAX_TASKS_IN_DUTY:
            violations.append(f"值勤任务数超限: {len(duty_day.tasks)} > {self.MAX_TASKS_IN_DUTY}")
        
        # 3. 检查飞行任务数量限制
        flight_count = sum(1 for task in duty_day.tasks if isinstance(task, Flight))
        if flight_count > self.MAX_FLIGHTS_IN_DUTY:
            violations.append(f"值勤飞行数超限: {flight_count} > {self.MAX_FLIGHTS_IN_DUTY}")
        
        # 4. 检查值勤内飞行时间限制
        total_flight_time = sum(task.flyTime / 60.0 for task in duty_day.tasks if isinstance(task, Flight))
        if total_flight_time > self.MAX_FLIGHT_TIME_IN_DUTY_HOURS:
            violations.append(f"值勤飞行时间超限: {total_flight_time:.1f}小时 > {self.MAX_FLIGHT_TIME_IN_DUTY_HOURS}小时")
        
        # 5. 检查连接时间
        for i in range(1, len(duty_day.tasks)):
            prev_task = duty_day.tasks[i-1]
            curr_task = duty_day.tasks[i]
            
            prev_end = getattr(prev_task, 'sta', getattr(prev_task, 'endTime', None))
            curr_start = getattr(curr_task, 'std', getattr(curr_task, 'startTime', None))
            
            if prev_end and curr_start:
                connection_time = curr_start - prev_end
                min_connection = self._get_min_connection_time_for_tasks(prev_task, curr_task)
                
                if connection_time < min_connection:
                    violations.append(f"连接时间不足: {connection_time} < {min_connection}")
        
        return violations
    
    def _get_min_connection_time_for_tasks(self, prev_task, curr_task) -> timedelta:
        """获取两个任务之间的最小连接时间"""
        if isinstance(curr_task, Flight):
            if (isinstance(prev_task, Flight) and 
                hasattr(prev_task, 'aircraftNo') and hasattr(curr_task, 'aircraftNo') and 
                prev_task.aircraftNo == curr_task.aircraftNo):
                return self.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
            else:
                return self.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT
        elif isinstance(curr_task, BusInfo):
            return self.MIN_CONNECTION_TIME_BUS
        else:
            return self.DEFAULT_MIN_CONNECTION_TIME
    
    def validate_flight_cycles(self, duty_days: List[DutyDay], crew: Crew) -> List[str]:
        """验证飞行周期约束"""
        violations = []
        current_cycle_duty_days = []
        last_cycle_end_date = None
        
        for i, duty_day in enumerate(duty_days):
            # 检查是否为飞行值勤日
            if duty_day.is_flight_duty_day:
                # 如果当前没有活跃的飞行周期，开始新的飞行周期
                if not current_cycle_duty_days:
                    # 检查开始前的休息要求（2个完整日历日）
                    if last_cycle_end_date and duty_day.start_date:
                        rest_days = (duty_day.start_date - last_cycle_end_date).days
                        if rest_days < self.MIN_CYCLE_REST_DAYS:
                            violations.append(f"飞行周期间休息不足: {rest_days}天 < {self.MIN_CYCLE_REST_DAYS}天")
                    
                    current_cycle_duty_days = [duty_day]
                else:
                    # 继续当前飞行周期
                    current_cycle_duty_days.append(duty_day)
                
                # 检查是否返回基地（飞行周期结束）
                last_task = duty_day.tasks[-1] if duty_day.tasks else None
                if last_task and hasattr(last_task, 'arriAirport'):
                    if last_task.arriAirport == crew.base:
                        # 返回基地，结束当前飞行周期
                        cycle_violations = self._validate_single_flight_cycle(current_cycle_duty_days)
                        violations.extend(cycle_violations)
                        
                        last_cycle_end_date = duty_day.end_date
                        current_cycle_duty_days = []
            else:
                # 非飞行值勤日
                if current_cycle_duty_days:
                    # 检查是否可以加入当前飞行周期（少于2个完整日历日的休息）
                    last_flight_duty = current_cycle_duty_days[-1]
                    if (duty_day.start_date and last_flight_duty.end_date and
                        (duty_day.start_date - last_flight_duty.end_date).days < 2):
                        # 可以加入当前飞行周期
                        current_cycle_duty_days.append(duty_day)
                    else:
                        # 休息时间过长，当前飞行周期异常结束（末尾不是飞行值勤日）
                        violations.append("飞行周期末尾必须是飞行值勤日")
                        current_cycle_duty_days = []
        
        # 检查最后一个未完成的周期
        if current_cycle_duty_days:
            # 检查最后一个值勤日是否为飞行值勤日
            if not current_cycle_duty_days[-1].is_flight_duty_day:
                violations.append("飞行周期末尾必须是飞行值勤日")
            
            cycle_violations = self._validate_single_flight_cycle(current_cycle_duty_days)
            violations.extend(cycle_violations)
        
        return violations
    
    def _validate_single_flight_cycle(self, cycle_duty_days: List[DutyDay]) -> List[str]:
        """验证单个飞行周期的完整性"""
        violations = []
        
        if not cycle_duty_days:
            return violations
        
        # 规则1: 飞行周期必须包含飞行值勤日
        has_flight_duty_day = any(duty_day.is_flight_duty_day for duty_day in cycle_duty_days)
        if not has_flight_duty_day:
            violations.append("飞行周期必须包含飞行值勤日")
        
        # 规则2: 飞行周期末尾必须是飞行值勤日
        if not cycle_duty_days[-1].is_flight_duty_day:
            violations.append("飞行周期末尾必须是飞行值勤日")
        
        # 规则3: 飞行周期最多横跨4个日历日
        if cycle_duty_days:
            start_date = cycle_duty_days[0].start_date
            end_date = cycle_duty_days[-1].end_date
            
            if start_date and end_date:
                calendar_days_span = (end_date - start_date).days + 1
                if calendar_days_span > self.MAX_FLIGHT_CYCLE_DAYS:
                    violations.append(f"飞行周期跨度超限: {calendar_days_span}天 > {self.MAX_FLIGHT_CYCLE_DAYS}天")
        
        return violations
    
    def validate_roster_constraints(self, tasks: List, crew: Crew) -> Dict[str, List[str]]:
        """验证整个排班方案的约束"""
        result = {
            'duty_day_violations': [],
            'flight_cycle_violations': [],
            'total_violations': []
        }
        
        # 组织为值勤日
        duty_days = self.organize_tasks_into_duty_days(tasks)
        
        # 验证每个值勤日
        for i, duty_day in enumerate(duty_days):
            violations = self.validate_duty_day(duty_day)
            if violations:
                result['duty_day_violations'].extend([f"值勤日{i+1}: {v}" for v in violations])
        
        # 验证飞行周期
        cycle_violations = self.validate_flight_cycles(duty_days, crew)
        result['flight_cycle_violations'].extend(cycle_violations)
        
        # 检查总飞行时间
        total_flight_time = sum(task.flyTime / 60.0 for task in tasks if isinstance(task, Flight))
        if total_flight_time > self.MAX_TOTAL_FLIGHT_HOURS:
            result['total_violations'].append(f"总飞行时间超限: {total_flight_time:.1f}小时 > {self.MAX_TOTAL_FLIGHT_HOURS}小时")
        
        return result