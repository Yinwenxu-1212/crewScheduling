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
from unified_config import UnifiedConfig

class UnifiedConstraintChecker:
    """统一的约束检查器，确保生成和评分使用相同的逻辑"""
    
    def __init__(self, layover_stations_set: Set[str]):
        self.layover_stations_set = layover_stations_set
        
        # 从统一配置获取约束参数
        self.MAX_DUTY_DAY_HOURS = UnifiedConfig.MAX_DUTY_DAY_HOURS
        self.MIN_REST_HOURS = UnifiedConfig.MIN_REST_HOURS
        self.MAX_FLIGHTS_IN_DUTY = UnifiedConfig.MAX_FLIGHTS_IN_DUTY
        self.MAX_TASKS_IN_DUTY = UnifiedConfig.MAX_TASKS_IN_DUTY
        self.MAX_FLIGHT_TIME_IN_DUTY_HOURS = UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS
        self.MAX_TOTAL_FLIGHT_HOURS = 60.0  # 修正：总飞行值勤时间不超过60小时
        self.MAX_FLIGHT_CYCLE_DAYS = 4
        self.MIN_CYCLE_REST_DAYS = 2
        self.MAX_CONSECUTIVE_DUTY_DAYS = 4
        
        # 从统一配置获取连接时间参数
        self.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT = timedelta(minutes=UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES)
        self.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT = timedelta(hours=UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS)
        self.MIN_CONNECTION_TIME_BUS = timedelta(hours=UnifiedConfig.MIN_CONNECTION_TIME_BUS_HOURS)
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
    
    def can_assign_task_to_label(self, current_label: Label, task: Dict, crew: Crew, crew_leg_match_dict: Dict[str, List[str]] = None) -> bool:
        """
        检查是否可以将任务分配给当前标签
        使用新的DutyDay结构进行约束检查
        
        Args:
            current_label: 当前标签状态
            task: 待分配的任务
            crew: 机组信息
            crew_leg_match_dict: 机组航班资格匹配字典，格式为 {crew_id: [flight_id_list]}
        """
        # 1. 基本时间顺序检查
        if current_label.node and task['startTime'] < current_label.node.time:
            return False
        
        # 2. 地点衔接检查
        dep_airport = task.get('depAirport') or task.get('depaAirport')
        if current_label.node and current_label.node.airport != dep_airport:
            return False
        
        # 3. 资格检查（飞行任务）
        if task['type'] == 'flight':
            # 获取航班ID（处理执行和置位任务的不同命名）
            flight_id = task.get('original_flight_id')
            if not flight_id:
                # 从taskId中提取原始航班ID
                task_id = task.get('taskId', '')
                if '_exec' in task_id:
                    flight_id = task_id.replace('_exec', '')
                elif '_pos' in task_id:
                    flight_id = task_id.replace('_pos', '')
                else:
                    flight_id = task_id
            
            # 如果是置位任务，通常不需要资格检查（任何机组都可以置位）
            if task.get('subtype') == 'positioning' or task.get('is_positioning', False):
                pass  # 置位任务不需要资格检查
            else:
                # 执行任务需要资格检查
                if not flight_id:
                    return False  # 无法确定航班ID，拒绝分配
                
                # 如果提供了资格匹配字典，进行严格的资格检查
                if crew_leg_match_dict is not None:
                    eligible_flights = crew_leg_match_dict.get(crew.crewId, [])
                    if flight_id not in eligible_flights:
                        return False  # 机组没有执行该航班的资格
                # 如果没有提供资格匹配字典，假设资格检查在其他地方已完成
        
        # 3.5. 占位任务机组匹配检查
        elif (task['type'] == 'ground_duty' or task['type'] == 'groundDuty' or 
              str(task.get('taskId', '')).startswith('Grd_')):
            # 占位任务只能分配给指定的机组
            task_crew_id = task.get('crewId')
            if task_crew_id and task_crew_id != crew.crewId:
                return False  # 占位任务不属于当前机组
        
        # 4. 连接时间检查
        if current_label.node and current_label.node.time:
            connection_time = task['startTime'] - current_label.node.time
            
            # 占位任务特殊处理：不受连接时间限制，但需要时间顺序正确
            if (task['type'] == 'ground_duty' or task['type'] == 'groundDuty' or 
                str(task.get('id', '')).startswith('Grd_')):
                # 占位任务只需要保证时间顺序正确
                if connection_time >= timedelta(0):
                    # 如果连接时间足够长，可以开始新值勤日
                    if connection_time >= timedelta(hours=self.MIN_REST_HOURS):
                        return self._check_new_duty_day_constraints(current_label, task, crew)
                    else:
                        # 继续当前值勤日
                        return self._check_continue_duty_day_constraints(current_label, task, crew)
                else:
                    return False
            
            # 其他任务的正常连接时间检查
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
    
    def _is_flight_duty_day_ending_enhanced(self, current_label: Label, task: Dict, is_new_duty: bool) -> bool:
        """检查飞行周期末尾是否为飞行值勤日"""
        # 如果当前任务是飞行任务，则满足条件
        if task['type'] == 'flight':
            return True
        
        # 如果是新值勤日且包含飞行任务，也满足条件
        if is_new_duty and hasattr(current_label, 'duty_flight_count') and current_label.duty_flight_count > 0:
            return True
        
        return False
    
    def _get_cycle_actual_start_date(self, current_label: Label, task: Dict, crew: Crew = None):
        """获取飞行周期的实际开始日期"""
        # 获取机组基地
        crew_base = crew.base if crew else None
        
        # 如果有路径记录，从第一个非基地任务开始计算
        if hasattr(current_label, 'path') and current_label.path:
            for path_task in current_label.path:
                dep_airport = getattr(path_task, 'depAirport', None) or getattr(path_task, 'depaAirport', None)
                if dep_airport and crew_base and dep_airport != crew_base:
                    if hasattr(path_task, 'std'):
                        return path_task.std.date()
                    elif hasattr(path_task, 'startTime'):
                        return path_task.startTime.date()
        
        # 否则使用当前任务的开始日期
        return task['startTime'].date()
    
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
        arr_airport = task.get('arrAirport') or task.get('arriAirport')
        if arr_airport == crew.base:
            # 检查飞行周期末尾是否为飞行值勤日
            if hasattr(current_label, 'current_cycle_start') and current_label.current_cycle_start:
                if not self._is_flight_duty_day_ending_enhanced(current_label, task, is_new_duty):
                    return False
            return True
        
        # 如果任务不在基地，检查飞行周期约束
        if hasattr(current_label, 'current_cycle_start') and current_label.current_cycle_start:
            cycle_duration = (task_date - current_label.current_cycle_start).days + 1
            if cycle_duration > 4:  # 飞行周期不能超过4个日历日
                return False
        elif not hasattr(current_label, 'current_cycle_start') or not current_label.current_cycle_start:
            # 开始新的飞行周期，需要考虑置位任务和值勤占位
            if (task['type'] == 'flight' or 
                'positioning' in task.get('type', '') or 
                task['type'] == 'ground_duty'):
                # 计算实际周期开始日期
                actual_start_date = self._get_cycle_actual_start_date(current_label, task, crew)
                cycle_duration = (task_date - actual_start_date).days + 1
                if cycle_duration > 4:
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
        """检查总飞行值勤时间约束 - 修正版"""
        # 飞行值勤时间 = 飞行值勤日的总时长（从第一个任务开始到最后一个任务结束）
        # 而不是飞行时间的总和
        
        # 如果当前任务是飞行任务，需要检查飞行值勤时间
        if task['type'] == 'flight':
            # 计算当前值勤日的飞行值勤时间
            current_duty_time = 0
            if hasattr(current_label, 'duty_start_time') and current_label.duty_start_time:
                current_duty_time = (task['endTime'] - current_label.duty_start_time).total_seconds() / 3600.0
            
            # 计算总飞行值勤时间（包括当前值勤日）
            current_total_flight_duty_hours = getattr(current_label, 'total_flight_duty_hours', 0)
            potential_total_flight_duty_hours = current_total_flight_duty_hours + current_duty_time
            
            if potential_total_flight_duty_hours > self.MAX_TOTAL_FLIGHT_HOURS:
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
                if last_task:
                    last_arr_airport = None
                    if hasattr(last_task, 'arrAirport'):
                        last_arr_airport = last_task.arrAirport
                    elif hasattr(last_task, 'arriAirport'):
                        last_arr_airport = last_task.arriAirport
                    
                    if last_arr_airport == crew.base:
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
    
    def check_roster_violations(self, roster: 'Roster', crew: 'Crew') -> int:
        """
        检查排班方案的违规情况，返回违规次数
        使用新的FDP和DutyDay概念
        """
        from data_models import DutyDay, FlightDutyPeriod
        
        if not roster.duties:
            return 0
        
        violations = 0
        
        # 按时间排序任务
        sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        # 组织任务为值勤日
        duty_days = self._organize_into_duty_days(sorted_duties)
        
        # 检查每个值勤日和其中的FDP
        total_flight_duty_time = 0.0
        all_fdps = []
        
        for duty_day in duty_days:
            # 组织值勤日中的FDP
            duty_day.organize_into_fdps()
            
            # 检查每个FDP的违规情况
            for fdp in duty_day.fdps:
                if fdp.is_valid():
                    violations += fdp.violates_constraints()
                    total_flight_duty_time += fdp.duty_time / 60.0  # 转换为小时
                    all_fdps.append(fdp)
        
        # 检查飞行周期违规
        cycle_violations = self._check_flight_cycle_violations_new(duty_days, crew)
        violations += cycle_violations
        
        # 检查总飞行值勤时间限制（规则9）
        if total_flight_duty_time > 60:  # 60小时限制
            violations += 1
        
        # 检查FDP间休息时间违规
        rest_violations = self._check_fdp_rest_violations(all_fdps)
        violations += rest_violations
        
        # 检查值四修二工作模式违规
        work_pattern_violations = self._check_work_rest_pattern_violations(sorted_duties, crew)
        violations += work_pattern_violations
        
        # 检查地点衔接规则违规
        location_violations = self._check_location_connection_violations(sorted_duties, crew)
        violations += location_violations
        
        return violations
    
    def _organize_into_duty_days(self, sorted_duties: List) -> List['DutyDay']:
        """
        将排序后的任务组织为值勤日
        """
        from data_models import DutyDay
        
        if not sorted_duties:
            return []
        
        duty_days = []
        current_duty_day = DutyDay()
        
        for i, duty in enumerate(sorted_duties):
            duty_start = getattr(duty, 'std', getattr(duty, 'startTime', None))
            
            if not duty_start:
                continue
            
            # 如果是第一个任务，直接加入当前值勤日
            if i == 0:
                current_duty_day.add_task(duty)
                continue
            
            # 检查与前一个任务的时间间隔
            prev_duty = sorted_duties[i-1]
            prev_end = getattr(prev_duty, 'sta', getattr(prev_duty, 'endTime', None))
            
            if prev_end and duty_start:
                rest_interval = duty_start - prev_end
                
                # 判断是否需要开始新的值勤日
                should_start_new_duty_day = False
                
                # 1. 休息时间超过12小时
                if rest_interval >= timedelta(hours=self.MIN_REST_HOURS):
                    should_start_new_duty_day = True
                    
                # 2. 当前值勤日已经超过24小时
                elif current_duty_day.start_time and (duty_start - current_duty_day.start_time) > timedelta(hours=24):
                    should_start_new_duty_day = True
                    
                # 3. 跨越了太多日历日（超过2个日历日）
                elif (current_duty_day.start_time and 
                      (duty_start.date() - current_duty_day.start_time.date()).days > 1):
                    should_start_new_duty_day = True
                
                if should_start_new_duty_day:
                    # 结束当前值勤日，开始新的值勤日
                    if current_duty_day.tasks:
                        duty_days.append(current_duty_day)
                    current_duty_day = DutyDay()
            
            current_duty_day.add_task(duty)
        
        # 添加最后一个值勤日
        if current_duty_day.tasks:
            duty_days.append(current_duty_day)
        
        # 更新所有值勤日的飞行值勤日状态
        for duty_day in duty_days:
            duty_day.set_layover_stations(self.layover_stations_set)
        
        return duty_days
    
    def _check_flight_cycle_violations_new(self, duty_days: List['DutyDay'], crew: 'Crew') -> int:
        """
        检查飞行周期违规情况
        """
        violations = 0
        current_cycle_duty_days = []
        last_cycle_end_date = None
        
        for duty_day in duty_days:
            # 检查是否为飞行值勤日
            if duty_day.is_flight_duty_day:
                # 如果当前没有活跃的飞行周期，开始新的飞行周期
                if not current_cycle_duty_days:
                    # 检查开始前的休息要求（2个完整日历日）
                    if last_cycle_end_date and duty_day.start_date:
                        rest_days = (duty_day.start_date - last_cycle_end_date).days
                        if rest_days < self.MIN_CYCLE_REST_DAYS:
                            violations += 1
                    
                    current_cycle_duty_days = [duty_day]
                else:
                    # 继续当前飞行周期
                    current_cycle_duty_days.append(duty_day)
                
                # 检查是否返回基地（飞行周期结束）
                last_task = duty_day.tasks[-1] if duty_day.tasks else None
                if last_task:
                    last_arr_airport = None
                    if hasattr(last_task, 'arrAirport'):
                        last_arr_airport = last_task.arrAirport
                    elif hasattr(last_task, 'arriAirport'):
                        last_arr_airport = last_task.arriAirport
                    
                    if last_arr_airport == crew.base:
                        # 返回基地，结束当前飞行周期
                        cycle_violations = self._validate_single_flight_cycle_violations(current_cycle_duty_days)
                        violations += cycle_violations
                        
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
                        violations += 1
                        current_cycle_duty_days = []
        
        # 检查最后一个未完成的周期
        if current_cycle_duty_days:
            # 检查最后一个值勤日是否为飞行值勤日
            if not current_cycle_duty_days[-1].is_flight_duty_day:
                violations += 1
            
            cycle_violations = self._validate_single_flight_cycle_violations(current_cycle_duty_days)
            violations += cycle_violations
        
        return violations
    
    def _validate_single_flight_cycle_violations(self, cycle_duty_days: List['DutyDay']) -> int:
        """
        验证单个飞行周期的违规情况
        """
        violations = 0
        
        if not cycle_duty_days:
            return violations
        
        # 规则1: 飞行周期必须包含飞行值勤日
        has_flight_duty_day = any(duty_day.is_flight_duty_day for duty_day in cycle_duty_days)
        if not has_flight_duty_day:
            violations += 1
        
        # 规则2: 飞行周期末尾必须是飞行值勤日
        if not cycle_duty_days[-1].is_flight_duty_day:
            violations += 1
        
        # 规则3: 飞行周期最多横跨4个日历日
        if cycle_duty_days:
            start_date = cycle_duty_days[0].start_date
            end_date = cycle_duty_days[-1].end_date
            
            if start_date and end_date:
                calendar_days_span = (end_date - start_date).days + 1
                if calendar_days_span > self.MAX_FLIGHT_CYCLE_DAYS:
                    violations += 1
        
        return violations
    
    def _check_fdp_rest_violations(self, all_fdps: List['FlightDutyPeriod']) -> int:
        """
        检查FDP间休息时间违规
        """
        violations = 0
        
        for i in range(1, len(all_fdps)):
            prev_fdp = all_fdps[i-1]
            curr_fdp = all_fdps[i]
            
            if prev_fdp.end_time and curr_fdp.start_time:
                rest_time = curr_fdp.start_time - prev_fdp.end_time
                if rest_time < timedelta(hours=self.MIN_REST_HOURS):
                    violations += 1
        
        return violations
    
    def _check_work_rest_pattern_violations(self, sorted_duties: List, crew: 'Crew') -> int:
        """
        检查值四修二工作模式违规
        """
        violations = 0
        
        # 统计连续工作天数
        consecutive_work_days = 0
        last_duty_date = None
        
        for duty in sorted_duties:
            duty_start = getattr(duty, 'std', getattr(duty, 'startTime', None))
            if not duty_start:
                continue
            
            duty_date = duty_start.date()
            
            if last_duty_date is None or (duty_date - last_duty_date).days <= 1:
                consecutive_work_days += 1
            else:
                consecutive_work_days = 1
            
            if consecutive_work_days > self.MAX_CONSECUTIVE_DUTY_DAYS:
                violations += 1
            
            last_duty_date = duty_date
        
        return violations
    
    def _check_location_connection_violations(self, sorted_duties: List, crew: 'Crew') -> int:
        """
        检查地点衔接规则违规
        """
        violations = 0
        
        for i in range(1, len(sorted_duties)):
            prev_duty = sorted_duties[i-1]
            curr_duty = sorted_duties[i]
            
            prev_end_airport = getattr(prev_duty, 'arrAirport', None) or getattr(prev_duty, 'arriAirport', None)
            curr_start_airport = getattr(curr_duty, 'depAirport', None) or getattr(curr_duty, 'depaAirport', None)
            
            if prev_end_airport and curr_start_airport:
                if prev_end_airport != curr_start_airport:
                    violations += 1
        
        return violations