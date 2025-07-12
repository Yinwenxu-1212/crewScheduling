#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构后的约束检查模块
提供清晰的约束管理和验证功能
"""

from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import logging
from data_models import Flight, BusInfo, GroundDuty, Crew, DutyDay, FlightDutyPeriod, FlightCycle, Label
from unified_config import UnifiedConfig

class ConstraintManager:
    """约束管理器 - 集中管理所有约束规则"""
    
    def __init__(self):
        # 值勤日约束
        self.MAX_DUTY_DAY_HOURS = UnifiedConfig.MAX_DUTY_DAY_HOURS
        self.MAX_TASKS_IN_DUTY = UnifiedConfig.MAX_TASKS_IN_DUTY
        self.MAX_FLIGHTS_IN_DUTY = UnifiedConfig.MAX_FLIGHTS_IN_DUTY
        self.MAX_FLIGHT_TIME_IN_DUTY_HOURS = UnifiedConfig.MAX_FLIGHT_TIME_IN_DUTY_HOURS
        
        # 飞行值勤日约束
        self.MAX_FDP_HOURS = UnifiedConfig.MAX_FDP_HOURS
        self.MAX_FDP_FLIGHTS = UnifiedConfig.MAX_FDP_FLIGHTS
        self.MAX_FDP_TASKS = UnifiedConfig.MAX_FDP_TASKS
        self.MAX_FDP_FLIGHT_TIME = UnifiedConfig.MAX_FDP_FLIGHT_TIME
        
        # 飞行周期约束
        self.MAX_FLIGHT_CYCLE_DAYS = UnifiedConfig.MAX_FLIGHT_CYCLE_DAYS
        self.MIN_CYCLE_REST_DAYS = UnifiedConfig.MIN_CYCLE_REST_DAYS
        self.MAX_TOTAL_FLIGHT_DUTY_HOURS = UnifiedConfig.MAX_TOTAL_FLIGHT_DUTY_HOURS
        
        # 休息时间约束
        self.MIN_REST_HOURS = UnifiedConfig.MIN_REST_HOURS
        self.MIN_OVERNIGHT_HOURS = UnifiedConfig.MIN_OVERNIGHT_HOURS
        
        # 连接时间约束
        self.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT = timedelta(minutes=UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES)
        self.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT = timedelta(hours=UnifiedConfig.MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS)
        self.MIN_CONNECTION_TIME_BUS = timedelta(hours=UnifiedConfig.MIN_CONNECTION_TIME_BUS_HOURS)
        self.DEFAULT_MIN_CONNECTION_TIME = timedelta(hours=UnifiedConfig.DEFAULT_MIN_CONNECTION_TIME_HOURS)
        
        # 工作休息模式约束
        self.MAX_CONSECUTIVE_DUTY_DAYS = UnifiedConfig.MAX_CONSECUTIVE_DUTY_DAYS
        self.MIN_REST_AFTER_CONSECUTIVE_DUTY = UnifiedConfig.MIN_REST_AFTER_CONSECUTIVE_DUTY
        
        # 置位任务约束
        self.MAX_POSITIONING_TASKS = UnifiedConfig.MAX_POSITIONING_TASKS
        
        # 新飞行周期约束
        self.MIN_REST_DAYS_FOR_NEW_CYCLE = UnifiedConfig.MIN_REST_DAYS_FOR_NEW_CYCLE
        
        # 总飞行时间约束
        self.MAX_TOTAL_FLIGHT_HOURS = UnifiedConfig.MAX_TOTAL_FLIGHT_HOURS

class UnifiedConstraintChecker:
    """统一的约束检查器"""
    
    def __init__(self, layover_stations_set: Set[str]):
        self.layover_stations_set = layover_stations_set
        self.constraints = ConstraintManager()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.ERROR)  # 抑制WARNING输出，只显示ERROR级别
        
    def organize_tasks_into_duty_days(self, tasks: List) -> tuple[List[DutyDay], List]:
        """
        将任务组织为值勤日
        
        值勤日定义：
        - 任务的集合，跨度不超过24小时
        - 可包含占位任务，可跨日历日
        - 任务间休息时间少于12小时则属于同一值勤日
        """
        if not tasks:
            return [], []
            
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
                
            # 检查是否需要开始新的值勤日
            if self._should_start_new_duty_day(current_day, task, sorted_tasks[i-1]):
                # 结束当前值勤日，开始新的值勤日
                if current_day.tasks:
                    duty_days.append(current_day)
                current_day = DutyDay()
            
            current_day.add_task(task)
        
        # 添加最后一个值勤日
        if current_day.tasks:
            duty_days.append(current_day)
        return duty_days, sorted_tasks
    
    def _should_start_new_duty_day(self, current_day: DutyDay, task, prev_task) -> bool:
        """判断是否应该开始新的值勤日"""
        task_start = getattr(task, 'std', getattr(task, 'startTime', None))
        prev_end = getattr(prev_task, 'sta', getattr(prev_task, 'endTime', None))
        
        if not task_start or not prev_end:
            return False
            
        rest_interval = task_start - prev_end
        
        # 规则1: 休息时间超过12小时
        if rest_interval >= timedelta(hours=self.constraints.MIN_REST_HOURS):
            return True
            
        # 规则2: 当前值勤日已经超过24小时
        if current_day.start_time:
            potential_duration = (task_start - current_day.start_time).total_seconds() / 3600
            if potential_duration > self.constraints.MAX_DUTY_DAY_HOURS:
                return True
                
        # 规则3: 跨越了太多日历日（超过2个日历日）
        if (current_day.start_time and 
            (task_start.date() - current_day.start_time.date()).days > 1):
            return True
        
        return False
    
    def organize_flight_duty_periods(self, duty_days: List[DutyDay]) -> List[FlightDutyPeriod]:
        """
        从值勤日中提取飞行值勤日
        
        飞行值勤日定义：
        - 必须包含飞行任务
        - 只能从可过夜机场出发到可过夜机场结束
        - 是值勤日的一个特殊子类型
        """
        flight_duty_periods = []
        
        for duty_day in duty_days:
            if duty_day.has_flight_tasks():
                # 创建飞行值勤日
                fdp = FlightDutyPeriod()
                
                # 添加所有任务到FDP
                for task in duty_day.tasks:
                    fdp.add_task(task)
                
                # 验证FDP是否有效（从可过夜机场出发到可过夜机场结束）
                if fdp.is_valid(self.layover_stations_set):
                    flight_duty_periods.append(fdp)
                else:
                    self.logger.warning(f"飞行值勤日无效：未从可过夜机场出发或结束")
        
        return flight_duty_periods
    
    def organize_flight_cycles(self, duty_days: List[DutyDay], crew: Crew) -> List[FlightCycle]:
        """
        将值勤日组织为飞行周期
        
        飞行周期定义：
        - 由值勤日组成，必须包含飞行值勤日
        - 末尾必须是飞行值勤日
        - 最多横跨4个日历日
        - 开始前需连续休息2个完整日历日
        """
        flight_cycles = []
        current_cycle = None
        last_cycle_end_date = None
        
        for duty_day in duty_days:
            # 检查是否为飞行值勤日
            if duty_day.has_flight_tasks():
                # 如果当前没有活跃的飞行周期，开始新的飞行周期
                if current_cycle is None:
                    # 检查开始前的休息要求
                    if self._can_start_new_flight_cycle(duty_day, last_cycle_end_date):
                        current_cycle = FlightCycle()
                        current_cycle.add_duty_day(duty_day)
                    else:
                        self.logger.warning(f"飞行周期开始前休息不足")
                        continue
                else:
                    # 继续当前飞行周期
                    current_cycle.add_duty_day(duty_day)
                
                # 检查是否返回基地（飞行周期结束）
                if self._cycle_ends_at_base(duty_day, crew.base):
                    # 返回基地，结束当前飞行周期
                    flight_cycles.append(current_cycle)
                    last_cycle_end_date = duty_day.end_date
                    current_cycle = None
            else:
                # 非飞行值勤日
                if current_cycle is not None:
                    # 检查是否可以加入当前飞行周期
                    if self._can_add_to_current_cycle(current_cycle, duty_day):
                        current_cycle.add_duty_day(duty_day)
                    else:
                        # 休息时间过长，当前飞行周期异常结束
                        self.logger.warning(f"飞行周期异常结束：末尾不是飞行值勤日")
                        current_cycle = None
        
        # 检查最后一个未完成的周期
        if current_cycle is not None:
            if current_cycle.ends_with_flight_duty_period():
                flight_cycles.append(current_cycle)
            else:
                self.logger.warning(f"飞行周期末尾必须是飞行值勤日")
        
        return flight_cycles
    
    def _can_start_new_flight_cycle(self, duty_day: DutyDay, last_cycle_end_date) -> bool:
        """检查是否可以开始新的飞行周期"""
        if last_cycle_end_date is None:
            return True  # 第一个飞行周期
            
        if duty_day.start_date:
            rest_days = (duty_day.start_date - last_cycle_end_date).days
            return rest_days >= self.constraints.MIN_CYCLE_REST_DAYS
            
        return True
    
    def _cycle_ends_at_base(self, duty_day: DutyDay, crew_base: str) -> bool:
        """检查飞行周期是否在基地结束"""
        if not duty_day.tasks:
            return False
            
        last_task = duty_day.tasks[-1]
        if isinstance(last_task, Flight):
            return getattr(last_task, 'arrAirport', None) == crew_base or getattr(last_task, 'arriAirport', None) == crew_base
            
        return False
    
    def _can_add_to_current_cycle(self, current_cycle: FlightCycle, duty_day: DutyDay) -> bool:
        """检查是否可以将值勤日加入当前飞行周期"""
        if not current_cycle.duty_days:
            return False
            
        last_duty_day = current_cycle.duty_days[-1]
        if duty_day.start_date and last_duty_day.end_date:
            rest_days = (duty_day.start_date - last_duty_day.end_date).days
            return rest_days < 2  # 少于2个完整日历日的休息可以加入
            
        return result
    
    def _validate_inconsistent_locations(self, crew: Crew, sorted_tasks: List) -> List[str]:
        """验证地点不衔接情况"""
        violations = []
        
        if not sorted_tasks or len(sorted_tasks) < 2:
            return violations
        
        for i in range(len(sorted_tasks) - 1):
            current_task = sorted_tasks[i]
            next_task = sorted_tasks[i + 1]
            
            # 获取当前任务的结束地点和下一个任务的开始地点
            current_end_location = getattr(current_task, 'arr', None)
            next_start_location = getattr(next_task, 'dep', None)
            
            # 如果地点不衔接
            if (current_end_location and next_start_location and 
                current_end_location != next_start_location):
                violations.append(
                    f"任务{i+1}结束地点({current_end_location})与任务{i+2}开始地点({next_start_location})不衔接"
                )
        
        return violations
    
    def _organize_duties_into_duty_days(self, sorted_duties: List) -> List[List]:
        """将任务组织为值勤日"""
        if not sorted_duties:
            return []
        
        duty_days = []
        current_duty = [sorted_duties[0]]
        
        for i in range(1, len(sorted_duties)):
            task = sorted_duties[i]
            prev_task = sorted_duties[i-1]
            
            prev_end = self._get_task_end_time(prev_task)
            task_start = self._get_task_start_time(task)
            
            if prev_end and task_start:
                rest_time = task_start - prev_end
                
                # 判断是否开始新值勤日（休息时间>=12小时或跨度>24小时）
                if (rest_time >= timedelta(hours=12) or
                    (task_start - self._get_task_start_time(current_duty[0])) > timedelta(hours=24)):
                    
                    # 结束当前值勤日
                    duty_days.append(current_duty)
                    
                    # 开始新值勤日
                    current_duty = [task]
                else:
                    # 继续当前值勤日
                    current_duty.append(task)
            else:
                # 无法获取时间信息，继续当前值勤日
                current_duty.append(task)
        
        # 添加最后一个值勤日
        if current_duty:
            duty_days.append(current_duty)
        
        return duty_days
    
    def _is_flight_task(self, task) -> bool:
        """判断是否为飞行任务"""
        if isinstance(task, dict):
            return task.get('type') == 'flight'
        else:
            return getattr(task, 'type', None) == 'flight' or hasattr(task, 'flightNo')
    
    def _get_task_start_time(self, task):
        """获取任务开始时间"""
        if isinstance(task, dict):
            return task.get('startTime') or task.get('std')
        else:
            return getattr(task, 'startTime', None) or getattr(task, 'std', None)
    
    def _get_task_end_time(self, task):
        """获取任务结束时间"""
        if isinstance(task, dict):
            return task.get('endTime') or task.get('sta')
        else:
            return getattr(task, 'endTime', None) or getattr(task, 'sta', None)
    
    def validate_duty_day(self, duty_day: DutyDay) -> List[str]:
        """验证值勤日约束"""
        violations = []
        
        # 1. 检查值勤时间限制（24小时）con6
        if duty_day.violates_24_hour_constraint():
            violations.append(f"值勤时间超限: {duty_day.get_duration_hours():.1f}小时 > {self.constraints.MAX_DUTY_DAY_HOURS}小时")
        
        # 2. 检查任务数量限制 con7
        if len(duty_day.tasks) > self.constraints.MAX_TASKS_IN_DUTY:
            violations.append(f"值勤任务数超限: {len(duty_day.tasks)} > {self.constraints.MAX_TASKS_IN_DUTY}")
        
        # 3. 检查飞行任务数量限制 con8
        flight_count = sum(1 for task in duty_day.tasks if isinstance(task, Flight))
        if flight_count > self.constraints.MAX_FLIGHTS_IN_DUTY:
            violations.append(f"值勤飞行数超限: {flight_count} > {self.constraints.MAX_FLIGHTS_IN_DUTY}")
        
        # 4. 检查值勤内飞行时间限制
        total_flight_time = sum(getattr(task, 'flyTime', 0) / 60.0 for task in duty_day.tasks if isinstance(task, Flight) and not getattr(task, 'is_positioning', False))
        if total_flight_time > self.constraints.MAX_FLIGHT_TIME_IN_DUTY_HOURS:
            violations.append(f"值勤飞行时间超限: {total_flight_time:.1f}小时 > {self.constraints.MAX_FLIGHT_TIME_IN_DUTY_HOURS}小时")
        
        # 5. 检查连接时间
        violations.extend(self._validate_connection_times(duty_day.tasks))
        
        return violations
     
    def validate_flight_duty_period(self, fdp: FlightDutyPeriod) -> List[str]:
        """验证飞行值勤日约束"""
        violations = []
        
        # 1. 检查是否包含执行飞行任务 con1
        if not fdp.has_flight:
            violations.append("飞行值勤日必须包含执行飞行任务")
        
        # 2. 检查飞行值勤日时间限制（12小时）con6
        # 根据规则6：飞行值勤开始时间为第一个任务的开始时间，飞行值勤结束时间为最后一个飞行任务的到达时间
        if fdp.get_flight_duty_duration_hours() > self.constraints.MAX_FDP_HOURS:
            violations.append(f"飞行值勤日时间超限: {fdp.get_flight_duty_duration_hours():.1f}小时 > {self.constraints.MAX_FDP_HOURS}小时")
        
        # 3. 检查飞行任务数量限制 con7
        if fdp.get_flight_count() > self.constraints.MAX_FDP_FLIGHTS:
            violations.append(f"飞行值勤日飞行数超限: {fdp.get_flight_count()} > {self.constraints.MAX_FDP_FLIGHTS}")
        
        # 4. 检查总任务数量限制 con8
        if len(fdp.tasks) > self.constraints.MAX_FDP_TASKS:
            violations.append(f"飞行值勤日任务数超限: {len(fdp.tasks)} > {self.constraints.MAX_FDP_TASKS}")
        
        # 5. 检查飞行时间限制 con9
        flight_time_hours = fdp.get_total_flight_time_minutes() / 60.0
        if flight_time_hours > self.constraints.MAX_FDP_FLIGHT_TIME:
            violations.append(f"飞行值勤日飞行时间超限: {flight_time_hours:.1f}小时 > {self.constraints.MAX_FDP_FLIGHT_TIME}小时")
        
        # 6. 检查可过夜机场约束
        if not fdp.is_valid(self.layover_stations_set):
            violations.append("飞行值勤日必须从可过夜机场出发到可过夜机场结束")
        
        # 7. 检查连接时间约束
        connection_violations = self._validate_connection_times(fdp.tasks)
        violations.extend(connection_violations)
        
        # 8. 检查置位规则（规则1：仅允许值勤日的开始或结束进行置位）
        positioning_violations = self._validate_positioning_rules(fdp.tasks)
        violations.extend(positioning_violations)
        
        return violations
     
    def validate_flight_cycle(self, cycle: FlightCycle, crew: Crew = None) -> List[str]:
        """验证飞行周期约束"""
        violations = []
        
        # 1. 检查是否包含飞行值勤日
        if not cycle.has_flight_duty_periods():
            violations.append("飞行周期必须包含飞行值勤日")
        
        # 2. 检查末尾是否为飞行值勤日
        if not cycle.ends_with_flight_duty_period():
            violations.append("飞行周期末尾必须是飞行值勤日")
        
        # 3. 检查日历日跨度限制 con12
        if cycle.get_calendar_days_span() > self.constraints.MAX_FLIGHT_CYCLE_DAYS:
            violations.append(f"飞行周期跨度超限: {cycle.get_calendar_days_span()}天 > {self.constraints.MAX_FLIGHT_CYCLE_DAYS}天")
        
        # 4. 检查是否在基地开始（如果有基地信息）
        if crew and crew.base and not cycle.starts_at_base(crew.base):
            violations.append("飞行周期应在基地开始")
        
        # 5. 检查是否返回基地结束（如果有基地信息）
        if crew and crew.base and not cycle.ends_at_base(crew.base):
            violations.append("飞行周期应返回基地结束")
        
        return violations
     
    def _validate_connection_times(self, tasks: List) -> List[str]:
        """验证任务间连接时间"""
        violations = []
        
        for i in range(1, len(tasks)):
            prev_task = tasks[i-1]
            curr_task = tasks[i]
            
            prev_end = getattr(prev_task, 'sta', getattr(prev_task, 'endTime', None))
            curr_start = getattr(curr_task, 'std', getattr(curr_task, 'startTime', None))
            
            if prev_end and curr_start:
                connection_time = curr_start - prev_end
                min_connection = self._get_min_connection_time_for_tasks(prev_task, curr_task)
                
                if connection_time < min_connection:
                    violations.append(f"连接时间不足: {connection_time} < {min_connection}")
        
        return violations
    
    def _validate_positioning_rules(self, tasks: List) -> List[str]:
        """验证置位规则：仅允许值勤日的开始或结束进行置位"""
        violations = []
        
        if len(tasks) <= 2:
            return violations  # 如果任务数量少于等于2个，不需要检查中间位置
        
        # 检查中间位置的任务是否有置位任务
        for i in range(1, len(tasks) - 1):  # 排除第一个和最后一个任务
            task = tasks[i]
            
            # 检查是否为置位任务
            is_positioning = False
            
            # 方法1：检查任务类型和子类型
            if hasattr(task, 'subtype') and task.subtype == 'positioning':
                is_positioning = True
            elif hasattr(task, 'is_positioning') and task.is_positioning:
                is_positioning = True
            elif hasattr(task, 'positioning_flight') and task.positioning_flight:
                is_positioning = True
            
            # 方法2：检查任务ID中是否包含置位标识
            task_id = getattr(task, 'taskId', '') or getattr(task, 'id', '')
            if '_pos' in str(task_id).lower() or 'positioning' in str(task_id).lower():
                is_positioning = True
            
            # 方法3：检查任务类型是否为大巴置位
            task_type = getattr(task, 'type', '')
            if 'bus' in str(task_type).lower() and ('positioning' in str(task_type).lower() or 'pos' in str(task_type).lower()):
                is_positioning = True
            
            if is_positioning:
                violations.append(f"置位任务只能在值勤日开始或结束位置，不能在中间位置（任务{i+1}）")
        
        return violations
     

    def validate_roster_constraints(self, tasks: List, crew: Crew) -> Dict[str, List[str]]:
        """验证整个排班方案的约束"""
        result = {
            'duty_day_violations': [],
            'flight_duty_period_violations': [],
            'flight_cycle_violations': [],
            'total_flight_time_violations': [],
            'inconsistent_location_violations': [],
        }
        
        # 1. 组织任务为值勤日
        duty_days, sorted_tasks = self.organize_tasks_into_duty_days(tasks)
        
        # 2. 验证每个值勤日 6,7,8 # 需要添加判断：非飞行执勤日
        for i, duty_day in enumerate(duty_days):
            violations = self.validate_duty_day(duty_day)
            if violations:
                result['duty_day_violations'].extend([f"值勤日{i+1}: {v}" for v in violations])
        
        # 3. 组织和验证飞行值勤日 1,(6,7,8),9
        flight_duty_periods = self.organize_flight_duty_periods(duty_days)
        for i, fdp in enumerate(flight_duty_periods):
            violations = self.validate_flight_duty_period(fdp)
            if violations:
                result['flight_duty_period_violations'].extend([f"飞行值勤日{i+1}: {v}" for v in violations])
         
        # 4. 组织和验证飞行周期 12
        flight_cycles = self.organize_flight_cycles(duty_days, crew)
        for i, cycle in enumerate(flight_cycles):
            violations = self.validate_flight_cycle(cycle, crew)
            if violations:
                result['flight_cycle_violations'].extend([f"飞行周期{i+1}: {v}" for v in violations])
        
        # 5. 验证总飞行值勤时间 con10
        total_flight_duty_time = sum(fdp.get_duration_hours() for fdp in flight_duty_periods)  # 使用方法获取小时数
        if total_flight_duty_time > self.constraints.MAX_TOTAL_FLIGHT_DUTY_HOURS:
            result['total_flight_time_violations'].append(
                f"总飞行值勤时间超限: {total_flight_duty_time:.1f}小时 > {self.constraints.MAX_TOTAL_FLIGHT_DUTY_HOURS}小时"
            )

        # 6. 不衔接扣分 con2
        inconsistent_violations = self._validate_inconsistent_locations(crew, sorted_tasks)
        if inconsistent_violations:
            result['inconsistent_location_violations'].extend([f"地点未衔接: {v}" for v in inconsistent_violations])
        
        return result
     
    def can_assign_task_to_label(self, current_label: Label, task: Dict, crew: Crew, crew_leg_match_dict: Dict[str, List[str]] = None) -> bool:
        """
        检查是否可以将任务分配给当前标签
        使用新的约束检查逻辑
        
        Args:
            current_label: 当前标签状态
            task: 待分配的任务
            crew: 机组信息
            crew_leg_match_dict: 机组航班资格匹配字典，格式为 {crew_id: [flight_id_list]}
        """
        # 1. 时间顺序检查
        task_start_time = task.get('startTime') if isinstance(task, dict) else getattr(task, 'startTime', None)
        if current_label.node and task_start_time < current_label.node.time:
            return False
        
        # 2. 地点衔接检查
        if isinstance(task, dict):
            dep_airport = task.get('depAirport') or task.get('depaAirport')
        else:
            # 处理GroundDuty对象
            dep_airport = getattr(task, 'depAirport', None) or getattr(task, 'depaAirport', None) or getattr(task, 'airport', None)
        if current_label.node and current_label.node.airport != dep_airport:
            return False
        
        # 3. 资格检查（飞行任务）
        task_type = task.get('type') if isinstance(task, dict) else getattr(task, 'type', None)
        if task_type == 'flight':
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
        elif (task_type == 'ground_duty' or task_type == 'groundDuty' or 
              str(task.get('taskId', '') if isinstance(task, dict) else getattr(task, 'id', '')).startswith('Grd_')):
            # 占位任务只能分配给指定的机组
            task_crew_id = task.get('crewId') if isinstance(task, dict) else getattr(task, 'crewId', None)
            if task_crew_id and task_crew_id != crew.crewId:
                return False  # 占位任务不属于当前机组
        
        # 4. 任务重叠检查
        if hasattr(current_label, 'path') and current_label.path:
            for existing_task in current_label.path:
                # 检查时间重叠
                existing_start = getattr(existing_task, 'startTime', None)
                existing_end = getattr(existing_task, 'endTime', None)
                new_start = task.get('startTime') if isinstance(task, dict) else getattr(task, 'startTime', None)
                new_end = task.get('endTime') if isinstance(task, dict) else getattr(task, 'endTime', None)
                
                if existing_start and existing_end:
                    # 判断是否重叠：新任务开始时间 < 已有任务结束时间 且 新任务结束时间 > 已有任务开始时间
                    is_overlapping = new_start < existing_end and new_end > existing_start
                    
                    if is_overlapping:
                        # 根据规则11：占位任务与占位任务之间可以重叠，其他任务类型两两之间不能重叠
                        is_new_ground = task_type == 'ground_duty'
                        is_existing_ground = getattr(existing_task, 'type', None) == 'ground_duty'
                        
                        # 如果两个都是占位任务，允许重叠
                        if is_new_ground and is_existing_ground:
                            continue
                        
                        # 其他情况不允许重叠
                        return False
        
        # 5. 连接时间检查
        if current_label.node and current_label.node.time:
            connection_time = task_start_time - current_label.node.time
            
            # 占位任务特殊处理：不受连接时间限制，但需要时间顺序正确
            if (task_type == 'ground_duty' or task_type == 'groundDuty' or 
                str(task.get('id', '') if isinstance(task, dict) else getattr(task, 'id', '')).startswith('Grd_')):
                # 占位任务只需要保证时间顺序正确
                if connection_time >= timedelta(0):
                    # 如果连接时间足够长，可以开始新值勤日
                    if connection_time >= timedelta(hours=self.constraints.MIN_REST_HOURS):
                        return self._check_new_duty_day_constraints(current_label, task, crew)
                    else:
                        # 继续当前值勤日
                        return self._check_continue_duty_day_constraints(current_label, task, crew)
                else:
                    return False
            
            # 其他任务的正常连接时间检查
            min_connection = self._get_min_connection_time(current_label, task)
            
            # 如果连接时间足够长，可以开始新值勤日
            if connection_time >= timedelta(hours=self.constraints.MIN_REST_HOURS):
                # 足够休息，可以开始新值勤日
                return self._check_new_duty_day_constraints(current_label, task, crew)
            elif connection_time >= min_connection:
                # 连接时间足够，继续当前值勤日
                return self._check_continue_duty_day_constraints(current_label, task, crew)
            else:
                # 连接时间不足
                return False
        
        # 6. 第一个任务的检查
        return self._check_new_duty_day_constraints(current_label, task, crew)
    
    def _get_min_connection_time(self, current_label: Label, task: Dict) -> timedelta:
        """获取最小连接时间"""
        if task['type'] == 'flight':
            # 检查是否为同一架飞机
            last_task = current_label.path[-1] if current_label.path else None
            if (last_task and hasattr(last_task, 'aircraftNo') and 
                hasattr(task, 'aircraftNo') and 
                last_task.aircraftNo == task.get('aircraftNo')):
                return self.constraints.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
            else:
                return self.constraints.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT
        elif task['type'] == 'bus':
            return self.constraints.MIN_CONNECTION_TIME_BUS
        else:
            return self.constraints.DEFAULT_MIN_CONNECTION_TIME
    
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
        task_type = task.get('type') if isinstance(task, dict) else getattr(task, 'type', None)
        if task_type == 'flight':
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
            potential_duty_end = task.get('endTime') if isinstance(task, dict) else getattr(task, 'endTime', None)
            duty_duration = (potential_duty_end - current_label.duty_start_time).total_seconds() / 3600
            if duty_duration > self.constraints.MAX_DUTY_DAY_HOURS:
                return False
        
        # 2. 检查任务数量限制
        if current_label.duty_task_count >= self.constraints.MAX_TASKS_IN_DUTY:
            return False
        
        # 3. 检查航班数量限制
        task_type = task.get('type') if isinstance(task, dict) else getattr(task, 'type', None)
        if task_type == 'flight' and current_label.duty_flight_count >= self.constraints.MAX_FLIGHTS_IN_DUTY:
            return False
        
        # 4. 检查值勤内飞行时间限制
        if task_type == 'flight':
            fly_time = task.get('flyTime', 0) if isinstance(task, dict) else getattr(task, 'flyTime', 0)
            potential_duty_flight_time = current_label.duty_flight_time + fly_time / 60.0
            if potential_duty_flight_time > self.constraints.MAX_FLIGHT_TIME_IN_DUTY_HOURS:
                return False
        
        # 5. 检查总飞行时间约束
        if not self._check_total_flight_time_constraint(current_label, task):
            return False
        
        return True
    
    def _check_flight_cycle_constraints(self, current_label: Label, task: Dict, crew: Crew, is_new_duty: bool = False) -> bool:
        """检查飞行周期约束"""
        task_start_time = task.get('startTime') if isinstance(task, dict) else getattr(task, 'startTime', None)
        task_date = task_start_time.date()
        
        # 如果任务结束在基地，飞行周期结束
        if isinstance(task, dict):
            arr_airport = task.get('arrAirport') or task.get('arriAirport')
        else:
            arr_airport = getattr(task, 'arrAirport', None) or getattr(task, 'arriAirport', None)
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
            task_type = task.get('type') if isinstance(task, dict) else getattr(task, 'type', None)
            if (task_type == 'flight' or 
                'positioning' in str(task_type) or 
                task_type == 'ground_duty'):
                # 计算实际周期开始日期
                actual_start_date = self._get_cycle_actual_start_date(current_label, task, crew)
                cycle_duration = (task_date - actual_start_date).days + 1
                if cycle_duration > 4:
                    return False
        
        # 检查周期间休息
        if (is_new_duty and current_label.last_base_return is not None and 
            current_label.current_cycle_start is None):
            days_since_base = (task_date - current_label.last_base_return).days
            if days_since_base < self.constraints.MIN_CYCLE_REST_DAYS:
                return False
        
        return True
    
    def _check_work_rest_pattern(self, current_label: Label, task: Dict, crew: Crew, is_new_duty: bool = False) -> bool:
        """检查值四修二工作模式约束"""
        if not hasattr(current_label, 'duty_days_count'):
            return True
        
        # 如果是新值勤日且已经连续工作4天
        task_type = task.get('type') if isinstance(task, dict) else getattr(task, 'type', None)
        task_start_time = task.get('startTime') if isinstance(task, dict) else getattr(task, 'startTime', None)
        if (is_new_duty and current_label.duty_days_count >= self.constraints.MAX_CONSECUTIVE_DUTY_DAYS and 
            task_type in ['flight']):
            
            # 检查是否有足够的休息时间
            if current_label.node and current_label.node.time:
                time_gap = task_start_time - current_label.node.time
                if time_gap.total_seconds() < self.constraints.MIN_REST_AFTER_CONSECUTIVE_DUTY * 3600:  # 少于48小时休息
                    return False
        
        return True
    
    def _check_total_flight_time_constraint(self, current_label: Label, task: Dict) -> bool:
        """检查总飞行值勤时间约束 - 修正版"""
        # 飞行值勤时间 = 飞行值勤日的总时长（从第一个任务开始到最后一个任务结束）
        # 而不是飞行时间的总和
        
        # 如果当前任务是飞行任务，需要检查飞行值勤时间
        task_type = task.get('type') if isinstance(task, dict) else getattr(task, 'type', None)
        if task_type == 'flight':
            # 计算当前值勤日的飞行值勤时间
            current_duty_time = 0
            if hasattr(current_label, 'duty_start_time') and current_label.duty_start_time:
                task_end_time = task.get('endTime') if isinstance(task, dict) else getattr(task, 'endTime', None)
                current_duty_time = (task_end_time - current_label.duty_start_time).total_seconds() / 3600.0
            
            # 计算总飞行值勤时间（包括当前值勤日）
            current_total_flight_duty_hours = getattr(current_label, 'total_flight_duty_hours', 0)
            potential_total_flight_duty_hours = current_total_flight_duty_hours + current_duty_time
            
            if potential_total_flight_duty_hours > self.constraints.MAX_TOTAL_FLIGHT_DUTY_HOURS:
                return False
        
        return True
    
    def _check_total_flight_time_constraint_for_duties(self, sorted_duties: List, crew: 'Crew') -> int:
        """检查总飞行值勤时间约束（改进版）"""
        violations = 0
        
        try:
            # 修正：计算飞行值勤时间而不是飞行时间
            # 飞行值勤时间 = 飞行值勤日的总时长（从第一个任务开始到最后一个飞行任务结束）
            
            # 将任务组织为值勤日
            duty_days = self._organize_duties_into_duty_days(sorted_duties)
            total_flight_duty_time = 0.0
            flight_duty_day_count = 0
            
            for duty_day in duty_days:
                # 检查是否包含飞行任务
                flight_tasks = [task for task in duty_day if self._is_flight_task(task)]
                if flight_tasks:
                    # 飞行值勤日：从第一个任务开始到最后一个飞行任务结束
                    first_task_start = self._get_task_start_time(duty_day[0])
                    last_flight_end = max(self._get_task_end_time(task) for task in flight_tasks)
                    
                    if first_task_start and last_flight_end:
                        flight_duty_duration = (last_flight_end - first_task_start).total_seconds() / 3600.0
                        total_flight_duty_time += flight_duty_duration
                        flight_duty_day_count += 1
                        
                        self.logger.debug(f"飞行值勤日 {flight_duty_day_count} 时长: {flight_duty_duration:.2f}小时")
            
            # 检查是否超过最大总飞行值勤时间
            if total_flight_duty_time > self.constraints.MAX_TOTAL_FLIGHT_DUTY_HOURS:
                violations += 1
                self.logger.warning(f"机组 {crew.id} 总飞行值勤时间 {total_flight_duty_time:.2f}小时 超过限制 {self.constraints.MAX_TOTAL_FLIGHT_DUTY_HOURS}小时")
            else:
                self.logger.info(f"机组 {crew.id} 总飞行值勤时间 {total_flight_duty_time:.2f}小时，包含 {flight_duty_day_count} 个飞行值勤日")
            
        except Exception as e:
            self.logger.error(f"检查总飞行值勤时间约束时发生错误: {e}")
            violations += 1
        
        return violations
    
    def validate_duty_day(self, duty_day: DutyDay) -> List[str]:
        """验证单个值勤日的约束"""
        violations = []
        
        # 1. 检查值勤时间限制
        if duty_day.get_duration_hours() > self.constraints.MAX_DUTY_DAY_HOURS:
            violations.append(f"值勤时间超限: {duty_day.get_duration_hours():.1f}小时 > {self.constraints.MAX_DUTY_DAY_HOURS}小时")
        
        # 2. 检查任务数量限制
        if len(duty_day.tasks) > self.constraints.MAX_TASKS_IN_DUTY:
            violations.append(f"值勤任务数超限: {len(duty_day.tasks)} > {self.constraints.MAX_TASKS_IN_DUTY}")
        
        # 3. 检查飞行任务数量限制
        flight_count = sum(1 for task in duty_day.tasks if isinstance(task, Flight))
        if flight_count > self.constraints.MAX_FLIGHTS_IN_DUTY:
            violations.append(f"值勤飞行数超限: {flight_count} > {self.constraints.MAX_FLIGHTS_IN_DUTY}")
        
        # 4. 检查值勤内飞行时间限制
        total_flight_time = sum(task.flyTime / 60.0 for task in duty_day.tasks if isinstance(task, Flight) and not getattr(task, 'is_positioning', False))
        if total_flight_time > self.constraints.MAX_FLIGHT_TIME_IN_DUTY_HOURS:
            violations.append(f"值勤飞行时间超限: {total_flight_time:.1f}小时 > {self.constraints.MAX_FLIGHT_TIME_IN_DUTY_HOURS}小时")
        
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
    
    def _validate_single_flight_cycle(self, cycle_duty_days: List) -> List[str]:
        """验证单个飞行周期的约束"""
        violations = []
        
        if not cycle_duty_days:
            return violations
        
        # 1. 检查是否包含飞行值勤日
        has_flight_duty = any(getattr(dd, 'is_flight_duty_day', False) for dd in cycle_duty_days)
        if not has_flight_duty:
            violations.append("飞行周期必须包含飞行值勤日")
        
        # 2. 检查末尾是否为飞行值勤日
        last_duty = cycle_duty_days[-1]
        if not getattr(last_duty, 'is_flight_duty_day', False):
            violations.append("飞行周期末尾必须是飞行值勤日")
        
        # 3. 检查日历日跨度限制
        if len(cycle_duty_days) > 0:
            start_date = getattr(cycle_duty_days[0], 'start_date', None)
            end_date = getattr(cycle_duty_days[-1], 'end_date', None)
            if start_date and end_date:
                span_days = (end_date - start_date).days + 1
                if span_days > 4:
                    violations.append(f"飞行周期跨度超限: {span_days}天 > 4天")
        
        return violations
    
    def _get_min_connection_time_for_tasks(self, prev_task, curr_task) -> timedelta:
        """获取两个任务之间的最小连接时间"""
        if isinstance(curr_task, Flight):
            if (isinstance(prev_task, Flight) and 
                hasattr(prev_task, 'aircraftNo') and hasattr(curr_task, 'aircraftNo') and 
                prev_task.aircraftNo == curr_task.aircraftNo):
                return self.constraints.MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
            else:
                return self.constraints.MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT
        elif isinstance(curr_task, BusInfo):
            return self.constraints.MIN_CONNECTION_TIME_BUS
        else:
            return self.constraints.DEFAULT_MIN_CONNECTION_TIME
    
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
                        if rest_days < self.constraints.MIN_CYCLE_REST_DAYS:
                            violations.append(f"飞行周期间休息不足: {rest_days}天 < {self.constraints.MIN_CYCLE_REST_DAYS}天")
                    
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
    
    def check_roster_violations(self, roster: 'Roster', crew: 'Crew') -> int:
        """
        检查排班方案的违规情况，返回违规次数
        使用新的约束验证框架
        """
        if not hasattr(roster, 'duties') or not roster.duties:
            # 尝试从labels获取任务
            all_tasks = []
            if hasattr(roster, 'labels'):
                for label in roster.labels:
                    all_tasks.extend(label.tasks)
            if not all_tasks:
                return 0
        else:
            all_tasks = roster.duties
        
        # 按时间排序任务
        all_tasks.sort(key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        # 使用新的验证方法
        validation_result = self.validate_roster_constraints(all_tasks, crew)
        
        # 统计违规数量
        violations = 0
        violations += len(validation_result['duty_day_violations'])
        violations += len(validation_result['flight_duty_period_violations'])
        violations += len(validation_result['flight_cycle_violations'])
        violations += len(validation_result['total_flight_time_violations'])
        
        return violations
    
    # 旧的_organize_into_duty_days方法已被新的organize_tasks_into_duty_days替代
    
    # 旧的_check_flight_cycle_violations_new方法已被新的validate_flight_cycle替代