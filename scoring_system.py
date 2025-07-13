#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整的排班方案评分系统
根据竞赛评分标准实现：
1. 值勤日日均飞时得分 = 值勤日日均飞时 * 1000
2. 未覆盖航班惩罚 = 未覆盖航班数量 * (-5)
3. 新增过夜站点惩罚 = 新增过夜站点数量 * (-10)
4. 外站过夜惩罚 = 外站过夜天数 * (-0.5)
5. 置位惩罚 = 置位次数 * (-0.5)
6. 违规惩罚 = 违规次数 * (-10)
"""

from datetime import datetime, timedelta
from typing import List, Dict, Set
from data_models import Flight, Roster, Crew, LayoverStation, BusInfo, GroundDuty
from unified_config import UnifiedConfig

class ScoringSystem:
    def __init__(self, flights: List[Flight], crews: List[Crew], layover_stations):
        self.flights = flights
        self.crews = crews
        # Handle both List[LayoverStation] and set of airport strings
        if isinstance(layover_stations, set):
            self.layover_stations_set = layover_stations
        else:
            self.layover_stations_set = {station.airport for station in layover_stations}
        
        # 使用统一配置的评分参数
        scoring_params = UnifiedConfig.get_scoring_params()
        optimization_params = UnifiedConfig.get_optimization_params()
        self.FLY_TIME_MULTIPLIER = scoring_params['fly_time_multiplier']
        self.FLIGHT_TIME_REWARD = optimization_params['flight_time_reward']
        self.UNCOVERED_FLIGHT_PENALTY = scoring_params['uncovered_flight_penalty']
        self.NEW_LAYOVER_STATION_PENALTY = scoring_params['new_layover_penalty']
        self.AWAY_OVERNIGHT_PENALTY = scoring_params['away_overnight_penalty']
        self.POSITIONING_PENALTY = scoring_params['positioning_penalty']
        self.VIOLATION_PENALTY = scoring_params['violation_penalty']
        
        # 优化配置参数
        self.MIN_OVERNIGHT_HOURS = UnifiedConfig.MIN_OVERNIGHT_HOURS  # 最小过夜时间（小时）
        self.VIOLATION_PENALTY = scoring_params['violation_penalty']
    
    # def calculate_roster_score(self, roster: Roster, crew: Crew) -> float:
    #     """
    #     计算单个排班方案的得分，严格按照赛题评分公式
    #     返回正值作为成本（用于最小化目标函数）使用满足覆盖率要求的初始解作为最终输出
        
    #     评分公式：
    #     1. 值勤日日均飞时得分 = 值勤日日均飞时 * 1000
    #     2. 新增过夜站点惩罚 = 新增过夜站点数量 * (-10)
    #     3. 外站过夜惩罚 = 外站过夜天数 * (-0.5)
    #     4. 置位惩罚 = 置位次数 * (-0.5)
    #     5. 违规惩罚 = 违规次数 * (-10)
    #     """
    #     if not roster.duties:
    #         return 0.0
        
    #     # 1. 计算飞行时间和值勤日历日
    #     total_flight_hours = 0.0
    #     duty_calendar_days = set()
    #     positioning_count = 0
    #     away_overnight_days = 0
    #     new_layover_stations = set()
        
    #     # 按时间排序任务
    #     sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
    #     # 处理每个任务
    #     for duty in sorted_duties:
    #         if isinstance(duty, Flight):
    #             # 计算飞行时间（分钟转小时）
    #             total_flight_hours += duty.flyTime / 60.0
                
    #             # 计算值勤日历日（跨零点时记为两个日历日）
    #             start_date = duty.std.date()
    #             end_date = duty.sta.date()
    #             current_date = start_date
    #             while current_date <= end_date:
    #                 duty_calendar_days.add(current_date)
    #                 current_date += timedelta(days=1)
                
    #             # 检查新增过夜站点
    #             # 1. 飞行值勤日以不可过夜机场作为起点或终点时，记为新增可过夜机场
    #             if duty.depaAirport not in self.layover_stations_set:
    #                 new_layover_stations.add(duty.depaAirport)
    #             if duty.arriAirport not in self.layover_stations_set:
    #                 new_layover_stations.add(duty.arriAirport)
            
    #         # 计算置位次数（包括飞行置位和大巴置位，但不包括占位任务groundDuty）
    #         elif self._is_positioning_task(duty):
    #             positioning_count += 1
                
    #             # 置位任务也可能跨日历日
    #             start_time = getattr(duty, 'startTime', None) or duty.get('startTime') if isinstance(duty, dict) else None
    #             end_time = getattr(duty, 'endTime', None) or duty.get('endTime') if isinstance(duty, dict) else None
                
    #             if start_time and end_time:
    #                 start_date = start_time.date()
    #                 end_date = end_time.date()
    #                 current_date = start_date
    #                 while current_date <= end_date:
    #                     duty_calendar_days.add(current_date)
    #                     current_date += timedelta(days=1)
        
    #     # 2. 计算外站过夜天数（改进版）
    #     away_overnight_days += self._calculate_overnight_days_enhanced(roster, crew, sorted_duties, new_layover_stations)
        
    #     # 3. 计算各项得分
    #     total_duty_days = len(duty_calendar_days)
    #     avg_daily_fly_time = total_flight_hours / total_duty_days if total_duty_days > 0 else 0
        
    #     # 按照赛题公式计算得分（但在列生成中，roster基础成本不包含飞行奖励）
    #     # 注意：这个方法主要用于最终评分，在列生成过程中飞行奖励通过执行变量单独计算
    #     fly_time_score = 0.0  # 在列生成中不计算飞行奖励，避免双重计算
    #     new_layover_penalty = len(new_layover_stations) * self.NEW_LAYOVER_STATION_PENALTY  # 每个新增过夜站点扣分
    #     away_overnight_penalty = away_overnight_days * self.AWAY_OVERNIGHT_PENALTY  # 每天外站过夜扣分
    #     positioning_penalty = positioning_count * self.POSITIONING_PENALTY  # 每个置位扣分
        
    #     # 4. 违规检查（完整实现）
    #     violation_count = self._check_roster_violations(roster, crew)
    #     violation_penalty = violation_count * self.VIOLATION_PENALTY  # 每次违规惩罚
        
    #     # 5. 总得分计算（不包含飞行奖励）
    #     total_score = (fly_time_score + new_layover_penalty + away_overnight_penalty + 
    #                   positioning_penalty + violation_penalty)
        
    #     # 转换为成本：得分越高，成本越低
    #     # 使用负得分作为成本，确保优化目标正确
    #     return -total_score
    
    def _calculate_overnight_days_enhanced(self, roster, crew, sorted_duties, new_layover_stations):
        """
        精确化的外站过夜天数计算方法
        
        外站过夜计算规则：
        1. 历史停留：计划期开始前机组在外站的过夜天数
        2. 任务间隔：两个任务之间在外站的过夜天数
        3. 计划期结束：计划期结束时机组在外站的过夜天数
        
        过夜判定标准：
        - 休息时间 >= 8小时（MIN_OVERNIGHT_HOURS）
        - 跨零点：按实际跨越的日历天数计算
        - 不跨零点但满足最小休息时间：计1天
        """
        try:
            away_overnight_days = 0
            crew_base = crew.base
            
            # 计划期边界定义（从数据中动态获取）
            try:
                # 尝试从attention模块的data_config动态获取
                import sys
                import os
                current_dir = os.path.dirname(os.path.abspath(__file__))
                attention_path = os.path.join(current_dir, 'attention')
                if attention_path not in sys.path:
                    sys.path.append(attention_path)
                
                from data_config import get_data_config
                data_config = get_data_config()
                plan_start_time, plan_end_time = data_config.get_planning_time_range()
                PLAN_START_DATE = plan_start_time.date()
                PLAN_END_DATE = plan_end_time.date()
            except Exception as e:
                # 降级到配置文件
                from unified_config import UnifiedConfig
                PLAN_START_DATE = UnifiedConfig.get_planning_start_date().date()
                PLAN_END_DATE = UnifiedConfig.get_planning_end_date().date()
            
            # 情况1：历史停留机场为外站的过夜天数
            if hasattr(crew, 'stayStation') and crew.stayStation and crew.stayStation != crew_base:
                # 检查历史停留机场是否为新增过夜站点
                if crew.stayStation not in self.layover_stations_set:
                    new_layover_stations.add(crew.stayStation)
                
                # 计算从计划期开始到第一个任务开始的过夜天数
                if sorted_duties:
                    first_task_start = getattr(sorted_duties[0], 'std', getattr(sorted_duties[0], 'startTime', None))
                    if first_task_start:
                        first_task_date = first_task_start.date()
                        # 从计划期开始日到第一个任务开始日的天数
                        days_before_first_task = (first_task_date - PLAN_START_DATE).days
                        away_overnight_days += max(0, days_before_first_task)
                else:
                    # 如果没有任务，整个计划期都在外站过夜
                    days_in_plan_period = (PLAN_END_DATE - PLAN_START_DATE).days + 1
                    away_overnight_days += days_in_plan_period
            
            # 情况2：任务间隔期间的外站过夜
            for i in range(len(sorted_duties) - 1):
                current_duty = sorted_duties[i]
                next_duty = sorted_duties[i + 1]
                
                # 获取当前任务的结束地点和时间
                current_end_airport = self._get_task_end_airport(current_duty, crew_base)
                current_end_time = self._get_task_end_time(current_duty)
                
                # 获取下一个任务的开始时间
                next_start_time = self._get_task_start_time(next_duty)
                
                # 计算外站过夜天数
                if (current_end_airport and current_end_airport != crew_base and 
                    current_end_time and next_start_time):
                    
                    # 检查过夜机场是否为新增过夜站点
                    if current_end_airport not in self.layover_stations_set:
                        new_layover_stations.add(current_end_airport)
                    
                    # 计算休息时间
                    rest_hours = (next_start_time - current_end_time).total_seconds() / 3600
                    
                    # 只有休息时间超过最小过夜时间才算过夜
                    if rest_hours >= self.MIN_OVERNIGHT_HOURS:
                        current_end_date = current_end_time.date()
                        next_start_date = next_start_time.date()
                        
                        if next_start_date > current_end_date:
                            # 跨零点：按实际跨越的日历天数计算
                            overnight_days = (next_start_date - current_end_date).days
                            away_overnight_days += overnight_days
                        else:
                            # 不跨零点但满足最小休息时间：计1天
                            away_overnight_days += 1
            
            # 情况3：计划期结束时在外站的过夜天数
            if sorted_duties:
                last_duty = sorted_duties[-1]
                last_end_airport = self._get_task_end_airport(last_duty, crew_base)
                last_end_time = self._get_task_end_time(last_duty)
                
                if last_end_airport and last_end_airport != crew_base and last_end_time:
                    # 检查计划期结束时的过夜机场是否为新增过夜站点
                    if last_end_airport not in self.layover_stations_set:
                        new_layover_stations.add(last_end_airport)
                    
                    last_end_date = last_end_time.date()
                    
                    # 计算从最后任务结束到计划期结束的过夜天数
                    if last_end_date <= PLAN_END_DATE:
                        # 如果最后任务在计划期内结束，计算到计划期结束的天数
                        days_after_last_task = (PLAN_END_DATE - last_end_date).days
                        away_overnight_days += max(0, days_after_last_task)
                    # 如果最后任务结束时间超出计划期，不计算额外过夜天数
            
            return away_overnight_days
            
        except Exception as e:
            # 发生错误时记录日志并返回0
            import logging
            logging.error(f"计算外站过夜天数时发生错误: {e}")
            return 0
    
    def _get_task_start_airport(self, task, crew_base: str) -> str:
        """获取任务的开始机场"""
        if isinstance(task, Flight):
            return task.depaAirport
        elif isinstance(task, BusInfo):
            return getattr(task, 'depaAirport', crew_base)
        elif isinstance(task, GroundDuty):
            return getattr(task, 'airport', crew_base)
        elif isinstance(task, dict):
            return task.get('depaAirport', task.get('airport', crew_base))
        else:
            return crew_base
    
    def _get_task_end_airport(self, task, default_base):
        """获取任务结束机场"""
        if isinstance(task, Flight):
            return task.arriAirport
        elif isinstance(task, BusInfo):
            return getattr(task, 'arriAirport', default_base)
        elif isinstance(task, GroundDuty):
            return getattr(task, 'airport', default_base)
        elif isinstance(task, dict):
            return task.get('arriAirport', task.get('airport', default_base))
        elif hasattr(task, 'arriAirport'):
            return task.arriAirport
        elif hasattr(task, 'arrAirport'):
            return task.arrAirport
        else:
            return default_base
    
    def _get_task_end_time(self, task):
        """获取任务结束时间"""
        if isinstance(task, Flight):
            return task.sta
        elif hasattr(task, 'endTime'):
            return task.endTime
        else:
            return None
    
    def _get_task_start_time(self, task):
        """获取任务开始时间"""
        if isinstance(task, Flight):
            return task.std
        elif hasattr(task, 'startTime'):
            return task.startTime
        else:
            return None
    
    def calculate_total_score(self, rosters: List[Roster]) -> Dict[str, float]:
        """
        计算所有排班方案的总得分，严格按照赛题评分公式
        返回各项得分的详细分解
        
        评分公式：
        1. 值勤日日均飞时得分 = 总飞行小时/总值勤日历日数量 * 1000
        2. 未覆盖航班惩罚 = 未覆盖航班数量 * (-5)
        3. 新增过夜站点惩罚 = 新增过夜站点数量 * (-10)
        4. 外站过夜惩罚 = 外站过夜天数 * (-0.5)
        5. 置位惩罚 = 置位次数 * (-0.5)
        6. 违规惩罚 = 违规次数 * (-10)
        """
        # 初始化统计变量
        total_flight_hours = 0.0
        all_duty_calendar_days = set()
        new_layover_stations = set()
        away_overnight_days = 0
        positioning_count = 0
        violation_count = 0
        
        covered_flight_ids = set()
        
        for roster in rosters:
            crew = next((c for c in self.crews if c.crewId == roster.crew_id), None)
            if not crew:
                continue

            # 按时间排序任务
            sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))

            # 检查违规情况
            roster_violations = self._check_roster_violations(roster, crew)
            violation_count += roster_violations

            # 统计每个roster的贡献
            for duty in sorted_duties:
                if isinstance(duty, Flight):
                    covered_flight_ids.add(duty.id)
                    # 只计算执飞航班的飞行时间，置位航班不计入
                    is_positioning = getattr(duty, 'is_positioning', False)
                    if not is_positioning:
                        total_flight_hours += duty.flyTime / 60.0
                    
                    # 计算值勤日历日（跨零点时记为两个日历日）
                    start_date = duty.std.date()
                    end_date = duty.sta.date()
                    current_date = start_date
                    while current_date <= end_date:
                        all_duty_calendar_days.add(current_date)
                        current_date += timedelta(days=1)
                    
                    # 检查新增过夜站点
                    if duty.depaAirport not in self.layover_stations_set:
                        new_layover_stations.add(duty.depaAirport)
                    if duty.arriAirport not in self.layover_stations_set:
                        new_layover_stations.add(duty.arriAirport)
                
                # 计算置位次数（注意：groundDuty是占位任务，不是置位任务）
                elif self._is_positioning_task(duty):
                    positioning_count += 1
                    
                    # 置位任务的日历日
                    start_time = getattr(duty, 'startTime', None) or duty.get('startTime') if isinstance(duty, dict) else None
                    end_time = getattr(duty, 'endTime', None) or duty.get('endTime') if isinstance(duty, dict) else None
                    
                    if start_time and end_time:
                        start_date = start_time.date()
                        end_date = end_time.date()
                        current_date = start_date
                        while current_date <= end_date:
                            all_duty_calendar_days.add(current_date)
                            current_date += timedelta(days=1)
            
            # 计算外站过夜天数（使用统一方法）
            roster_overnight_penalty = self._calculate_unified_overnight_penalty(roster, crew)
            # 将惩罚转换为天数（用于统计）
            optimization_params = UnifiedConfig.get_optimization_params()
            away_overnight_penalty_rate = optimization_params['away_overnight_penalty']
            if away_overnight_penalty_rate > 0:
                roster_overnight_days = int(roster_overnight_penalty / away_overnight_penalty_rate)
                away_overnight_days += roster_overnight_days
            
            # 检查新增过夜站点
            for i in range(len(sorted_duties) - 1):
                current_duty = sorted_duties[i]
                next_duty = sorted_duties[i + 1]
                
                current_end_airport = self._get_task_end_airport(current_duty, crew.base)
                current_end_time = self._get_task_end_time(current_duty)
                next_start_time = self._get_task_start_time(next_duty)
                
                if (current_end_airport and current_end_airport != crew.base and 
                    current_end_time and next_start_time):
                    
                    # 检查过夜机场是否为新增过夜站点
                    if current_end_airport not in self.layover_stations_set:
                        new_layover_stations.add(current_end_airport)
                    
                    rest_time = next_start_time - current_end_time
                    min_rest_hours = getattr(UnifiedConfig, 'MIN_REST_HOURS', 12)
                    if rest_time >= timedelta(hours=min_rest_hours):
                        # 符合过夜条件，新增过夜站点已在上面添加
                        pass
        
        # 计算未覆盖航班数量
        uncovered_flights = len(self.flights) - len(covered_flight_ids)
        
        # 计算各项得分（严格按照赛题公式）
        total_duty_days = len(all_duty_calendar_days)
        avg_daily_fly_time = total_flight_hours / total_duty_days if total_duty_days > 0 else 0
        
        fly_time_score = avg_daily_fly_time * self.FLY_TIME_MULTIPLIER  # 值勤日日均飞时 * FLY_TIME_MULTIPLIER
        uncovered_penalty = uncovered_flights * self.UNCOVERED_FLIGHT_PENALTY  # 每个未覆盖航班惩罚
        new_layover_penalty = len(new_layover_stations) * self.NEW_LAYOVER_STATION_PENALTY  # 每个新增过夜站点扣分
        away_overnight_penalty = away_overnight_days * self.AWAY_OVERNIGHT_PENALTY  # 每天外站过夜扣分
        positioning_penalty = positioning_count * self.POSITIONING_PENALTY  # 每个置位扣分
        violation_penalty = violation_count * (-10)  # 每次违规扣10分
        
        total_score = (fly_time_score + uncovered_penalty + new_layover_penalty + 
                      away_overnight_penalty + positioning_penalty + violation_penalty)
        
        return {
            'total_score': total_score,
            'fly_time_score': fly_time_score,
            'uncovered_penalty': uncovered_penalty,
            'new_layover_penalty': new_layover_penalty,
            'away_overnight_penalty': away_overnight_penalty,
            'positioning_penalty': positioning_penalty,
            'violation_penalty': violation_penalty,
            'avg_daily_fly_time': avg_daily_fly_time,
            'uncovered_flights': uncovered_flights,
            'new_layover_stations': len(new_layover_stations),
            'away_overnight_days': away_overnight_days,
            'positioning_count': positioning_count,
            'violation_count': violation_count
        }
    
    def calculate_unified_roster_cost(self, roster: Roster, crew: Crew, global_duty_days_denominator: float = 0.0) -> float:
        """
        计算roster的统一成本，与主问题和子问题保持一致
        使用统一配置的参数，替代initial_solution_generator中的同名函数
        
        Args:
            roster: 排班方案
            crew: 机组
            global_duty_days_denominator: 全局执勤日分母，用于全局分母飞行奖励计算
        """
        if not roster.duties:
            return 0.0
        
        # 获取统一配置参数
        optimization_params = UnifiedConfig.get_optimization_params()
        positioning_penalty_rate = optimization_params['positioning_penalty']
        
        # 计算各项成本
        total_cost = 0.0
        
        # 1. 计算总飞行时间
        total_flight_hours = 0.0
        for duty in roster.duties:
            if hasattr(duty, 'flyTime') and duty.flyTime is not None:
                # 只计算执飞航班的飞行时间
                if not getattr(duty, 'is_positioning', False):
                    total_flight_hours += duty.flyTime / 60.0
        
        # 2. 飞行时间奖励（统一使用全局分母方式）
        flight_reward = 0.0
        if global_duty_days_denominator > 0:
            # 使用全局日均飞时近似分配：FLIGHT_TIME_REWARD * 该列飞行时间 / 全局执勤日分母
            flight_reward = self.FLIGHT_TIME_REWARD * total_flight_hours / global_duty_days_denominator
        else:
            # 当分母为0时，飞行奖励为0
            flight_reward = 0.0
        
        # 3. 置位惩罚
        positioning_penalty = 0.0
        for duty in roster.duties:
            if self._is_positioning_task(duty):
                positioning_penalty += positioning_penalty_rate
        
        # 4. 外站过夜惩罚（使用统一的计算方法）
        overnight_penalty = self._calculate_unified_overnight_penalty(roster, crew)
        
        # 计算总成本：惩罚项 - 奖励项
        total_cost = positioning_penalty + overnight_penalty - flight_reward
        
        return total_cost
    
    def _calculate_unified_overnight_penalty(self, roster: Roster, crew: Crew) -> float:
        """
        统一的外站过夜惩罚计算方法
        替代各模块中不一致的外站过夜计算逻辑
        """
        optimization_params = UnifiedConfig.get_optimization_params()
        away_overnight_penalty_rate = optimization_params['away_overnight_penalty']
        
        overnight_penalty = 0.0
        sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        # 首尾任务的逻辑
        if len(sorted_duties) == 0:
            return overnight_penalty
        first_task = sorted_duties[0]
        last_task = sorted_duties[-1]

        base = crew.base
        
        # 从配置中获取计划期开始和结束日期
        UnifiedConfig.initialize_planning_dates()
        PLAN_START_DATE = UnifiedConfig.PLANNING_START_DATE
        PLAN_END_DATE = UnifiedConfig.PLANNING_END_DATE

        first_task_start = self._get_task_start_time(first_task)
        first_task_place = self._get_task_start_airport(first_task, crew.base)
        if first_task_start and first_task_place and first_task_place != base:
            days_before_first_task = (first_task_start - PLAN_START_DATE).days
            overnight_penalty += max(1, days_before_first_task) * away_overnight_penalty_rate

        last_task_end = self._get_task_end_time(last_task)
        last_task_place = self._get_task_end_airport(last_task, crew.base)
        if last_task_end and last_task_place and last_task_place != base:
            days_after_last_task = (PLAN_END_DATE - last_task_end).days
            overnight_penalty += max(0, days_after_last_task) * away_overnight_penalty_rate
        for i in range(len(sorted_duties) - 1):
            current_duty = sorted_duties[i]
            next_duty = sorted_duties[i + 1]
            
            # 获取当前任务的结束地点和时间
            current_end_airport = self._get_task_end_airport(current_duty, crew.base)
            current_end_time = self._get_task_end_time(current_duty)
            
            # 获取下一个任务的开始时间
            next_start_time = self._get_task_start_time(next_duty)
            
            # 检查外站过夜
            if (current_end_airport and current_end_airport != crew.base and 
                current_end_time and next_start_time):
                
                rest_time = next_start_time - current_end_time
                min_rest_hours = getattr(UnifiedConfig, 'MIN_REST_HOURS', 12)
                if rest_time >= timedelta(hours=min_rest_hours):
                    overnight_days = (next_start_time.date() - current_end_time.date()).days
                    if overnight_days > 0:
                        overnight_penalty += overnight_days * away_overnight_penalty_rate
                    elif rest_time >= timedelta(hours=self.MIN_OVERNIGHT_HOURS):
                        # 不跨零点但满足最小过夜时间：计1天
                        overnight_penalty += away_overnight_penalty_rate
        
        return overnight_penalty
    
    def calculate_roster_cost_with_violations(self, roster: Roster, crew: Crew, global_duty_days_denominator: float = 0.0) -> Dict[str, float]:
        """
        计算包含违规检查的完整roster成本
        用于主问题的完整评估，包括违规惩罚
        
        Args:
            roster: 排班方案
            crew: 机组
            global_duty_days_denominator: 全局执勤日分母，用于全局分母飞行奖励计算
        """
        if not roster.duties:
            return {
                'total_cost': 0.0,
                'flight_reward': 0.0,
                'positioning_penalty': 0.0,
                'overnight_penalty': 0.0,
                'violation_penalty': 0.0,
                'violation_count': 0
            }
        
        # 获取统一配置参数
        optimization_params = UnifiedConfig.get_optimization_params()
        positioning_penalty_rate = optimization_params['positioning_penalty']
        
        # 1. 计算总飞行时间
        total_flight_hours = 0.0
        for duty in roster.duties:
            if hasattr(duty, 'flyTime') and duty.flyTime is not None:
                # 只计算执飞航班的飞行时间
                if not getattr(duty, 'is_positioning', False):
                    total_flight_hours += duty.flyTime / 60.0
        
        # 2. 飞行时间奖励（统一使用全局分母方式）
        flight_reward = 0.0
        if global_duty_days_denominator > 0:
            # 使用全局日均飞时近似分配：FLIGHT_TIME_REWARD * 该列飞行时间 / 全局执勤日分母
            flight_reward = self.FLIGHT_TIME_REWARD * total_flight_hours / global_duty_days_denominator
        else:
            # 当分母为0时，飞行奖励为0
            flight_reward = 0.0
        
        # 3. 置位惩罚
        positioning_penalty = 0.0
        for duty in roster.duties:
            if self._is_positioning_task(duty):
                positioning_penalty += positioning_penalty_rate
        
        # 4. 外站过夜惩罚
        overnight_penalty = self._calculate_unified_overnight_penalty(roster, crew)
        
        # 5. 违规检查和惩罚
        violation_count = self._check_roster_violations(roster, crew)
        violation_penalty = violation_count * self.VIOLATION_PENALTY
        
        # 计算总成本
        total_cost = positioning_penalty + overnight_penalty + violation_penalty - flight_reward
        
        return {
            'total_cost': total_cost,
            'flight_reward': flight_reward,
            'positioning_penalty': positioning_penalty,
            'overnight_penalty': overnight_penalty,
            'violation_penalty': violation_penalty,
            'violation_count': violation_count
        }
    
    def calculate_roster_cost_with_dual_prices(self, roster: Roster, crew: Crew, 
                                             dual_prices: Dict[str, float], 
                                             crew_sigma_dual: float,
                                             global_duty_days_denominator: float = 0.0,
                                             ground_duty_duals: Dict[str, float] = None) -> Dict[str, float]:
        """
        计算单个排班方案的完整成本，包括对偶价格
        返回详细的成本分解，用于reduced cost计算
        """
        if not roster.duties:
            return {
                'total_cost': 0.0,  # c_j = 0 for empty roster
                'flight_reward': 0.0,
                'dual_price_total': 0.0,
                'dual_contribution': -crew_sigma_dual,  # π^T A_j = -crew_sigma_dual
                'positioning_penalty': 0.0,
                'overnight_penalty': 0.0,
                'other_costs': 0.0,
                'crew_sigma_dual': crew_sigma_dual,
                'reduced_cost': 0.0 - (-crew_sigma_dual),  # c_j - π^T A_j = 0 - (-crew_sigma_dual) = crew_sigma_dual
                'flight_count': 0,
                'total_flight_hours': 0.0,
                'duty_days': 0,
                'avg_daily_flight_hours': 0.0,
                'positioning_count': 0,
                'overnight_count': 0
            }
        
        # 1. 计算基础统计信息
        total_flight_hours = 0.0
        duty_calendar_days = set()
        flight_count = 0
        
        # 按时间排序任务
        sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        for duty in sorted_duties:
            if isinstance(duty, Flight):
                total_flight_hours += duty.flyTime / 60.0
                flight_count += 1
                
                # 计算值勤日历日（跨零点时记为两个日历日）
                start_date = duty.std.date()
                end_date = duty.sta.date()
                current_date = start_date
                while current_date <= end_date:
                    duty_calendar_days.add(current_date)
                    current_date += timedelta(days=1)
        
        # 根据新要求：分母直接为该roster的值勤天数，不再考虑不重复日历天数
        total_duty_days = len(duty_calendar_days)  # 这个roster的值勤天数
        avg_daily_flight_hours = total_flight_hours / total_duty_days if total_duty_days > 0 else 0.0
        
        # 根据全局日均飞时近似分配逻辑计算飞行奖励
        optimization_params = UnifiedConfig.get_optimization_params()
        flight_reward = 0.0
        
        if global_duty_days_denominator > 0:
            # 使用全局日均飞时近似分配：FLIGHT_TIME_REWARD * 该列飞行时间 / 全局执勤日分母
            flight_reward = self.FLIGHT_TIME_REWARD * total_flight_hours / global_duty_days_denominator
        else:
            # 当分母为0时，飞行奖励为0
            flight_reward = 0.0
        
        # 2. 计算对偶价格收益（分别计算航班和占位任务）
        flight_dual_total = 0.0
        ground_duty_dual_total = 0.0
        if ground_duty_duals is None:
            ground_duty_duals = {}
        
        for duty in roster.duties:
            if isinstance(duty, Flight):
                flight_dual_total += dual_prices.get(duty.id, 0.0)
            elif isinstance(duty, GroundDuty):
                # 添加占位任务的对偶价格
                ground_duty_dual_total += ground_duty_duals.get(duty.id, 0.0)
        
        # 总对偶价格收益
        dual_price_total = flight_dual_total + ground_duty_dual_total
        
        # 3. 计算置位惩罚（使用统一配置的核心成本参数）
        optimization_params = UnifiedConfig.get_optimization_params()
        positioning_penalty_rate = optimization_params['positioning_penalty']
        positioning_penalty = 0.0
        positioning_count = 0
        for duty in roster.duties:
            if isinstance(duty, Flight):
                # 检查是否为置位航班（根据is_positioning属性或任务类型判断）
                if (getattr(duty, 'is_positioning', False) or 
                    (hasattr(duty, 'type') and 'positioning' in str(duty.type))):
                    positioning_penalty += positioning_penalty_rate
                    positioning_count += 1
            elif isinstance(duty, BusInfo):
                # 巴士任务（大巴置位）- 在attention模块中标记为positioning_bus
                positioning_penalty += positioning_penalty_rate
                positioning_count += 1
            elif isinstance(duty, GroundDuty):
                # 地面值勤任务（占位任务，如培训、待命等），不是置位任务
                pass
        
        # 4. 计算外站过夜惩罚（使用统一方法）
        overnight_penalty = self._calculate_unified_overnight_penalty(roster, crew)
        
        # 5. 计算违规惩罚（新增）
        violation_count = self._check_roster_violations(roster, crew)
        violation_penalty_rate = optimization_params['violation_penalty']
        violation_penalty = violation_count * violation_penalty_rate
        
        # 计算过夜次数（用于统计）
        overnight_count = 0
        for i in range(len(sorted_duties) - 1):
            current_duty = sorted_duties[i]
            next_duty = sorted_duties[i + 1]
            
            current_end_airport = self._get_task_end_airport(current_duty, crew.base)
            current_end_time = self._get_task_end_time(current_duty)
            next_start_time = self._get_task_start_time(next_duty)
            
            if (current_end_airport and current_end_airport != crew.base and 
                current_end_time and next_start_time):
                
                rest_time = next_start_time - current_end_time
                min_rest_hours = getattr(UnifiedConfig, 'MIN_REST_HOURS', 12)
                if rest_time >= timedelta(hours=min_rest_hours):
                    overnight_days = (next_start_time.date() - current_end_time.date()).days
                    if overnight_days > 0:
                        overnight_count += overnight_days
                    elif rest_time >= timedelta(hours=self.MIN_OVERNIGHT_HOURS):
                        overnight_count += 1
        
        # 6. 其他成本
        other_costs = 0.0
        for duty in roster.duties:
            if hasattr(duty, 'cost'):
                other_costs += duty.cost
        
        # 7. 计算总成本和reduced cost
        # 最小化问题的reduced cost计算: c_j - π^T A_j
        # c_j = 原始成本 (penalties + violation_penalty - flight_reward，负值表示收益)
        # π^T A_j = 对偶价格贡献 (dual_price_total - crew_sigma_dual)
        # 注意：机组约束 ∑(x_r) ≤ 1 转换为标准形式 -∑(x_r) ≥ -1 时，系数矩阵中对应-1
        # 因此对偶价格贡献为 dual_price_total - crew_sigma_dual
        # 当reduced_cost < 0时，表示该roster有价值，应该加入主问题
        total_cost = positioning_penalty + overnight_penalty + violation_penalty  - flight_reward
        dual_contribution = dual_price_total - crew_sigma_dual  # 修正：机组约束转换为标准形式后系数为-1
        reduced_cost = total_cost - dual_contribution
        
        return {
            'total_cost': total_cost,
            'flight_reward': flight_reward,
            'dual_price_total': dual_price_total,
            'flight_dual_total': flight_dual_total,
            'ground_duty_dual_total': ground_duty_dual_total,
            'dual_contribution': dual_contribution,
            'positioning_penalty': positioning_penalty,
            'overnight_penalty': overnight_penalty,
            'violation_penalty': violation_penalty,
            'other_costs': other_costs,
            'crew_sigma_dual': crew_sigma_dual,
            'reduced_cost': reduced_cost,
            'flight_count': flight_count,
            'total_flight_hours': total_flight_hours,
            'duty_days': total_duty_days,
            'avg_daily_flight_hours': avg_daily_flight_hours,
            'positioning_count': positioning_count,
            'overnight_count': overnight_count,
            'violation_count': violation_count
        }
    
    def _check_roster_violations(self, roster: Roster, crew: Crew) -> int:
        """
        检查排班方案的违规情况
        使用统一的约束检查器
        """
        from constraint_checker import UnifiedConstraintChecker
        
        # 创建约束检查器实例
        constraint_checker = UnifiedConstraintChecker(self.layover_stations_set)
        
        # 使用统一的约束检查方法
        return constraint_checker.check_roster_violations(roster, crew)
    
    def _is_positioning_task(self, task):
        """
        识别置位任务（包括飞行置位和大巴置位，但不包括占位任务groundDuty）
        支持字典类型和对象类型的任务数据
        """
        if isinstance(task, dict):
            task_type = task.get('type', '')
        else:
            task_type = getattr(task, 'type', '')
        
        # 置位任务：飞行置位和大巴置位
        return (str(task_type) == 'positioning_flight' or 
                str(task_type) == 'positioning_bus' or
                ('positioning' in str(task_type).lower() and 'ground' not in str(task_type).lower()))
    
    def _is_ground_duty_task(self, task):
        """
        识别占位任务（groundDuty）
        支持字典类型和对象类型的任务数据
        """
        if isinstance(task, dict):
            task_type = task.get('type', '')
            task_id = task.get('id', '') or task.get('taskId', '')
        else:
            task_type = getattr(task, 'type', '')
            task_id = getattr(task, 'id', '')
        
        # 占位任务：groundDuty类型或ID以Grd_开头
        return (str(task_type) == 'ground_duty' or 
                str(task_type) == 'groundDuty' or
                str(task_id).startswith('Grd_'))
