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
        self.FLY_TIME_MULTIPLIER = scoring_params['fly_time_multiplier']
        self.UNCOVERED_FLIGHT_PENALTY = scoring_params['uncovered_flight_penalty']
        self.NEW_LAYOVER_STATION_PENALTY = scoring_params['new_layover_penalty']
        self.AWAY_OVERNIGHT_PENALTY = scoring_params['away_overnight_penalty']
        self.POSITIONING_PENALTY = scoring_params['positioning_penalty']
        self.VIOLATION_PENALTY = scoring_params['violation_penalty']
    
    def calculate_roster_score(self, roster: Roster, crew: Crew) -> float:
        """
        计算单个排班方案的得分，严格按照赛题评分公式
        返回正值作为成本（用于最小化目标函数）使用满足覆盖率要求的初始解作为最终输出
        
        评分公式：
        1. 值勤日日均飞时得分 = 值勤日日均飞时 * 1000
        2. 新增过夜站点惩罚 = 新增过夜站点数量 * (-10)
        3. 外站过夜惩罚 = 外站过夜天数 * (-0.5)
        4. 置位惩罚 = 置位次数 * (-0.5)
        5. 违规惩罚 = 违规次数 * (-10)
        """
        if not roster.duties:
            return 0.0
        
        # 1. 计算飞行时间和值勤日历日
        total_flight_hours = 0.0
        duty_calendar_days = set()
        positioning_count = 0
        away_overnight_days = 0
        new_layover_stations = set()
        
        # 按时间排序任务
        sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        # 处理每个任务
        for duty in sorted_duties:
            if isinstance(duty, Flight):
                # 计算飞行时间（分钟转小时）
                total_flight_hours += duty.flyTime / 60.0
                
                # 计算值勤日历日（跨零点时记为两个日历日）
                start_date = duty.std.date()
                end_date = duty.sta.date()
                current_date = start_date
                while current_date <= end_date:
                    duty_calendar_days.add(current_date)
                    current_date += timedelta(days=1)
                
                # 检查新增过夜站点
                # 1. 飞行值勤日以不可过夜机场作为起点或终点时，记为新增可过夜机场
                if duty.depaAirport not in self.layover_stations_set:
                    new_layover_stations.add(duty.depaAirport)
                if duty.arriAirport not in self.layover_stations_set:
                    new_layover_stations.add(duty.arriAirport)
            
            # 计算置位次数（包括飞行置位和大巴置位，但不包括占位任务groundDuty）
            elif self._is_positioning_task(duty):
                positioning_count += 1
                
                # 置位任务也可能跨日历日
                start_time = getattr(duty, 'startTime', None) or duty.get('startTime') if isinstance(duty, dict) else None
                end_time = getattr(duty, 'endTime', None) or duty.get('endTime') if isinstance(duty, dict) else None
                
                if start_time and end_time:
                    start_date = start_time.date()
                    end_date = end_time.date()
                    current_date = start_date
                    while current_date <= end_date:
                        duty_calendar_days.add(current_date)
                        current_date += timedelta(days=1)
        
        # 2. 计算外站过夜天数
        # 检查历史停留机场（计划期开始前的外站过夜）
        if hasattr(crew, 'stayStation') and crew.stayStation != crew.base:
            # 2. 检查历史停留机场是否为新增过夜站点
            if crew.stayStation not in self.layover_stations_set:
                new_layover_stations.add(crew.stayStation)
            
            # 情况②：历史过夜站点为外站，计算到第一个任务开始的跨零点天数
            if sorted_duties:
                first_task_start = getattr(sorted_duties[0], 'std', getattr(sorted_duties[0], 'startTime', None))
                if first_task_start:
                    # 假设计划期开始日为第一个任务的日期
                    plan_start_date = first_task_start.date()
                    overnight_days = (first_task_start.date() - plan_start_date).days
                    away_overnight_days += max(0, overnight_days)
        
        # 计算值勤日间隔的外站过夜
        for i in range(len(sorted_duties) - 1):
            current_duty = sorted_duties[i]
            next_duty = sorted_duties[i + 1]
            
            # 获取当前任务的结束地点和时间
            current_end_airport = None
            current_end_time = None
            
            if isinstance(current_duty, Flight):
                current_end_airport = current_duty.arriAirport
                current_end_time = current_duty.sta
            elif hasattr(current_duty, 'endTime'):
                current_end_time = current_duty.endTime
                current_end_airport = getattr(current_duty, 'arriAirport', crew.base)
            
            # 获取下一个任务的开始时间
            next_start_time = None
            if isinstance(next_duty, Flight):
                next_start_time = next_duty.std
            elif hasattr(next_duty, 'startTime'):
                next_start_time = next_duty.startTime
            
            # 计算外站过夜天数
            if (current_end_airport and current_end_airport != crew.base and 
                current_end_time and next_start_time):
                
                # 3. 检查过夜机场是否为新增过夜站点
                if current_end_airport not in self.layover_stations_set:
                    new_layover_stations.add(current_end_airport)
                
                # 情况①：值勤日间隔跨零点，计跨零点天数
                if next_start_time.date() > current_end_time.date():
                    overnight_days = (next_start_time.date() - current_end_time.date()).days
                    away_overnight_days += overnight_days
                # 情况③：值勤日间隔不跨零点，计1天
                elif next_start_time.date() == current_end_time.date():
                    # 如果同一天但有足够休息时间，也算过夜
                    rest_hours = (next_start_time - current_end_time).total_seconds() / 3600
                    if rest_hours >= 8:  # 假设8小时以上算过夜
                        away_overnight_days += 1
        
        # 情况④：计划期内任务结束于非基地站点
        if sorted_duties:
            last_duty = sorted_duties[-1]
            last_end_airport = None
            last_end_time = None
            
            if isinstance(last_duty, Flight):
                last_end_airport = last_duty.arriAirport
                last_end_time = last_duty.sta
            elif hasattr(last_duty, 'endTime'):
                last_end_time = last_duty.endTime
                last_end_airport = getattr(last_duty, 'arriAirport', crew.base)
            
            if last_end_airport and last_end_airport != crew.base and last_end_time:
                # 4. 检查计划期结束时的过夜机场是否为新增过夜站点
                if last_end_airport not in self.layover_stations_set:
                    new_layover_stations.add(last_end_airport)
                
                # 假设计划期结束日为最后任务结束日的下一天
                plan_end_date = last_end_time.date() + timedelta(days=1)
                overnight_days = (plan_end_date - last_end_time.date()).days
                away_overnight_days += max(0, overnight_days)
        
        # 3. 计算各项得分
        total_duty_days = len(duty_calendar_days)
        avg_daily_fly_time = total_flight_hours / total_duty_days if total_duty_days > 0 else 0
        
        # 按照赛题公式计算得分（但在列生成中，roster基础成本不包含飞行奖励）
        # 注意：这个方法主要用于最终评分，在列生成过程中飞行奖励通过执行变量单独计算
        fly_time_score = 0.0  # 在列生成中不计算飞行奖励，避免双重计算
        new_layover_penalty = len(new_layover_stations) * self.NEW_LAYOVER_STATION_PENALTY  # 每个新增过夜站点扣分
        away_overnight_penalty = away_overnight_days * self.AWAY_OVERNIGHT_PENALTY  # 每天外站过夜扣分
        positioning_penalty = positioning_count * self.POSITIONING_PENALTY  # 每个置位扣分
        
        # 4. 违规检查（完整实现）
        violation_count = self._check_roster_violations(roster, crew)
        violation_penalty = violation_count * (-10)  # 每次违规扣10分
        
        # 5. 总得分计算（不包含飞行奖励）
        total_score = (fly_time_score + new_layover_penalty + away_overnight_penalty + 
                      positioning_penalty + violation_penalty)
        
        # 转换为成本：得分越高，成本越低
        # 使用负得分作为成本，确保优化目标正确
        return -total_score
    
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
            
            # 统计每个roster的贡献
            for duty in sorted_duties:
                if isinstance(duty, Flight):
                    covered_flight_ids.add(duty.id)
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
            
            # 计算外站过夜天数
            for i in range(len(sorted_duties) - 1):
                current_duty = sorted_duties[i]
                next_duty = sorted_duties[i + 1]
                
                # 获取当前任务的结束地点和时间
                current_end_airport = None
                current_end_time = None
                
                if isinstance(current_duty, Flight):
                    current_end_airport = current_duty.arriAirport
                    current_end_time = current_duty.sta
                elif hasattr(current_duty, 'endTime'):
                    current_end_time = current_duty.endTime
                    current_end_airport = getattr(current_duty, 'arriAirport', crew.base)
                
                # 获取下一个任务的开始时间
                next_start_time = None
                if isinstance(next_duty, Flight):
                    next_start_time = next_duty.std
                elif hasattr(next_duty, 'startTime'):
                    next_start_time = next_duty.startTime
                
                # 计算外站过夜天数
                if (current_end_airport and current_end_airport != crew.base and 
                    current_end_time and next_start_time):
                    
                    # 检查过夜机场是否为新增过夜站点
                    if current_end_airport not in self.layover_stations_set:
                        new_layover_stations.add(current_end_airport)
                    
                    if next_start_time.date() > current_end_time.date():
                        overnight_days = (next_start_time.date() - current_end_time.date()).days
                        away_overnight_days += overnight_days
                    elif next_start_time.date() == current_end_time.date():
                        rest_hours = (next_start_time - current_end_time).total_seconds() / 3600
                        if rest_hours >= 8:
                            away_overnight_days += 1
        
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
    
    def calculate_roster_cost_with_dual_prices(self, roster: Roster, crew: Crew, 
                                             dual_prices: Dict[str, float], 
                                             crew_sigma_dual: float) -> Dict[str, float]:
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
        
        # 1. 计算基础统计信息（不包含飞行奖励，因为飞行奖励现在只通过执行变量获得）
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
        
        # 根据数学模型：计算执行航班的飞行奖励（α·t_f）
        optimization_params = UnifiedConfig.get_optimization_params()
        flight_reward_rate = optimization_params['flight_time_reward']  # α = 100
        flight_reward = 0.0
        
        for duty in sorted_duties:
            if isinstance(duty, Flight):
                # 只有执行航班才能获得飞行奖励
                if not getattr(duty, 'is_positioning', False):
                    flight_reward += flight_reward_rate * (duty.flyTime / 60.0)
        
        # 2. 计算对偶价格收益
        dual_price_total = 0.0
        for duty in roster.duties:
            if isinstance(duty, Flight):
                dual_price_total += dual_prices.get(duty.id, 0.0)
        
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
        
        # 4. 计算外站过夜惩罚
        overnight_penalty = 0.0
        overnight_count = 0
        
        for i in range(len(sorted_duties) - 1):
            current_duty = sorted_duties[i]
            next_duty = sorted_duties[i + 1]
            
            # 获取当前任务的结束地点和时间
            current_end_airport = None
            current_end_time = None
            
            if isinstance(current_duty, Flight):
                current_end_airport = current_duty.arriAirport
                current_end_time = current_duty.sta
            elif hasattr(current_duty, 'endTime'):
                current_end_time = current_duty.endTime
                current_end_airport = getattr(current_duty, 'arriAirport', None)
            
            # 获取下一个任务的开始时间
            next_start_time = None
            if isinstance(next_duty, Flight):
                next_start_time = next_duty.std
            elif hasattr(next_duty, 'startTime'):
                next_start_time = next_duty.startTime
            
            # 检查外站过夜
            if (current_end_airport and current_end_airport != crew.base and 
                current_end_time and next_start_time):
                
                rest_time = next_start_time - current_end_time
                if rest_time >= timedelta(hours=8):  # MIN_REST_HOURS
                    overnight_days = (next_start_time.date() - current_end_time.date()).days
                    if overnight_days > 0:
                        away_overnight_penalty_rate = optimization_params['away_overnight_penalty']
                        overnight_penalty += overnight_days * away_overnight_penalty_rate
                        overnight_count += overnight_days
        
        # 5. 其他成本
        other_costs = 0.0
        for duty in roster.duties:
            if hasattr(duty, 'cost'):
                other_costs += duty.cost
        
        # 6. 计算总成本和reduced cost
        # 最小化问题的reduced cost计算: c_j - π^T A_j
        # c_j = 原始成本 (penalties - flight_reward，负值表示收益)
        # π^T A_j = 对偶价格贡献 (dual_price_total - crew_sigma_dual)
        # 注意：机组约束是≤1的不等式，对偶价格为负，在reduced cost中应该用减法
        # 当reduced_cost < 0时，表示该roster有价值，应该加入主问题
        total_cost = positioning_penalty + overnight_penalty + other_costs - flight_reward
        dual_contribution = dual_price_total - crew_sigma_dual  # 修正：机组对偶价格用减法
        reduced_cost = total_cost - dual_contribution
        
        return {
            'total_cost': total_cost,
            'flight_reward': flight_reward,
            'dual_price_total': dual_price_total,
            'dual_contribution': dual_contribution,
            'positioning_penalty': positioning_penalty,
            'overnight_penalty': overnight_penalty,
            'other_costs': other_costs,
            'crew_sigma_dual': crew_sigma_dual,
            'reduced_cost': reduced_cost,
            'flight_count': flight_count,
            'total_flight_hours': total_flight_hours,
            'duty_days': total_duty_days,
            'avg_daily_flight_hours': avg_daily_flight_hours,
            'positioning_count': positioning_count,
            'overnight_count': overnight_count
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
