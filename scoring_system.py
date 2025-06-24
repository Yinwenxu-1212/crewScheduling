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
from data_models import Flight, Roster, Crew, LayoverStation

class ScoringSystem:
    def __init__(self, flights: List[Flight], crews: List[Crew], layover_stations):
        self.flights = flights
        self.crews = crews
        # Handle both List[LayoverStation] and set of airport strings
        if isinstance(layover_stations, set):
            self.layover_stations_set = layover_stations
        else:
            self.layover_stations_set = {station.airport for station in layover_stations}
        
        # 评分参数
        self.FLY_TIME_MULTIPLIER = 1000
        self.UNCOVERED_FLIGHT_PENALTY = -5
        self.NEW_LAYOVER_STATION_PENALTY = -10
        self.AWAY_OVERNIGHT_PENALTY = -0.5
        self.POSITIONING_PENALTY = -0.5
        self.VIOLATION_PENALTY = -10
    
    def calculate_roster_score(self, roster: Roster, crew: Crew) -> float:
        """
        计算单个排班方案的得分，严格按照赛题评分公式
        返回正值作为成本（用于最小化目标函数）
        
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
                # 起点或终点为不可过夜机场时，记为新增可过夜机场
                if duty.depaAirport not in self.layover_stations_set:
                    new_layover_stations.add(duty.depaAirport)
                if duty.arriAirport not in self.layover_stations_set:
                    new_layover_stations.add(duty.arriAirport)
            
            # 计算置位次数（包括飞行置位和地面置位）
            elif hasattr(duty, 'startTime') and hasattr(duty, 'endTime'):
                positioning_count += 1
                
                # 置位任务也可能跨日历日
                start_date = duty.startTime.date()
                end_date = duty.endTime.date()
                current_date = start_date
                while current_date <= end_date:
                    duty_calendar_days.add(current_date)
                    current_date += timedelta(days=1)
        
        # 2. 计算外站过夜天数
        # 检查历史停留机场（计划期开始前的外站过夜）
        if hasattr(crew, 'stayStation') and crew.stayStation != crew.base:
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
                # 假设计划期结束日为最后任务结束日的下一天
                plan_end_date = last_end_time.date() + timedelta(days=1)
                overnight_days = (plan_end_date - last_end_time.date()).days
                away_overnight_days += max(0, overnight_days)
        
        # 3. 计算各项得分
        total_duty_days = len(duty_calendar_days)
        avg_daily_fly_time = total_flight_hours / total_duty_days if total_duty_days > 0 else 0
        
        # 按照赛题公式计算得分
        fly_time_score = avg_daily_fly_time * 1000  # 值勤日日均飞时 * 1000
        new_layover_penalty = len(new_layover_stations) * (-10)  # 每个新增过夜站点扣10分
        away_overnight_penalty = away_overnight_days * (-0.5)  # 每天外站过夜扣0.5分
        positioning_penalty = positioning_count * (-0.5)  # 每个置位扣0.5分
        
        # 4. 违规检查（简化处理，暂时设为0）
        violation_penalty = 0 * (-10)  # TODO: 实现详细的违规检查，每次违规扣10分
        
        # 5. 总得分计算
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
                
                # 计算置位次数
                elif hasattr(duty, 'startTime') and hasattr(duty, 'endTime'):
                    positioning_count += 1
                    
                    # 置位任务的日历日
                    start_date = duty.startTime.date()
                    end_date = duty.endTime.date()
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
        
        fly_time_score = avg_daily_fly_time * 1000  # 值勤日日均飞时 * 1000
        uncovered_penalty = uncovered_flights * (-5)  # 每个未覆盖航班扣5分
        new_layover_penalty = len(new_layover_stations) * (-10)  # 每个新增过夜站点扣10分
        away_overnight_penalty = away_overnight_days * (-0.5)  # 每天外站过夜扣0.5分
        positioning_penalty = positioning_count * (-0.5)  # 每个置位扣0.5分
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
                'total_cost': -crew_sigma_dual,
                'flight_reward': 0.0,
                'dual_price_total': 0.0,
                'positioning_penalty': 0.0,
                'overnight_penalty': 0.0,
                'other_costs': 0.0,
                'crew_sigma_dual': crew_sigma_dual,
                'reduced_cost': -crew_sigma_dual,
                'flight_count': 0,
                'total_flight_hours': 0.0,
                'duty_days': 0,
                'avg_daily_flight_hours': 0.0,
                'positioning_count': 0,
                'overnight_count': 0
            }
        
        # 1. 计算飞行奖励（基于日均飞时，严格按照赛题公式）
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
        
        total_duty_days = len(duty_calendar_days)
        avg_daily_flight_hours = total_flight_hours / total_duty_days if total_duty_days > 0 else 0.0
        flight_reward = -(avg_daily_flight_hours * 1000)  # 负值表示奖励，与赛题公式一致
        
        # 2. 计算对偶价格收益
        dual_price_total = 0.0
        for duty in roster.duties:
            if isinstance(duty, Flight):
                dual_price_total += dual_prices.get(duty.id, 0.0)
        
        # 3. 计算置位惩罚
        positioning_penalty = 0.0
        positioning_count = 0
        for duty in roster.duties:
            if hasattr(duty, 'type') and 'positioning' in str(duty.type):
                positioning_penalty += 0.5  # PENALTY_PER_POSITIONING
                positioning_count += 1
            elif not isinstance(duty, Flight):  # 假设非Flight任务为置位
                positioning_penalty += 0.5
                positioning_count += 1
        
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
                        overnight_penalty += overnight_days * 0.5  # PENALTY_PER_AWAY_OVERNIGHT
                        overnight_count += overnight_days
        
        # 5. 其他成本
        other_costs = 0.0
        for duty in roster.duties:
            if hasattr(duty, 'cost'):
                other_costs += duty.cost
        
        # 6. 计算总成本和reduced cost
        total_cost = flight_reward - dual_price_total + positioning_penalty + overnight_penalty + other_costs
        reduced_cost = total_cost - crew_sigma_dual
        
        return {
            'total_cost': total_cost,
            'flight_reward': flight_reward,
            'dual_price_total': dual_price_total,
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