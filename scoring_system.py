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
        计算单个排班方案的得分
        返回正值作为成本（用于最小化目标函数）
        """
        if not roster.duties:
            return 0.0
        
        # 1. 计算飞行时间和值勤日
        total_flight_hours = 0.0
        duty_calendar_days = set()
        positioning_count = 0
        away_overnight_days = 0
        new_layover_stations = set()
        
        # 按时间排序任务
        sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        for i, duty in enumerate(sorted_duties):
            # 计算飞行时间
            if isinstance(duty, Flight):
                total_flight_hours += duty.flyTime / 60.0  # 转换为小时
                
                # 计算值勤日历日
                start_date = duty.std.date()
                end_date = duty.sta.date()
                for day in range((end_date - start_date).days + 1):
                    duty_calendar_days.add(start_date + timedelta(days=day))
                
                # 检查新增过夜站点
                if duty.arriAirport not in self.layover_stations_set:
                    new_layover_stations.add(duty.arriAirport)
                if duty.depaAirport not in self.layover_stations_set:
                    new_layover_stations.add(duty.depaAirport)
            
            # 计算置位次数（简化处理，假设BusInfo为置位）
            elif hasattr(duty, 'startTime') and hasattr(duty, 'endTime'):
                positioning_count += 1
                
                # 计算值勤日历日
                start_date = duty.startTime.date()
                end_date = duty.endTime.date()
                for day in range((end_date - start_date).days + 1):
                    duty_calendar_days.add(start_date + timedelta(days=day))
        
        # 2. 计算外站过夜天数
        for i in range(len(sorted_duties) - 1):
            current_duty = sorted_duties[i]
            next_duty = sorted_duties[i + 1]
            
            # 获取当前任务的结束地点
            current_end_airport = None
            if isinstance(current_duty, Flight):
                current_end_airport = current_duty.arriAirport
            elif hasattr(current_duty, 'airport'):
                current_end_airport = current_duty.airport
            
            # 如果结束地点不是基地，计算过夜天数
            if current_end_airport and current_end_airport != crew.base:
                current_end_time = getattr(current_duty, 'sta', getattr(current_duty, 'endTime', None))
                next_start_time = getattr(next_duty, 'std', getattr(next_duty, 'startTime', None))
                
                if current_end_time and next_start_time:
                    overnight_days = (next_start_time.date() - current_end_time.date()).days
                    if overnight_days > 0:
                        away_overnight_days += overnight_days
        
        # 3. 计算各项得分
        total_duty_days = len(duty_calendar_days)
        avg_daily_fly_time = total_flight_hours / total_duty_days if total_duty_days > 0 else 0
        
        fly_time_score = avg_daily_fly_time * self.FLY_TIME_MULTIPLIER
        new_layover_penalty = len(new_layover_stations) * self.NEW_LAYOVER_STATION_PENALTY
        away_overnight_penalty = away_overnight_days * self.AWAY_OVERNIGHT_PENALTY
        positioning_penalty = positioning_count * self.POSITIONING_PENALTY
        
        # 4. 违规检查（简化处理，暂时设为0）
        violation_penalty = 0  # TODO: 实现详细的违规检查
        
        # 5. 总得分计算
        total_score = fly_time_score + new_layover_penalty + away_overnight_penalty + positioning_penalty + violation_penalty
        
        # 修改：返回正值作为成本
        # 如果total_score是负数（表示惩罚多于奖励），取绝对值作为成本
        # 如果total_score是正数（表示奖励多于惩罚），用一个基准值减去它作为成本
        if total_score < 0:
            return abs(total_score)  # 负分直接转为正成本
        else:
            # 正分转为成本：基准值减去得分
            return max(0, 10000 - total_score)  # 10000是一个基准值
    
    def calculate_total_score(self, rosters: List[Roster]) -> Dict[str, float]:
        """
        计算所有排班方案的总得分
        返回各项得分的详细分解
        """
        total_flight_hours = 0.0
        total_duty_days = 0
        uncovered_flights = len(self.flights)
        new_layover_stations = set()
        away_overnight_days = 0
        positioning_count = 0
        violation_count = 0
        
        covered_flight_ids = set()
        
        for roster in rosters:
            crew = next((c for c in self.crews if c.crewId == roster.crew_id), None)
            if not crew:
                continue
            
            # 统计覆盖的航班
            for duty in roster.duties:
                if isinstance(duty, Flight):
                    covered_flight_ids.add(duty.id)
                    total_flight_hours += duty.flyTime / 60.0
        
        uncovered_flights = len(self.flights) - len(covered_flight_ids)
        
        # 计算各项得分
        avg_daily_fly_time = total_flight_hours / total_duty_days if total_duty_days > 0 else 0
        fly_time_score = avg_daily_fly_time * self.FLY_TIME_MULTIPLIER
        uncovered_penalty = uncovered_flights * self.UNCOVERED_FLIGHT_PENALTY
        new_layover_penalty = len(new_layover_stations) * self.NEW_LAYOVER_STATION_PENALTY
        away_overnight_penalty = away_overnight_days * self.AWAY_OVERNIGHT_PENALTY
        positioning_penalty = positioning_count * self.POSITIONING_PENALTY
        violation_penalty = violation_count * self.VIOLATION_PENALTY
        
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