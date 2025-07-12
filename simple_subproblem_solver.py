#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化的子问题求解器
用于分支定价算法中当attention模块不可用时的备用方案
"""

from typing import Dict, List, Optional
from data_models import Roster, Crew, Flight, DutyDay
from datetime import timedelta
import random


def solve_subproblem_simple(
    crew: Crew,
    pi_duals: Dict[str, float],
    sigma_duals: Dict[str, float], 
    ground_duty_duals: Dict[str, float],
    flights: List[Flight],
    ground_duties: List,
    bus_info: Dict,
    crew_leg_match_dict: Dict,
    layover_stations: List[str],
    branching_constraints: Optional[Dict] = None
) -> Optional[Roster]:
    """
    简化的子问题求解器
    使用贪心算法生成roster
    """
    
    # 获取机组可执行的航班
    eligible_flight_ids = crew_leg_match_dict.get(crew.crewId, [])
    if not eligible_flight_ids:
        return None
    
    # 筛选可执行航班
    eligible_flights = [f for f in flights if f.id in eligible_flight_ids]
    if not eligible_flights:
        return None
    
    # 计算每个航班的"价值"（考虑对偶价格）
    flight_values = []
    for flight in eligible_flights:
        # 基础价值 = 对偶价格 - 成本
        dual_price = pi_duals.get(flight.id, 0)
        flight_time_hours = (flight.sta - flight.std).total_seconds() / 3600
        base_cost = -flight_time_hours * 5  # 简化的飞行时间奖励
        
        value = dual_price + base_cost
        flight_values.append((flight, value))
    
    # 按价值排序
    flight_values.sort(key=lambda x: x[1], reverse=True)
    
    # 贪心构造roster
    roster_duties = []
    current_time = None
    current_airport = crew.base
    
    for flight, value in flight_values:
        # 检查是否可以添加这个航班
        if current_time is None or _can_add_flight(
            current_time, current_airport, flight, roster_duties
        ):
            # 创建duty
            duty = DutyDay()
            duty.id = flight.id
            duty.flightNo = flight.flightNo
            duty.deptAirport = flight.depaAirport
            duty.arrAirport = flight.arriAirport
            duty.startTime = flight.std
            duty.endTime = flight.sta
            duty.is_positioning = False
            
            roster_duties.append(duty)
            current_time = flight.sta
            current_airport = flight.arriAirport
            
            # 简单的停止条件
            if len(roster_duties) >= 10:  # 最多10个任务
                break
    
    if not roster_duties:
        return None
    
    # 创建roster
    roster = Roster(crew_id=crew.crewId, duties=roster_duties, cost=0.0)
    
    # 计算reduced cost
    roster_cost = _calculate_roster_cost(roster, pi_duals, sigma_duals)
    crew_dual = sigma_duals.get(crew.crewId, 0)
    roster.reduced_cost = roster_cost - crew_dual
    
    # 只返回负reduced cost的roster
    if roster.reduced_cost < -1e-6:
        return roster
    else:
        return None


def _can_add_flight(current_time, current_airport, flight, existing_duties):
    """检查是否可以添加航班"""
    # 检查机场连接
    if current_airport != flight.depaAirport:
        return False
    
    # 检查时间间隔（至少40分钟）
    min_gap = timedelta(minutes=40)
    if current_time + min_gap > flight.std:
        return False
    
    # 检查最大间隔（不超过4小时）
    max_gap = timedelta(hours=4)
    if current_time + max_gap < flight.std:
        return False
    
    return True


def _calculate_roster_cost(roster, pi_duals, sigma_duals):
    """计算roster的成本"""
    cost = 0
    
    # 飞行时间奖励
    for duty in roster.duties:
        if hasattr(duty, 'flightNo'):
            flight_hours = (duty.endTime - duty.startTime).total_seconds() / 3600
            cost -= flight_hours * 5  # 奖励
            
            # 减去对偶价格（覆盖航班的收益）
            cost -= pi_duals.get(duty.id, 0)
    
    # 简化的其他成本
    cost += len(roster.duties) * 10  # 每个任务的基础成本
    
    return cost
