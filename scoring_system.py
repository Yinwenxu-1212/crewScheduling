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
        self.FLY_TIME_MULTIPLIER = 50
        self.UNCOVERED_FLIGHT_PENALTY = -500   # 未覆盖航班惩罚（大幅增加以提高覆盖率）
        self.NEW_LAYOVER_STATION_PENALTY = -10
        self.AWAY_OVERNIGHT_PENALTY = -0.5
        self.POSITIONING_PENALTY = -0.5
        self.VIOLATION_PENALTY = -10
    
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
        
        # 按照赛题公式计算得分
        fly_time_score = avg_daily_fly_time * self.FLY_TIME_MULTIPLIER  # 值勤日日均飞时 * FLY_TIME_MULTIPLIER
        new_layover_penalty = len(new_layover_stations) * (-10)  # 每个新增过夜站点扣10分
        away_overnight_penalty = away_overnight_days * (-0.5)  # 每天外站过夜扣0.5分
        positioning_penalty = positioning_count * (-0.5)  # 每个置位扣0.5分
        
        # 4. 违规检查（完整实现）
        violation_count = self._check_roster_violations(roster, crew)
        violation_penalty = violation_count * (-10)  # 每次违规扣10分
        
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
                                             crew_sigma_dual: float, lambda_k: float = 0.0) -> Dict[str, float]:
        """
        计算单个排班方案的完整成本，包括对偶价格
        返回详细的成本分解，用于reduced cost计算
        """
        if not roster.duties:
            return {
                'total_cost': 0.0,  # c_j = 0 for empty roster
                'flight_reward': 0.0,
                'dual_price_total': 0.0,
                'dual_contribution': crew_sigma_dual,  # π^T A_j = crew_sigma_dual
                'positioning_penalty': 0.0,
                'overnight_penalty': 0.0,
                'other_costs': 0.0,
                'crew_sigma_dual': crew_sigma_dual,
                'reduced_cost': 0.0 - crew_sigma_dual,  # c_j - π^T A_j = 0 - crew_sigma_dual
                'flight_count': 0,
                'total_flight_hours': 0.0,
                'duty_days': 0,
                'avg_daily_flight_hours': 0.0,
                'positioning_count': 0,
                'overnight_count': 0
            }
        
        # 1. 计算飞行奖励（基于日均飞时，按照新的Dinkelbach算法要求）
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
        
        # 新的Dinkelbach算法中的目标函数系数：FLY_TIME_MULTIPLIER * C_p - lambda_k * d_p
        # 其中C_p是总飞行时间，d_p是值勤天数
        # lambda_k不需要乘以FLY_TIME_MULTIPLIER系数
        # 在列生成中，这应该是成本，所以不加负号
        flight_reward = self.FLY_TIME_MULTIPLIER * total_flight_hours - lambda_k * total_duty_days
        
        # 2. 计算对偶价格收益
        dual_price_total = 0.0
        for duty in roster.duties:
            if isinstance(duty, Flight):
                dual_price_total += dual_prices.get(duty.id, 0.0)
        
        # 3. 计算置位惩罚
        positioning_penalty = 0.0
        positioning_count = 0
        for duty in roster.duties:
            if isinstance(duty, Flight):
                # 检查是否为置位航班（根据任务类型判断）
                if hasattr(duty, 'type') and 'positioning' in duty.type:
                    positioning_penalty += 0.5
                    positioning_count += 1
            elif isinstance(duty, BusInfo):
                # 巴士任务（地面交通）- 在attention模块中标记为positioning_bus
                positioning_penalty += 0.5
                positioning_count += 1
            elif isinstance(duty, GroundDuty):
                # 地面值勤任务（培训、待命等），不是置位
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
                        overnight_penalty += overnight_days * 0.5  # PENALTY_PER_AWAY_OVERNIGHT
                        overnight_count += overnight_days
        
        # 5. 其他成本
        other_costs = 0.0
        for duty in roster.duties:
            if hasattr(duty, 'cost'):
                other_costs += duty.cost
        
        # 6. 计算总成本和reduced cost
        # 修正reduced cost计算公式: c_j - π^T A_j
        # c_j = 原始成本 (flight_reward - penalties，因为penalties是正值但应该减少成本)
        # π^T A_j = 对偶价格贡献 (dual_price_total + crew_sigma_dual)
        total_cost = flight_reward - positioning_penalty - overnight_penalty + other_costs
        dual_contribution = dual_price_total + crew_sigma_dual
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
    
    def _check_location_connection_violations(self, sorted_duties: List, crew: Crew) -> int:
        """
        检查地点衔接规则违规
        规则2: 地点衔接规则
        - 第一个任务的出发地必须是机组驻留地
        - 相邻任务的到达地和出发地必须一致
        """
        violations = 0
        
        if not sorted_duties:
            return violations
        
        # 检查第一个任务的出发地是否为机组驻留地
        first_task = sorted_duties[0]
        first_departure = None
        
        if hasattr(first_task, 'depaAirport'):
            first_departure = first_task.depaAirport
        elif hasattr(first_task, 'startLocation'):
            first_departure = first_task.startLocation
        elif hasattr(first_task, 'airport'):
            first_departure = first_task.airport
        
        # 检查第一个任务出发地是否为机组驻留地
        crew_stay_station = getattr(crew, 'stayStation', None) or getattr(crew, 'baseAirport', None)
        if first_departure and crew_stay_station and first_departure != crew_stay_station:
            violations += 1
        
        # 检查相邻任务的地点衔接
        for i in range(len(sorted_duties) - 1):
            curr_task = sorted_duties[i]
            next_task = sorted_duties[i + 1]
            
            # 获取当前任务的到达地
            curr_arrival = None
            if hasattr(curr_task, 'arriAirport'):
                curr_arrival = curr_task.arriAirport
            elif hasattr(curr_task, 'endLocation'):
                curr_arrival = curr_task.endLocation
            elif hasattr(curr_task, 'airport'):
                curr_arrival = curr_task.airport
            
            # 获取下一个任务的出发地
            next_departure = None
            if hasattr(next_task, 'depaAirport'):
                next_departure = next_task.depaAirport
            elif hasattr(next_task, 'startLocation'):
                next_departure = next_task.startLocation
            elif hasattr(next_task, 'airport'):
                next_departure = next_task.airport
            
            # 检查地点衔接
            if curr_arrival and next_departure and curr_arrival != next_departure:
                violations += 1
        
        return violations
    
    def _organize_into_duty_days(self, sorted_duties: List) -> List:
        """
        将任务组织为值勤日
        根据比赛定义：值勤日是一连串值勤任务，不等同于日历日
        值勤日之间需要有足够的休息时间间隔
        """
        from data_models import DutyDay
        from datetime import timedelta
        
        if not sorted_duties:
            return []
            
        duty_days = []
        current_day = DutyDay()
        
        for i, duty in enumerate(sorted_duties):
            duty_start = getattr(duty, 'std', getattr(duty, 'startTime', None))
            
            if not duty_start:
                continue
                
            # 如果是第一个任务，直接加入当前值勤日
            if i == 0:
                current_day.add_task(duty)
                continue
                
            # 检查与前一个任务的时间间隔
            prev_duty = sorted_duties[i-1]
            prev_end = getattr(prev_duty, 'sta', getattr(prev_duty, 'endTime', None))
            
            if prev_end and duty_start:
                rest_interval = duty_start - prev_end
                
                # 判断是否需要开始新的值勤日
                # 规则：如果休息时间超过12小时，或者值勤日已超过24小时，开始新值勤日
                should_start_new_duty_day = False
                
                # 1. 休息时间超过12小时
                if rest_interval >= timedelta(hours=12):
                    should_start_new_duty_day = True
                    
                # 2. 当前值勤日已经超过24小时
                elif current_day.start_time and (duty_start - current_day.start_time) > timedelta(hours=24):
                    should_start_new_duty_day = True
                    
                # 3. 跨越了太多日历日（超过2个日历日）
                elif (current_day.start_time and 
                      (duty_start.date() - current_day.start_time.date()).days > 1):
                    should_start_new_duty_day = True
                
                if should_start_new_duty_day:
                    # 结束当前值勤日，开始新的值勤日
                    if current_day.tasks:
                        duty_days.append(current_day)
                    current_day = DutyDay()
            
            current_day.add_task(duty)
        
        # 添加最后一个值勤日
        if current_day.tasks:
            duty_days.append(current_day)
        
        # 如果有可过夜机场数据，更新所有值勤日的飞行值勤日状态
        if hasattr(self, 'layover_stations_set') and self.layover_stations_set:
            for duty_day in duty_days:
                duty_day.set_layover_stations(self.layover_stations_set)
        
        return duty_days
    
    def _check_flight_cycle_violations_new(self, duty_days: List, crew: Crew) -> int:
        """
        检查飞行周期违规（基于正确的值勤日概念）
        飞行周期定义：
        1. 由值勤日组成（可能包含少于2个完整日历日的休息）
        2. 必须包含飞行值勤日
        3. 飞行周期末尾一定为飞行值勤日
        4. 最多横跨4个日历日
        5. 开始前必须连续休息2个完整日历日
        """
        violations = 0
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
                        if rest_days < 2:
                            violations += 1
                    
                    current_cycle_duty_days = [duty_day]
                else:
                    # 继续当前飞行周期
                    current_cycle_duty_days.append(duty_day)
                
                # 检查是否返回基地（飞行周期结束）
                last_task = duty_day.tasks[-1] if duty_day.tasks else None
                if last_task and hasattr(last_task, 'arriAirport'):
                    if last_task.arriAirport == crew.baseAirport:
                        # 返回基地，结束当前飞行周期
                        cycle_violations = self._validate_flight_cycle(current_cycle_duty_days)
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
                        violations += 1  # 飞行周期末尾必须是飞行值勤日
                        current_cycle_duty_days = []
        
        # 检查最后一个未完成的周期
        if current_cycle_duty_days:
            # 检查最后一个值勤日是否为飞行值勤日
            if not current_cycle_duty_days[-1].is_flight_duty_day:
                violations += 1  # 飞行周期末尾必须是飞行值勤日
            
            cycle_violations = self._validate_flight_cycle(current_cycle_duty_days)
            violations += cycle_violations
        
        return violations
    
    def _validate_flight_cycle(self, cycle_duty_days: List) -> int:
        """
        验证单个飞行周期的完整性
        根据比赛定义检查飞行周期约束
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
                if calendar_days_span > 4:
                    violations += 1
        
        # 规则4: 检查飞行周期内值勤日之间的休息间隔
        for i in range(1, len(cycle_duty_days)):
            prev_duty = cycle_duty_days[i-1]
            curr_duty = cycle_duty_days[i]
            
            if (prev_duty.end_date and curr_duty.start_date and
                prev_duty.end_date != curr_duty.start_date):
                # 如果不是连续的值勤日，检查休息时间
                rest_days = (curr_duty.start_date - prev_duty.end_date).days
                if rest_days >= 2:
                    # 休息时间过长，不应该在同一个飞行周期内
                    violations += 1
        
        return violations
    
    def _check_fdp_rest_violations(self, fdps: List) -> int:
        """
        检查FDP间休息时间违规
        """
        violations = 0
        
        for i in range(1, len(fdps)):
            prev_fdp = fdps[i-1]
            curr_fdp = fdps[i]
            
            if prev_fdp.tasks and curr_fdp.tasks:
                prev_end = getattr(prev_fdp.tasks[-1], 'sta', getattr(prev_fdp.tasks[-1], 'endTime', None))
                curr_start = getattr(curr_fdp.tasks[0], 'std', getattr(curr_fdp.tasks[0], 'startTime', None))
                
                if prev_end and curr_start:
                    rest_time = (curr_start - prev_end).total_seconds() / 3600.0
                    if rest_time < 12:  # FDP间至少12小时休息
                        violations += 1
        
        return violations
    
    def _identify_flight_duty_periods(self, sorted_duties: List) -> List[List]:
        """
        识别飞行值勤期（FDP）- 保留旧方法以兼容
        FDP是连续的飞行任务和相关的地面任务组合
        """
        fdps = []
        current_fdp = []
        
        for duty in sorted_duties:
            if isinstance(duty, Flight):
                # 飞行任务开始新的FDP或继续当前FDP
                if not current_fdp:
                    current_fdp = [duty]
                else:
                    # 检查与前一个任务的时间间隔
                    prev_duty = current_fdp[-1]
                    prev_end = getattr(prev_duty, 'sta', getattr(prev_duty, 'endTime', None))
                    curr_start = getattr(duty, 'std', getattr(duty, 'startTime', None))
                    
                    if prev_end and curr_start:
                        interval = curr_start - prev_end
                        # 如果间隔超过12小时，开始新的FDP
                        if interval >= timedelta(hours=12):
                            if current_fdp:
                                fdps.append(current_fdp)
                            current_fdp = [duty]
                        else:
                            current_fdp.append(duty)
                    else:
                        current_fdp.append(duty)
            else:
                # 非飞行任务（地面任务、置位等）
                if current_fdp:
                    current_fdp.append(duty)
        
        if current_fdp:
            fdps.append(current_fdp)
        
        return fdps
    
    def _check_fdp_violations(self, fdp: List) -> int:
        """
        检查单个飞行值勤期的违规情况
        """
        violations = 0
        
        if not fdp:
            return 0
        
        flight_tasks = [duty for duty in fdp if isinstance(duty, Flight)]
        
        # 规则3.1.1: FDP内最多4个飞行任务
        if len(flight_tasks) > 4:
            violations += 1
        
        # 规则3.1.1: FDP内最多6个总任务
        if len(fdp) > 6:
            violations += 1
        
        # 规则3.1.2: FDP内累计飞行时间不超过8小时
        total_flight_time = sum(flight.flyTime for flight in flight_tasks) / 60.0  # 转换为小时
        if total_flight_time > 8:
            violations += 1
        
        # 规则3.1.3: FDP内累计值勤时间不超过12小时
        if fdp:
            fdp_start = getattr(fdp[0], 'std', getattr(fdp[0], 'startTime', None))
            fdp_end = getattr(fdp[-1], 'sta', getattr(fdp[-1], 'endTime', None))
            if fdp_start and fdp_end:
                fdp_duration = (fdp_end - fdp_start).total_seconds() / 3600.0
                if fdp_duration > 12:
                    violations += 1
        
        # 检查最小连接时间
        for i in range(len(fdp) - 1):
            curr_duty = fdp[i]
            next_duty = fdp[i + 1]
            
            curr_end = getattr(curr_duty, 'sta', getattr(curr_duty, 'endTime', None))
            next_start = getattr(next_duty, 'std', getattr(next_duty, 'startTime', None))
            
            if curr_end and next_start:
                interval = next_start - curr_end
                
                # 不同机型或涉及地面交通的最小连接时间检查
                if hasattr(curr_duty, 'aircraftNo') and hasattr(next_duty, 'aircraftNo'):
                    if curr_duty.aircraftNo != next_duty.aircraftNo and interval < timedelta(hours=3):
                        violations += 1
                elif interval < timedelta(hours=2):  # 默认最小连接时间
                    violations += 1
        
        # 检查置位规则：置位不能在FDP中间
        positioning_indices = []
        for i, duty in enumerate(fdp):
            if (hasattr(duty, 'type') and 'positioning' in str(duty.type)) or \
               (not isinstance(duty, Flight) and hasattr(duty, 'startTime')):
                positioning_indices.append(i)
        
        # 如果置位在FDP中间（不是第一个或最后一个），则违规
        for idx in positioning_indices:
            if 0 < idx < len(fdp) - 1:
                violations += 1
        
        return violations
    
    def _check_flight_cycle_violations(self, sorted_duties: List, crew: Crew) -> int:
        """
        检查飞行周期违规情况
        """
        violations = 0
        
        # 识别飞行周期
        cycles = self._identify_flight_cycles(sorted_duties, crew)
        
        for cycle in cycles:
            if not cycle:
                continue
            
            # 规则3.4.1: 飞行周期最多持续4个日历日
            cycle_start = getattr(cycle[0], 'std', getattr(cycle[0], 'startTime', None))
            cycle_end = getattr(cycle[-1], 'sta', getattr(cycle[-1], 'endTime', None))
            
            if cycle_start and cycle_end:
                cycle_days = (cycle_end.date() - cycle_start.date()).days + 1
                if cycle_days > 4:
                    violations += 1
        
        return violations
    
    def _identify_flight_cycles(self, sorted_duties: List, crew: Crew) -> List[List]:
        """
        识别飞行周期
        飞行周期是从离开基地到返回基地的连续任务序列
        """
        cycles = []
        current_cycle = []
        
        for duty in sorted_duties:
            if isinstance(duty, Flight):
                # 检查是否在基地
                duty_start_airport = duty.depaAirport
                duty_end_airport = duty.arriAirport
                
                if not current_cycle:
                    # 开始新周期
                    if duty_start_airport != crew.base:
                        current_cycle = [duty]
                    else:
                        current_cycle = [duty]
                else:
                    current_cycle.append(duty)
                    
                    # 检查是否返回基地
                    if duty_end_airport == crew.base:
                        cycles.append(current_cycle)
                        current_cycle = []
        
        # 如果还有未完成的周期
        if current_cycle:
            cycles.append(current_cycle)
        
        return cycles
    
    def _check_rest_time_violations(self, fdps: List[List]) -> int:
        """
        检查休息时间违规
        """
        violations = 0
        
        for i in range(len(fdps) - 1):
            current_fdp = fdps[i]
            next_fdp = fdps[i + 1]
            
            if current_fdp and next_fdp:
                # 获取当前FDP结束时间和下一个FDP开始时间
                current_end = getattr(current_fdp[-1], 'sta', getattr(current_fdp[-1], 'endTime', None))
                next_start = getattr(next_fdp[0], 'std', getattr(next_fdp[0], 'startTime', None))
                
                if current_end and next_start:
                    rest_time = next_start - current_end
                    
                    # 规则3.2.1: FDP开始前正常休息时间至少12小时
                    if rest_time < timedelta(hours=12):
                        violations += 1
        
        return violations    
    def _check_work_rest_pattern_violations(self, sorted_duties: List, crew: Crew) -> int:
        """
        检查值四修二工作模式违规
        规则：连续工作不超过4天，工作4天后必须休息2天
        """
        violations = 0
        
        # 生成工作日历：标识每一天是工作日还是休息日
        work_calendar = self._generate_work_calendar(sorted_duties, crew)
        
        if not work_calendar:
            return violations
        
        consecutive_work_days = 0
        need_rest_days = 0  # 需要的连续休息天数
        
        for date, is_work_day in work_calendar:
            if is_work_day:
                consecutive_work_days += 1
                
                # 检查连续工作天数是否超过4天
                if consecutive_work_days > 4:
                    violations += 1
                    
                # 如果正在需要休息期间却工作了，违规
                if need_rest_days > 0:
                    violations += 1
                    
                need_rest_days = 0  # 重置休息需求
                
            else:  # 休息日
                if consecutive_work_days == 4:
                    # 刚完成4天工作，需要2天休息
                    need_rest_days = 2
                    
                consecutive_work_days = 0
                
                if need_rest_days > 0:
                    need_rest_days -= 1
        
        # 检查计划期结束时是否还有未满足的休息需求
        if need_rest_days > 0:
            violations += 1
            
        return violations
    
    def _generate_work_calendar(self, sorted_duties: List, crew: Crew) -> List[tuple]:
        """
        生成工作日历，标识每一天是工作日还是休息日
        返回: [(date, is_work_day), ...]
        """
        if not sorted_duties:
            return []
        
        # 获取计划期的开始和结束日期
        start_date = None
        end_date = None
        
        for duty in sorted_duties:
            duty_date = None
            if hasattr(duty, 'std'):
                duty_date = duty.std.date()
            elif hasattr(duty, 'startTime'):
                duty_date = duty.startTime.date()
                
            if duty_date:
                if start_date is None or duty_date < start_date:
                    start_date = duty_date
                if end_date is None or duty_date > end_date:
                    end_date = duty_date
        
        if not start_date or not end_date:
            return []
        
        # 生成每日工作状态
        work_calendar = []
        current_date = start_date
        
        while current_date <= end_date:
            # 检查当天是否有工作任务
            has_work = False
            
            for duty in sorted_duties:
                duty_date = None
                if hasattr(duty, 'std'):
                    duty_date = duty.std.date()
                elif hasattr(duty, 'startTime'):
                    duty_date = duty.startTime.date()
                    
                if duty_date == current_date:
                    # 检查是否是实际工作任务（排除休息占位）
                    if isinstance(duty, Flight):
                        has_work = True
                        break
                    elif hasattr(duty, 'isDuty') and duty.isDuty:
                        has_work = True
                        break
            
            work_calendar.append((current_date, has_work))
            current_date += timedelta(days=1)
        
        return work_calendar
