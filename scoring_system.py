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
        self.UNCOVERED_FLIGHT_PENALTY = -5   # 未覆盖航班惩罚
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
        fly_time_score = avg_daily_fly_time * 1000  # 值勤日日均飞时 * 1000
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
                                             crew_sigma_dual: float, lambda_k: float = 0.0) -> Dict[str, float]:
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
        
        # 新的Dinkelbach算法中的目标函数系数：1000 * C_p - lambda_k * d_p
        # 其中C_p是总飞行时间，d_p是值勤天数
        # lambda_k不需要乘以1000系数
        flight_reward = -(1000 * total_flight_hours - lambda_k * total_duty_days)  # 负值表示奖励
        
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
    
    def _check_roster_violations(self, roster: Roster, crew: Crew) -> int:
        """
        检查排班方案的违规情况，返回违规次数
        基于竞赛规则实现完整的违规检查
        """
        if not roster.duties:
            return 0
        
        violations = 0
        
        # 按时间排序任务
        sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
        
        # 识别飞行值勤期（FDP）
        fdps = self._identify_flight_duty_periods(sorted_duties)
        
        # 检查每个FDP的违规情况
        total_flight_duty_time = 0.0
        for fdp in fdps:
            fdp_violations = self._check_fdp_violations(fdp)
            violations += fdp_violations
            
            # 计算FDP的飞行值勤时间
            if fdp:
                fdp_start = getattr(fdp[0], 'std', getattr(fdp[0], 'startTime', None))
                fdp_end = getattr(fdp[-1], 'sta', getattr(fdp[-1], 'endTime', None))
                if fdp_start and fdp_end:
                    fdp_duration = (fdp_end - fdp_start).total_seconds() / 3600.0
                    total_flight_duty_time += fdp_duration
        
        # 检查飞行周期违规
        cycle_violations = self._check_flight_cycle_violations(sorted_duties, crew)
        violations += cycle_violations
        
        # 检查总飞行值勤时间限制（规则3.5）
        if total_flight_duty_time > 60:  # 60小时限制
            violations += 1
        
        # 检查休息时间违规
        rest_violations = self._check_rest_time_violations(fdps)
        violations += rest_violations
        
        # 检查值四修二工作模式违规（新增）
        work_pattern_violations = self._check_work_rest_pattern_violations(sorted_duties, crew)
        violations += work_pattern_violations
        
        return violations
    
    def _identify_flight_duty_periods(self, sorted_duties: List) -> List[List]:
        """
        识别飞行值勤期（FDP）
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
