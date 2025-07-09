# file: data_models.py
# Final version based on official column names from a-T-集.xlsx

from datetime import datetime
from typing import List, Any
import pandas as pd

class Flight:
    """Represents a flight segment. Columns from flight.csv."""
    def __init__(self, id, depaAirport, arriAirport, std, sta, fleet, aircraftNo, flyTime, flightNo=None):
        self.id = str(id).strip() if pd.notna(id) else None
        # 新数据中id就是原来的flightNo
        self.flightNo = flightNo if flightNo is not None else self.id
        self.depaAirport = depaAirport
        self.arriAirport = arriAirport
        
        # 支持两种日期格式：新格式 '2025-05-06 08:00:00' 和旧格式 '2025/5/1 10:20'
        try:
            self.std = datetime.strptime(std, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.std = datetime.strptime(std, '%Y/%m/%d %H:%M')
        
        try:
            self.sta = datetime.strptime(sta, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.sta = datetime.strptime(sta, '%Y/%m/%d %H:%M')
            
        self.fleet = fleet
        self.aircraftNo = aircraftNo
        self.flyTime = int(flyTime)
        self.cost = self.flyTime 

    def __repr__(self):
        return (f"Flight(ID: {self.id}, {self.depaAirport} -> {self.arriAirport}, "
                f"STD: {self.std.strftime('%y/%m/%d %H:%M')}, STA: {self.sta.strftime('%y/%m/%d %H:%M')})")

class Crew:
    """Represents a crew member. Columns from crew.csv."""
    def __init__(self, crewId, base, stayStation):
        self.crewId = str(crewId).strip() if pd.notna(crewId) else None
        self.base = base
        self.stayStation = stayStation

    def __repr__(self):
        return f"Crew(ID: {self.crewId}, Base: {self.base})"

class GroundDuty:
    """Represents a ground duty. Columns from groundDuty.csv."""
    def __init__(self, id, crewId, startTime, endTime, airport, isDuty):
        self.id = str(id).strip() if pd.notna(id) else None
        self.crewId = str(crewId).strip() if pd.notna(crewId) else None
        self.isDuty = isDuty
        
        # 支持两种日期格式
        try:
            self.startTime = datetime.strptime(startTime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.startTime = datetime.strptime(startTime, '%Y/%m/%d %H:%M')
            
        try:
            self.endTime = datetime.strptime(endTime, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.endTime = datetime.strptime(endTime, '%Y/%m/%d %H:%M')
            
        self.airport = airport

    def __repr__(self):
        duty_status = "Duty" if self.isDuty else "Rest"
        return (f"GroundDuty(ID: {self.id}, Crew: {self.crewId}, "
                f"Status: {duty_status}, Start: {self.startTime}, End: {self.endTime})")

class BusInfo:
    """Represents ground transportation. Columns from bus.csv."""
    def __init__(self, id, depaAirport, arriAirport, td, ta):
        self.id = id
        self.depaAirport = depaAirport
        self.arriAirport = arriAirport
        
        # 支持两种日期格式
        try:
            self.td = datetime.strptime(td, '%Y-%m-%d %H:%M:%S')
            self.startTime = datetime.strptime(td, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.td = datetime.strptime(td, '%Y/%m/%d %H:%M')
            self.startTime = datetime.strptime(td, '%Y/%m/%d %H:%M')
            
        try:
            self.ta = datetime.strptime(ta, '%Y-%m-%d %H:%M:%S')
            self.endTime = datetime.strptime(ta, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            self.ta = datetime.strptime(ta, '%Y/%m/%d %H:%M')
            self.endTime = datetime.strptime(ta, '%Y/%m/%d %H:%M')
            
        self.cost = 0

    def __repr__(self):
        return f"Bus(Dep: {self.depaAirport}, Arr: {self.arriAirport}, Time: {self.startTime} -> {self.endTime})"

class FlightDutyPeriod:
    """飞行值勤日(FDP) - 必须包含飞行任务的值勤期"""
    def __init__(self):
        self.tasks = []  # 飞行和置位任务列表
        self.start_time = None  # 第一个任务开始时间
        self.end_time = None    # 最后一个飞行任务结束时间
        self.total_flight_time = 0  # 累计飞行时间(分钟)
        self.duty_time = 0      # 值勤时间(分钟)
        self.has_flight = False # 是否包含飞行任务
        self.flight_count = 0   # 飞行任务数量
        self.total_task_count = 0  # 总任务数量
        
    def add_task(self, task):
        """添加任务到FDP"""
        self.tasks.append(task)
        self.total_task_count += 1
        
        # 更新开始时间
        task_start = getattr(task, 'std', getattr(task, 'startTime', None))
        if self.start_time is None or (task_start and task_start < self.start_time):
            self.start_time = task_start
            
        # 如果是飞行任务
        if isinstance(task, Flight):
            self.has_flight = True
            self.flight_count += 1
            self.total_flight_time += task.flyTime
            # 更新FDP结束时间为最后一个飞行任务的结束时间
            if self.end_time is None or task.sta > self.end_time:
                self.end_time = task.sta
                
        # 计算值勤时间(从第一个任务开始到最后一个飞行任务结束)
        if self.start_time and self.end_time:
            self.duty_time = int((self.end_time - self.start_time).total_seconds() / 60)
            
    def is_valid(self):
        """检查FDP是否有效(必须包含飞行任务)"""
        return self.has_flight
        
    def violates_constraints(self):
        """检查FDP是否违反约束"""
        violations = 0
        
        # 规则5: FDP最大飞行时间8小时
        if self.total_flight_time > 8 * 60:  # 480分钟
            violations += 1
            
        # 规则6: FDP最大值勤时间12小时  
        if self.duty_time > 12 * 60:  # 720分钟
            violations += 1
            
        # 规则: FDP内最多4个飞行任务
        if self.flight_count > 4:
            violations += 1
            
        # 规则: FDP内最多6个总任务
        if self.total_task_count > 6:
            violations += 1
            
        # 额外检查：连接时间约束（规则3）
        connection_violations = self._check_connection_time_constraints()
        violations += connection_violations
            
        return violations
        
    def _check_connection_time_constraints(self):
        """检查FDP内任务间的连接时间约束"""
        violations = 0
        
        for i in range(len(self.tasks) - 1):
            curr_task = self.tasks[i]
            next_task = self.tasks[i + 1]
            
            curr_end = getattr(curr_task, 'sta', getattr(curr_task, 'endTime', None))
            next_start = getattr(next_task, 'std', getattr(next_task, 'startTime', None))
            
            if curr_end and next_start:
                from datetime import timedelta
                interval = next_start - curr_end
                
                # 判断任务类型
                is_curr_flight = hasattr(curr_task, 'flightNumber')
                is_next_flight = hasattr(next_task, 'flightNumber')
                is_curr_bus = hasattr(curr_task, 'type') and 'bus' in str(curr_task.type).lower()
                is_next_bus = hasattr(next_task, 'type') and 'bus' in str(next_task.type).lower()
                
                # 航班飞行任务及飞行置位任务：不同机型间隔不小于3小时
                if (is_curr_flight or is_next_flight) and not (is_curr_bus or is_next_bus):
                    if hasattr(curr_task, 'aircraftNo') and hasattr(next_task, 'aircraftNo'):
                        if curr_task.aircraftNo != next_task.aircraftNo and interval < timedelta(hours=3):
                            violations += 1
                    else:
                        # 如果无法确定机型，按保守策略检查3小时
                        if interval < timedelta(hours=3):
                            violations += 1
                
                # 大巴置位：与相邻任务间隔不小于2小时
                elif is_curr_bus or is_next_bus:
                    if interval < timedelta(hours=2):
                        violations += 1
                
                # 其他情况：默认最小连接时间1小时
                else:
                    if interval < timedelta(hours=1):
                        violations += 1
        
        return violations
        
class DutyDay:
    """值勤日 - 一连串值勤任务，不等同于日历日，可以跨日历日但一般不超过24小时"""
    def __init__(self):
        self.tasks = []  # 所有类型任务
        self.start_time = None
        self.end_time = None
        self.start_date = None  # 第一个任务开始日期
        self.end_date = None    # 最后一个任务结束日期
        self.is_flight_duty_day = False  # 是否为飞行值勤日
        self.layover_stations = set()  # 可过夜机场集合
        self.fdps = []  # 飞行值勤期列表
        
    def add_task(self, task):
        """添加任务到值勤日"""
        self.tasks.append(task)
        
        # 更新时间范围
        task_start = getattr(task, 'std', getattr(task, 'startTime', None))
        task_end = getattr(task, 'sta', getattr(task, 'endTime', None))
        
        if self.start_time is None or (task_start and task_start < self.start_time):
            self.start_time = task_start
            if task_start:
                self.start_date = task_start.date()
                
        if self.end_time is None or (task_end and task_end > self.end_time):
            self.end_time = task_end
            if task_end:
                self.end_date = task_end.date()
        
        # 检查是否为飞行值勤日：必须包含飞行任务且从可过夜机场开始到可过夜机场结束
        if isinstance(task, Flight):
            self._update_flight_duty_status()
    
    def _update_flight_duty_status(self, layover_stations_set=None):
        """更新飞行值勤日状态"""
        # 检查是否包含飞行任务
        has_flight = any(isinstance(task, Flight) for task in self.tasks)
        if not has_flight:
            self.is_flight_duty_day = False
            return
            
        # 检查起始和结束位置是否为可过夜机场
        if self.tasks:
            first_task = self.tasks[0]
            last_task = self.tasks[-1]
            
            # 获取起始机场
            start_airport = None
            if hasattr(first_task, 'depaAirport'):
                start_airport = first_task.depaAirport
            elif hasattr(first_task, 'startLocation'):
                start_airport = first_task.startLocation
            elif hasattr(first_task, 'airport'):
                start_airport = first_task.airport
            
            # 获取结束机场
            end_airport = None
            if hasattr(last_task, 'arriAirport'):
                end_airport = last_task.arriAirport
            elif hasattr(last_task, 'endLocation'):
                end_airport = last_task.endLocation
            elif hasattr(last_task, 'airport'):
                end_airport = last_task.airport
                
            # 如果提供了可过夜机场集合，进行验证
            if layover_stations_set is not None:
                start_is_layover = start_airport in layover_stations_set if start_airport else False
                end_is_layover = end_airport in layover_stations_set if end_airport else False
                
                # 飞行值勤日必须从可过夜机场开始到可过夜机场结束
                self.is_flight_duty_day = has_flight and start_is_layover and end_is_layover
            else:
                # 如果没有可过夜机场数据，暂时简化：包含飞行任务就认为是飞行值勤日
                self.is_flight_duty_day = has_flight
    
    def set_layover_stations(self, layover_stations_set):
        """设置可过夜机场集合并重新评估飞行值勤日状态"""
        self.layover_stations = layover_stations_set
        self._update_flight_duty_status(layover_stations_set)
    
    def get_duration_hours(self):
        """获取值勤日持续时间（小时）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600.0
        return 0
    
    def spans_calendar_days(self):
        """检查是否跨日历日"""
        return self.start_date != self.end_date if (self.start_date and self.end_date) else False
            
    def organize_into_fdps(self):
        """将任务组织成飞行值勤日"""
        if not self.is_flight_duty_day:
            return
            
        current_fdp = None
        
        for task in self.tasks:
            # 如果是飞行或置位任务，加入FDP
            if isinstance(task, Flight) or (hasattr(task, 'type') and 'positioning' in str(task.type)):
                if current_fdp is None:
                    current_fdp = FlightDutyPeriod()
                    self.fdps.append(current_fdp)
                current_fdp.add_task(task)
            else:
                # 其他任务(如占位)可能结束当前FDP
                if current_fdp is not None and current_fdp.has_flight:
                    current_fdp = None
                    
class LayoverStation:
    """Represents a layover station. Columns from layoverStation.csv."""
    def __init__(self, airport):
        self.airport = airport

    def __repr__(self):
        return f"LayoverStation(Airport: {self.airport})"

class CrewLegMatch:
    """Represents crew-flight compatibility. Columns from crewLegMatch.csv."""
    def __init__(self, crewId, legId):
        self.crewId = str(crewId).strip() if pd.notna(crewId) else None
        self.flightId = str(legId).strip() if pd.notna(legId) else None

    def __repr__(self):
        return f"CrewLegMatch(Crew: {self.crewId}, Flight: {self.flightId})"

class RestPeriod:
    """Represents a rest period in a roster."""
    def __init__(self, start_time, end_time, location):
        self.start_time = start_time
        self.end_time = end_time
        self.location = location

    def __repr__(self):
        # Calculating duration for display
        duration = self.end_time - self.start_time
        return f"Rest(at:{self.location}, {duration.total_seconds()/3600:.1f}h)"

    # Add a dummy .cost and .id attribute so it can be added to a path without breaking other code
    @property
    def cost(self):
        return 0
    @property
    def id(self):
        return f"Rest_{self.location}_{self.start_time.isoformat()}"
    
class Roster:
    """Represents a full schedule for one crew member (a column in the master problem)."""
    def __init__(self, crew_id: str, duties: List[Any], cost: float):
        self.crew_id = crew_id
        self.duties = duties
        self.cost = cost
        self.is_ddh = 'DDH' in str(duties)

    def __repr__(self):
        duty_repr = ", ".join([d.flightNo if isinstance(d, Flight) else d.id if isinstance(d, GroundDuty) else type(d).__name__ for d in self.duties])
        return f"Roster(Crew: {self.crew_id}, Cost: {self.cost:.2f}, Duties: [{duty_repr}])"

# --- Helper classes for the subproblem solver ---

class Node:
    """Node for the shortest path algorithm in the subproblem."""
    def __init__(self, airport, time):
        self.airport = airport
        self.time = time

    def __eq__(self, other):
        return self.airport == other.airport and self.time == other.time

    def __hash__(self):
        return hash((self.airport, self.time))
        
    def __repr__(self):
        return f"Node(At: {self.airport}, Time: {self.time.strftime('%H:%M')})"

class Label:
    """Label for resource-constrained shortest path algorithm."""
    def __init__(self, cost, path, current_node, duty_start_time=None, 
                 duty_flight_time=0.0, duty_flight_count=0, duty_task_count=0,
                 total_flight_hours=0.0, total_flight_duty_hours=0.0, total_positioning=0, 
                 total_away_overnights=0, total_calendar_days=None, 
                 has_flown_in_duty=False, used_task_ids=None, tie_breaker=0,
                 current_cycle_start=None, current_cycle_days=0, last_base_return=None,
                 duty_days_count=1):
        self.cost = cost
        self.path = path
        self.current_node = current_node
        self.node = current_node  # 添加这行，保持向后兼容
        
        # 添加额外属性
        self.duty_start_time = duty_start_time
        self.duty_flight_time = duty_flight_time
        self.duty_flight_count = duty_flight_count
        self.duty_task_count = duty_task_count
        self.total_flight_hours = total_flight_hours
        self.total_flight_duty_hours = total_flight_duty_hours  # 总飞行值勤时间（飞行值勤日的总时长）
        self.total_positioning = total_positioning
        self.total_away_overnights = total_away_overnights
        self.total_calendar_days = total_calendar_days if total_calendar_days is not None else set()
        self.has_flown_in_duty = has_flown_in_duty
        self.used_task_ids = used_task_ids if used_task_ids is not None else set()
        self.tie_breaker = tie_breaker
        # 飞行周期管理字段
        self.current_cycle_start = current_cycle_start  # 当前飞行周期开始日期
        self.current_cycle_days = current_cycle_days    # 当前飞行周期已持续天数
        self.last_base_return = last_base_return        # 最后一次返回基地的日期
        self.duty_days_count = duty_days_count          # 值勤日数量

    def __lt__(self, other):
        return self.cost < other.cost
