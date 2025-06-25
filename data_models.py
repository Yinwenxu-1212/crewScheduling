# file: data_models.py
# Final version based on official column names from a-T-集.xlsx

from datetime import datetime
from typing import List, Any
import pandas as pd

class Flight:
    """Represents a flight segment. Columns from flight.csv."""
    def __init__(self, id, flightNo, depaAirport, arriAirport, std, sta, fleet, aircraftNo, flyTime):
        self.id = str(id).strip() if pd.notna(id) else None
        self.flightNo = flightNo
        self.depaAirport = depaAirport
        self.arriAirport = arriAirport
        self.std = datetime.strptime(std, '%Y/%m/%d %H:%M')
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
        self.startTime = datetime.strptime(startTime, '%Y/%m/%d %H:%M')
        self.endTime = datetime.strptime(endTime, '%Y/%m/%d %H:%M')
        self.airport = airport

    def __repr__(self):
        duty_status = "Duty" if self.isDuty else "Rest"
        return (f"GroundDuty(ID: {self.id}, Crew: {self.crewId}, "
                f"Status: {duty_status}, Start: {self.startTime}, End: {self.endTime})")

class BusInfo:
    """Represents ground transportation. Columns from bus.csv."""
    def __init__(self, id, depaAirport, arriAirport, td, ta):
        self.id = id  # 移除多余的逗号
        self.depaAirport = depaAirport
        self.arriAirport = arriAirport
        self.td = datetime.strptime(td, '%Y/%m/%d %H:%M')  # 添加缺失的属性
        self.ta = datetime.strptime(ta, '%Y/%m/%d %H:%M')   # 添加缺失的属性
        self.startTime = datetime.strptime(td, '%Y/%m/%d %H:%M')
        self.endTime = datetime.strptime(ta, '%Y/%m/%d %H:%M')
        self.cost = 0

    def __repr__(self):
        return f"Bus(Dep: {self.depaAirport}, Arr: {self.arriAirport}, Time: {self.startTime} -> {self.endTime})"

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
        duty_repr = ", ".join([d.flightNo if isinstance(d, Flight) else d.task if isinstance(d, GroundDuty) else type(d).__name__ for d in self.duties])
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
                 total_flight_hours=0.0, total_positioning=0, 
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
