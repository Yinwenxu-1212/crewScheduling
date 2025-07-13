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

class DutyDay:
    """值勤日 - 一连串值勤任务的集合
    
    定义：
    - 包含一连串值勤任务（飞行、置位、占位）
    - 跨度不超过24小时
    - 可以只包含占位任务
    - 不等同于日历日，可以跨日历日
    """
    def __init__(self):
        self.tasks = []  # 所有类型任务
        self.start_time = None
        self.end_time = None
        self.start_date = None  # 第一个任务开始日期
        self.end_date = None    # 最后一个任务结束日期
        self.layover_stations = set()  # 可过夜机场集合
        
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
    
    def set_layover_stations(self, layover_stations_set):
        """设置可过夜机场集合"""
        self.layover_stations = layover_stations_set
    
    def get_duration_hours(self):
        """获取值勤日持续时间（小时）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600.0
        return 0
    
    def spans_calendar_days(self):
        """检查是否跨日历日"""
        return self.start_date != self.end_date if (self.start_date and self.end_date) else False
    
    def has_flight_tasks(self):
        """检查是否包含飞行任务"""
        return any(isinstance(task, Flight) for task in self.tasks)
    
    def violates_24_hour_constraint(self):
        """检查是否违反24小时约束"""
        return self.get_duration_hours() > 24
    
    @property
    def is_flight_duty_day(self):
        """判断是否为飞行值勤日"""
        return self.has_flight_tasks()
    
    def get_flight_count(self):
        """获取飞行任务数量"""
        return sum(1 for task in self.tasks if isinstance(task, Flight))
    
    def get_total_flight_time_minutes(self):
        """获取总飞行时间（分钟）"""
        return sum(task.flyTime for task in self.tasks if isinstance(task, Flight) and not getattr(task, 'is_positioning', False))
    
    def get_start_airport(self):
        """获取开始机场"""
        if not self.tasks:
            return None
        first_task = self.tasks[0]
        return getattr(first_task, 'depaAirport', getattr(first_task, 'airport', None))
    
    def get_end_airport(self):
        """获取结束机场"""
        if not self.tasks:
            return None
        last_task = self.tasks[-1]
        return getattr(last_task, 'arriAirport', getattr(last_task, 'airport', None))

class FlightDutyPeriod(DutyDay):
    """飞行值勤日(FDP) - 值勤日的子类，专门处理包含飞行任务的值勤日
    
    定义：
    - 继承自DutyDay，具有值勤日的所有特性
    - 必须包含执行飞行任务（非置位飞行任务）
    - 不包含占位任务（GroundDuty）
    - 只能从可过夜机场出发到可过夜机场结束
    - 时长不能超过24小时
    """
    def __init__(self):
        super().__init__()  # 调用父类构造函数
        # FDP特有的统计属性
        self.has_flight = False
        self.flight_count = 0
        self.total_flight_time = 0
        
    def add_task(self, task):
        """添加任务到FDP
        
        注意：飞行值勤日不应包含占位任务（GroundDuty）
        """
        # 检查是否为占位任务，如果是则不添加
        if isinstance(task, GroundDuty):
            return False  # 飞行值勤日不包含占位任务
            
        # 调用父类的add_task方法
        super().add_task(task)
        
        # FDP特有的逻辑：统计执行飞行任务（非置位飞行任务）
        if isinstance(task, Flight) and not getattr(task, 'is_ddh', False) and not getattr(task, 'positioning_flight', False):
            self.has_flight = True
            self.flight_count += 1
            self.total_flight_time += task.flyTime
            
        return True  # 成功添加任务
            
    def is_valid(self, layover_stations_set=None):
        """检查FDP是否有效
        
        条件：
        1. 必须包含执行飞行任务（非置位）
        2. 不能包含占位任务（groundDuty）
        3. 持续时间不能超过24小时
        4. 必须从可过夜机场出发到可过夜机场结束
        """
        # 首先检查基本的值勤日有效性
        if self.violates_24_hour_constraint():
            return False
            
        # FDP特有的验证：必须包含执行飞行任务
        if not self.has_flight:
            return False
            
        # 检查是否包含占位任务
        for task in self.tasks:
            if isinstance(task, GroundDuty):
                return False
                
        # 如果提供了可过夜机场集合，验证起始和结束机场
        if layover_stations_set is not None:
            start_airport = self.get_start_airport()
            end_airport = self.get_end_airport()
            start_is_layover = start_airport in layover_stations_set if start_airport else False
            end_is_layover = end_airport in layover_stations_set if end_airport else False
            return start_is_layover and end_is_layover
            
        return True
        
    def get_flight_duty_duration_hours(self):
        """获取飞行值勤时间（小时）
        
        根据规则6：飞行值勤开始时间为第一个任务的开始时间，
        飞行值勤结束时间为最后一个飞行任务的到达时间
        """
        if not self.tasks:
            return 0
            
        # 第一个任务的开始时间
        first_task = self.tasks[0]
        start_time = getattr(first_task, 'std', getattr(first_task, 'startTime', None))
        
        # 最后一个飞行任务的到达时间
        end_time = None
        for task in reversed(self.tasks):
            if isinstance(task, Flight):
                end_time = getattr(task, 'sta', getattr(task, 'endTime', None))
                break
                
        if start_time and end_time:
            return (end_time - start_time).total_seconds() / 3600.0
        return 0
    
    def violates_constraints(self):
        """检查FDP是否违反约束"""
        violations = 0
        
        # 继承父类的24小时约束检查
        if self.violates_24_hour_constraint():
            violations += 1
            
        # FDP特有约束：
        # 规则5: FDP最大飞行时间8小时
        if self.get_total_flight_time_minutes() > 8 * 60:  # 480分钟
            violations += 1
            
        # 规则6: FDP最大值勤时间12小时（使用正确的飞行值勤时间计算）
        if self.get_flight_duty_duration_hours() > 12:
            violations += 1
            
        # 规则: FDP内最多4个飞行任务
        if self.get_flight_count() > 4:
            violations += 1
            
        # 规则: FDP内最多6个总任务
        if len(self.tasks) > 6:  # 使用父类的tasks列表
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
                is_curr_flight = isinstance(curr_task, Flight)
                is_next_flight = isinstance(next_task, Flight)
                is_curr_bus = isinstance(curr_task, BusInfo)
                is_next_bus = isinstance(next_task, BusInfo)
                
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
                
                # 其他情况：题目约束规则未明确规定，不进行额外约束检查
        
        return violations
        

class FlightCycle:
    """飞行周期 - 由值勤日组成的周期
    
    定义：
    - 由值勤日组成，必须包含飞行值勤日
    - 末尾必须是飞行值勤日
    - 最多横跨4个日历日
    - 开始前需连续休息2个完整日历日
    """
    def __init__(self):
        self.duty_days = []  # 值勤日列表
        self.flight_duty_periods = []  # 飞行值勤日列表
        self.start_date = None  # 周期开始日期
        self.end_date = None    # 周期结束日期
        self.start_time = None  # 周期开始时间（停止休息的时间）
        self.end_time = None    # 周期结束时间（最后一个飞行任务的结束时间）
        
    def add_duty_day(self, duty_day):
        """添加值勤日到飞行周期"""
        self.duty_days.append(duty_day)
        
        # 更新时间范围
        if self.start_date is None or (duty_day.start_date and duty_day.start_date < self.start_date):
            self.start_date = duty_day.start_date
            self.start_time = duty_day.start_time
            
        if self.end_date is None or (duty_day.end_date and duty_day.end_date > self.end_date):
            self.end_date = duty_day.end_date
            
        # 如果是飞行值勤日，更新结束时间为最后一个飞行任务的结束时间
        if duty_day.has_flight_tasks():
            for task in reversed(duty_day.tasks):
                if isinstance(task, Flight):
                    task_end = getattr(task, 'sta', getattr(task, 'endTime', None))
                    if self.end_time is None or (task_end and task_end > self.end_time):
                        self.end_time = task_end
                    break
    
    def add_flight_duty_period(self, fdp):
        """添加飞行值勤日到飞行周期"""
        self.flight_duty_periods.append(fdp)
        
    def get_calendar_days_span(self):
        """获取飞行周期跨越的日历日数"""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days + 1
        return 0
        
    def has_flight_duty_periods(self):
        """检查是否包含飞行值勤日"""
        return len(self.flight_duty_periods) > 0 or any(dd.has_flight_tasks() for dd in self.duty_days)
        
    def ends_with_flight_duty_period(self):
        """检查是否以飞行值勤日结束"""
        if not self.duty_days:
            return False
        return self.duty_days[-1].has_flight_tasks()
        
    def violates_constraints(self):
        """检查是否违反飞行周期约束"""
        violations = 0
        
        # 规则1: 必须包含飞行值勤日
        if not self.has_flight_duty_periods():
            violations += 1
            
        # 规则2: 末尾必须是飞行值勤日
        if not self.ends_with_flight_duty_period():
            violations += 1
            
        # 规则3: 最多横跨4个日历日
        if self.get_calendar_days_span() > 4:
            violations += 1
            
        return violations
    
    def get_cycle_duration_hours(self):
        """获取飞行周期持续时间（小时）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds() / 3600.0
        return 0
    
    def get_total_flight_time_minutes(self):
        """获取周期内总飞行时间（分钟）"""
        total = 0
        for duty_day in self.duty_days:
            total += duty_day.get_total_flight_time_minutes()
        return total
    
    def get_flight_duty_count(self):
        """获取飞行值勤日数量"""
        return sum(1 for dd in self.duty_days if dd.is_flight_duty_day)
    
    def starts_at_base(self, crew_base):
        """检查是否在基地开始"""
        if not self.duty_days:
            return False
        first_duty = self.duty_days[0]
        start_airport = first_duty.get_start_airport()
        return start_airport == crew_base
    
    def ends_at_base(self, crew_base):
        """检查是否在基地结束"""
        if not self.duty_days:
            return False
        last_duty = self.duty_days[-1]
        end_airport = last_duty.get_end_airport()
        return end_airport == crew_base

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
                 duty_days_count=1, is_first_cycle_done=False):
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
        self.is_first_cycle_done = is_first_cycle_done  # 是否已完成第一个飞行周期

    def __lt__(self, other):
        return self.cost < other.cost
