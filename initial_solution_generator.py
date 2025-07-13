# file: initial_solution_generator.py

from datetime import datetime, timedelta
from typing import List, Dict
import os
from data_models import Flight, Crew, BusInfo, GroundDuty, Roster
from scoring_system import ScoringSystem
from results_writer import write_results_to_csv
from unified_config import UnifiedConfig

# 辅助函数：计算有效休息期（考虑休息占位任务）
# 完善的休息期计算逻辑：
# 1. 休息占位任务（isDuty=0）在基地时计为有效休息期
# 2. 值勤占位任务（isDuty=1）不计为有效休息期
# 3. 休息占位任务不在基地时不计为有效休息期
def _calculate_effective_rest_time(rest_start_time, rest_end_time, base_airport, crew_schedule):
    """
    计算有效休息期，考虑休息占位任务对休息期的影响
    
    Args:
        rest_start_time: 休息开始时间
        rest_end_time: 休息结束时间
        base_airport: 基地机场
        crew_schedule: 机组排班列表
    
    Returns:
        timedelta: 有效休息期（天数）
    """
    # 基础休息期
    total_rest_time = rest_end_time - rest_start_time
    
    # 查找休息期间的占位任务
    rest_positioning_time = timedelta(0)
    
    for task in crew_schedule:
        task_start = None
        task_end = None
        task_location = None
        
        # 获取任务时间和地点信息
        if hasattr(task, 'startTime') and hasattr(task, 'endTime'):
             task_start = task.startTime
             task_end = task.endTime
             # 地面任务的地点信息
             task_location = getattr(task, 'airport', getattr(task, 'location', None))
        elif hasattr(task, 'std') and hasattr(task, 'sta'):
             task_start = task.std
             task_end = task.sta
             # 飞行任务的到达机场
             task_location = getattr(task, 'arriAirport', getattr(task, 'arrAirport', None))
        
        # 检查任务是否在休息期间
        if (task_start and task_end and 
            task_start >= rest_start_time and task_end <= rest_end_time):
            
            # 检查是否为地面任务
            if hasattr(task, 'isDuty'):
                if task.isDuty == 1:
                    # 值勤占位任务（isDuty=1）完全占用休息时间，不计为有效休息
                    rest_positioning_time += (task_end - task_start)
                elif task.isDuty == 0 and task_location != base_airport:
                    # 休息占位任务（isDuty=0）如果不在基地，也不计为有效休息
                    rest_positioning_time += (task_end - task_start)
                # 注意：休息占位任务（isDuty=0）在基地时计为有效休息，不减少休息时间
    
    # 有效休息期 = 总休息期 - 休息占位任务时间
    effective_rest_time = total_rest_time - rest_positioning_time
    
    # 返回 timedelta 对象，而不是天数
    return effective_rest_time

def _calculate_effective_rest_hours(rest_start_time, rest_end_time, base_airport, crew_schedule):
    """
    计算有效休息小时数，考虑休息占位任务对休息期的影响 - 增强版
    
    Args:
        rest_start_time: 休息开始时间
        rest_end_time: 休息结束时间
        base_airport: 基地机场
        crew_schedule: 机组排班列表
    
    Returns:
        float: 有效休息小时数
    """
    # 基础休息期
    total_rest_time = rest_end_time - rest_start_time
    
    # 查找休息期间的占位任务
    rest_positioning_time = timedelta(0)
    
    for task in crew_schedule:
        task_start = None
        task_end = None
        task_location = None
        
        # 获取任务时间和地点信息
        if hasattr(task, 'startTime') and hasattr(task, 'endTime'):
            task_start = task.startTime
            task_end = task.endTime
            # 地面任务的地点信息
            task_location = getattr(task, 'airport', getattr(task, 'location', None))
        elif hasattr(task, 'std') and hasattr(task, 'sta'):
            task_start = task.std
            task_end = task.sta
            # 飞行任务的到达机场
            task_location = getattr(task, 'arriAirport', getattr(task, 'arrAirport', None))
        
        # 检查任务是否在休息期间或与休息期间重叠
        if task_start and task_end:
            # 检查时间重叠：任务与休息期间有任何重叠
            if (task_start < rest_end_time and task_end > rest_start_time):
                # 计算重叠时间
                overlap_start = max(task_start, rest_start_time)
                overlap_end = min(task_end, rest_end_time)
                overlap_duration = overlap_end - overlap_start
                
                # 检查是否为地面任务
                if hasattr(task, 'isDuty'):
                    if task.isDuty == 1:
                        # 值勤占位任务（isDuty=1）完全占用休息时间，不计为有效休息
                        rest_positioning_time += overlap_duration
                    elif task.isDuty == 0 and task_location != base_airport:
                        # 休息占位任务（isDuty=0）如果不在基地，也不计为有效休息
                        rest_positioning_time += overlap_duration
                    # 注意：休息占位任务（isDuty=0）在基地时计为有效休息，不减少休息时间
                elif hasattr(task, 'id') and str(task.id).startswith('Grd_'):
                    # 对于没有isDuty属性的地面任务，根据类型判断
                    if hasattr(task, 'type') and task.type in ['ground_duty', 'groundDuty']:
                        # 如果不在基地，减少休息时间
                        if task_location != base_airport:
                            rest_positioning_time += overlap_duration
    
    # 有效休息期 = 总休息期 - 休息占位任务时间
    effective_rest_time = total_rest_time - rest_positioning_time
    
    # 转换为小时数
    return effective_rest_time.total_seconds() / 3600.0

def _has_rest_ground_duty_in_period(rest_start_time, rest_end_time, crew_schedule):
    """
    检查休息期间是否有休息占位任务（isDuty=0）- 增强版
    
    Args:
        rest_start_time: 休息开始时间
        rest_end_time: 休息结束时间
        crew_schedule: 机组排班列表
    
    Returns:
        bool: 是否有休息占位任务
    """
    for task in crew_schedule:
        task_start = None
        task_end = None
        
        # 获取任务时间信息
        if hasattr(task, 'startTime') and hasattr(task, 'endTime'):
            task_start = task.startTime
            task_end = task.endTime
        elif hasattr(task, 'std') and hasattr(task, 'sta'):
            task_start = task.std
            task_end = task.sta
        
        # 检查任务是否在休息期间或与休息期间重叠
        if task_start and task_end:
            # 检查时间重叠：任务与休息期间有任何重叠
            if (task_start < rest_end_time and task_end > rest_start_time):
                # 检查是否为休息占位任务（isDuty=0）
                is_rest_duty = False
                
                # 检查isDuty属性
                if hasattr(task, 'isDuty') and task.isDuty == 0:
                    is_rest_duty = True
                # 检查任务ID是否为地面任务且可能是休息占位
                elif hasattr(task, 'id') and str(task.id).startswith('Grd_'):
                    # 进一步检查是否为休息占位任务
                    if hasattr(task, 'type') and task.type in ['ground_duty', 'groundDuty']:
                        # 如果没有isDuty属性，假设地面任务为休息占位
                        if not hasattr(task, 'isDuty'):
                            is_rest_duty = True
                
                if is_rest_duty:
                    return True
    return False

def _can_return_to_base_via_positioning(crew, cycle_end_time, next_task_start_time, all_tasks):
    """
    检查是否可以通过置位任务回到基地
    
    Args:
        crew: 机组对象
        cycle_end_time: 飞行周期结束时间
        next_task_start_time: 下一个任务开始时间
        all_tasks: 所有可用任务列表
    
    Returns:
        bool: 是否可以通过置位回到基地
    """
    if not crew.last_activity_end_location or crew.last_activity_end_location == crew.base:
        return True
    
    # 查找可能的置位任务（大巴或航班）
    current_location = crew.last_activity_end_location
    
    # 检查是否有从当前位置到基地的置位任务
    for task_info in all_tasks:
        if isinstance(task_info, dict):
            task = task_info.get('task_obj')
            task_type = task_info.get('type')
        else:
            task = task_info
            task_type = getattr(task, 'type', None)
        
        if not task:
            continue
            
        # 获取任务时间
        task_start = getattr(task, 'startTime', getattr(task, 'std', None))
        task_end = getattr(task, 'endTime', getattr(task, 'sta', None))
        
        if not task_start or not task_end:
            continue
            
        # 检查时间是否在飞行周期结束后、下一个任务开始前
        if task_start >= cycle_end_time and task_end <= next_task_start_time:
            # 检查是否为置位任务（大巴或航班）
            if task_type in ['bus', 'flight']:
                # 获取出发地和目的地
                origin = getattr(task, 'depaAirport', getattr(task, 'origin', None))
                destination = getattr(task, 'arriAirport', getattr(task, 'destination', None))
                
                # 检查是否从当前位置出发到基地
                if origin == current_location and destination == crew.base:
                    return True
                    
                # 检查是否可以通过多段置位回到基地（简化版本，只检查一段）
                if origin == current_location:
                    # 递归检查是否可以从destination继续回到基地
                    # 为简化，这里只检查直达的情况
                    pass
    
    return False


def _validate_positioning_rules(crew, task, task_type):
    """
    验证置位规则：同一值勤日内，仅允许在开始或结束进行置位
    
    Args:
        crew: 机组对象
        task: 任务对象
        task_type: 任务类型
    
    Returns:
        bool: 是否符合置位规则
    """
    # 只对置位任务进行检查
    if task_type not in ['flight', 'bus']:
        return True
    
    # 检查是否为置位任务
    is_positioning = False
    if task_type == 'flight':
        # 对于航班，需要检查是否为置位
        is_positioning = getattr(task, 'is_positioning', False)
    elif task_type == 'bus':
        # 大巴任务通常都是置位
        is_positioning = True
    
    if not is_positioning:
        return True
    
    # 如果当前没有活跃的FDP，置位任务可以作为值勤日开始
    if not hasattr(crew, 'fdp_start_time') or not crew.fdp_start_time:
        return True
    
    # 检查当前值勤日的任务组成
    if hasattr(crew, 'fdp_tasks_details') and crew.fdp_tasks_details:
        # 统计当前值勤日中的置位任务数量和飞行任务数量
        positioning_count = 0
        flight_count = 0
        
        for task_detail in crew.fdp_tasks_details:
            if task_detail['type'] == 'flight':
                if task_detail.get('is_positioning', False):
                    positioning_count += 1
                else:
                    flight_count += 1
            elif task_detail['type'] == 'bus':
                positioning_count += 1
        
        # 置位规则：仅允许在值勤日开始或结束进行置位
        # 如果值勤日中已经有飞行任务，且已经有置位任务，不允许再添加置位
        if flight_count > 0 and positioning_count > 0:
            return False
        
        # 如果已经有多个置位任务，不允许
        if positioning_count >= 1:
            return False
    
    return True


# FDP (Flight Duty Period) Rules - Centralized Configuration
FDP_RULES = {
    'max_fdp_hours': 12,  # Maximum FDP duration in hours
    'max_flight_hours_in_fdp': 8, # Maximum flight hours within a single FDP
    'max_legs_in_fdp': 6, # Maximum number of flight legs in an FDP
    'min_rest_period_hours': 12, # Minimum rest period between FDPs
}

# 定义排班规则常量
MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT = timedelta(minutes=30) # 飞机尾号相同时的最小衔接时间 (规则3.3.1)
MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT = timedelta(hours=3)    # 飞机尾号不同时的最小衔接时间 (规则3.3.1)
MIN_CONNECTION_TIME_BUS = timedelta(hours=2)                     # 大巴置位最小衔接时间 (规则3.3.1)
DEFAULT_MIN_CONNECTION_TIME = timedelta(minutes=30) # 默认的最小连接时间，用于地面任务或其他未明确的情况
BRIEFING_TIME = timedelta(minutes=0)  # 飞行任务前简报时间
DEBRIEFING_TIME = timedelta(minutes=0) # 飞行任务后讲评时间

# FDP 和 周期规则常量
MAX_DAILY_FLIGHT_TASKS = 4               # Rule 3.1.1: FDP内最多飞行任务数
MAX_DAILY_TOTAL_TASKS = 6                # Rule 3.1.1: FDP内最多总任务数 (飞行+地面+大巴)
MAX_DAILY_FLIGHT_TIME = timedelta(hours=8) # Rule 3.1.2: FDP内累计飞行时间限制
MAX_DAILY_DUTY_TIME = timedelta(hours=12)  # Rule 3.1.3: FDP内累计值勤时间限制
MAX_DUTY_PERIOD_SPAN = timedelta(hours=24) # Max span of any duty period (FDP or ground duty day) from first task start to last task end.
MIN_REST_TIME_NORMAL = timedelta(hours=12) # Rule 3.2.1: FDP开始前正常休息时间
MIN_REST_TIME_LONG = timedelta(hours=48)   # 超过34小时的休息可重置周期
LAYOVER_STATIONS = set() # 将在加载数据时填充 (Rule 3.2.3)
MAX_CONSECUTIVE_DUTY_DAYS_AWAY = 6 # Rule 3.4.2: 在外站连续执勤（FDP）不超过6天 (请根据具体规则核实此值)
MIN_REST_DAYS_AT_BASE_FOR_CYCLE_RESET = timedelta(days=2) # Rule 3.4.1: 周期结束后在基地的休息时间至少为两个完整日历日

MAX_FLIGHT_CYCLE_DAYS = 4          # 飞行周期最大持续日历天数 (规则3.4.1)
MIN_CYCLE_REST_DAYS = 2            # 飞行周期结束后在基地的完整休息日历天数 (规则3.4.1)

MAX_TOTAL_FLIGHT_DUTY_TIME = timedelta(hours=60) # 计划期内总飞行值勤时间上限 (规则3.5)

# 辅助函数：检查占位任务与航班的连接关系
def check_ground_duty_flight_connection(crew, task, task_type):
    """
    检查占位任务与航班任务之间的特殊连接关系
    """
    if task_type != "ground_duty":
        return True
    
    # 检查占位任务是否为真正的占位（isDuty=0）
    if hasattr(task, 'isDuty') and task.isDuty == 0:
        # 占位任务必须在合适的时间和地点执行
        # 1. 占位任务应该在航班任务前后的合理时间窗口内
        # 2. 占位任务的地点应该与相关航班的起降地点一致
        
        # 如果机组当前有任务，检查占位与前一个任务的关系
        if crew.schedule:
            last_task = crew.schedule[-1]
            # 占位任务应该在同一地点或相关地点
            if hasattr(last_task, 'arriAirport') and last_task.arriAirport != task.airport:
                return False
        
        # 占位任务不应该跨越过长时间
        task_duration = task.endTime - task.startTime
        if task_duration > timedelta(hours=8):  # 占位任务不应超过8小时
            return False
            
    return True

# 辅助函数：检查大巴任务与航班的连接关系
def check_bus_flight_connection(crew, task, task_type, all_flights):
    """
    检查大巴任务与航班任务之间的逻辑关系
    """
    if task_type != "bus":
        return True
    
    # 大巴任务应该服务于航班连接
    # 1. 检查大巴任务是否连接了有效的航班
    # 2. 确保大巴任务的时间安排合理
    
    bus_origin = task.depaAirport
    bus_destination = task.arriAirport
    bus_start_time = task.startTime
    bus_end_time = task.endTime
    
    # 检查大巴任务前后是否有相关的航班任务
    # 这里简化处理，实际应该检查整个航班网络
    if crew.schedule:
        last_task = crew.schedule[-1]
        if hasattr(last_task, 'arriAirport'):
            # 大巴起点应该与上一个任务的终点一致
            if last_task.arriAirport != bus_origin:
                return False
    
    # 大巴任务的持续时间应该合理（不超过6小时）
    bus_duration = bus_end_time - bus_start_time
    if bus_duration > timedelta(hours=6):
        return False
        
    return True

# 辅助函数：检查任务是否可以分配给机组 (现在也处理其他类型任务)
def can_assign_task_greedy(crew, task, task_type, crew_leg_matches_set, layover_stations_set, start_date, all_flights=None, assigned_flights=None): # task可以是Flight, BusInfo, GroundDuty
    # 增强的占位任务连接检查
    if not check_ground_duty_flight_connection(crew, task, task_type):
        return False
    
    # 增强的大巴任务连接检查
    if not check_bus_flight_connection(crew, task, task_type, all_flights):
        return False
    
    # 0. 置位规则检查 (大巴任务)
    if task_type == 'bus':
        # 大巴只能在FDP的开始（第一个任务）或结束（最后一个任务）时进行
        # 'pre_flight' 意味着FDP已开始但尚未有飞行任务
        # 'post_flight' 意味着FDP的飞行任务已全部结束
        # 'none' 意味着这是一个全新的FDP的第一个任务
        if crew.fdp_phase not in ['none', 'pre_flight', 'post_flight']:
            return False
        
        # 大巴任务优先级检查：确保大巴任务真正服务于航班连接
        if crew.schedule:
            # 如果机组已有任务，大巴任务应该是合理的连接
            last_task = crew.schedule[-1]
            if hasattr(last_task, 'arriAirport') and last_task.arriAirport == task.depaAirport:
                # 检查时间间隔是否合理
                if hasattr(last_task, 'sta'):
                    time_gap = task.startTime - last_task.sta
                    if time_gap < timedelta(minutes=30) or time_gap > timedelta(hours=4):
                        return False

    # 占位任务的特殊检查
    if task_type == "ground_duty":
        # 严格检查地面任务的crewId匹配（这是必须满足的约束）
        if hasattr(task, 'crewId') and task.crewId != crew.crewId:
            return False  # 地面任务只能由指定的机组执行
        
        # 检查是否为占位任务（isDuty=0）
        if hasattr(task, 'isDuty') and task.isDuty == 0:
            # 占位任务的特殊规则
            # 1. 占位任务不能在FDP的飞行阶段执行
            if crew.fdp_phase == 'in_flight':
                return False
            
            # 2. 占位任务应该与机组当前位置一致
            if crew.current_location and crew.current_location != task.airport:
                return False
            
            # 3. 占位任务不应该与其他占位任务重叠
            if crew.is_on_ground_duty:
                # 检查时间是否重叠
                if crew.current_ground_duty_end_time and task.startTime < crew.current_ground_duty_end_time:
                    return False

    # 1. 资格资质检查 (规则 10) - 针对飞行任务
    if task_type == "flight":
        # 检查机组是否有资格执飞此航班
        can_operate = (crew.crewId, task.id) in crew_leg_matches_set
        
        # 检查此航班是否已被其他机组执飞（可用作置位）
        can_position = assigned_flights is not None and task.id in assigned_flights
        
        # 机组必须能执飞或者航班已被执飞（可置位）
        if not (can_operate or can_position):
            return False # 机组与航班不匹配且航班未被执飞

    # 统一任务属性获取
    if task_type == "flight":
        task_start_time = task.std
        task_end_time = task.sta
        task_origin = task.depaAirport
        task_destination = task.arriAirport
        flight_duration = timedelta(minutes=task.flyTime)
    elif task_type == "bus":
        # 修正大巴任务的时间属性
        task_start_time = task.td if hasattr(task, 'td') else task.startTime
        task_end_time = task.ta if hasattr(task, 'ta') else task.endTime
        task_origin = task.depaAirport
        task_destination = task.arriAirport
        flight_duration = timedelta(0)
    else:  # ground_duty
        task_start_time = task.startTime
        task_end_time = task.endTime
        task_origin = task.airport
        task_destination = task.airport
        flight_duration = timedelta(0)

    # 1. 检查时间顺序：任务开始时间必须晚于或等于机组当前可用时间
    if crew.current_time and task_start_time < crew.current_time:
        return False

    # 2. 地点衔接规则 (Rule 2.1, 2.2) - 增强版
    if crew.last_activity_end_location != task_origin:
        if not crew.schedule and crew.stayStation != task_origin: # 第一个任务且不在历史停留地
            # 允许通过大巴或置位航班从基地出发 (Rule 2.2.1, 2.2.2)
            if task_type == 'bus':
                # 大巴任务：检查起点是否与机组当前位置匹配
                if crew.stayStation != task.depaAirport:
                    return False
            elif task_type == 'flight':
                # 飞行任务：如果机组在基地但任务不在基地，需要置位
                if crew.stayStation == crew.base and task_origin != crew.base:
                    # 检查是否可以通过置位到达
                    if task_origin not in layover_stations_set:
                        return False
                elif crew.stayStation != crew.base and crew.stayStation != task_origin:
                    return False
            else:  # ground_duty
                # 地面任务：必须在当前位置执行
                if crew.stayStation != task_origin:
                    return False
        elif crew.schedule: # 非第一个任务
            # 严格检查地点衔接：前一个任务的终点必须与当前任务的起点一致
            if crew.last_activity_end_location != task_origin:
                # 特殊情况：如果是大巴任务，可能用于地点转换
                if task_type == 'bus' and crew.last_activity_end_location == task.depaAirport:
                    pass  # 允许大巴任务进行地点转换
                else:
                    return False # 地点衔接失败

    # 决定是否开始新的FDP (Flight Duty Period)
    is_new_fdp = False
    connection_to_this_task_duration = timedelta(0)
    if not crew.fdp_start_time: # 机组的第一个任务，必定是新FDP
        is_new_fdp = True
    elif crew.last_activity_end_time: # 如果已有任务
        connection_to_this_task_duration = task_start_time - crew.last_activity_end_time
        if connection_to_this_task_duration >= MIN_REST_TIME_NORMAL: # Rule 3.2.1, 3.3
            is_new_fdp = True 
        else: 
            # 3. 最小连接时间检查 (Rule 3.3.1)
            min_connection_this_task = DEFAULT_MIN_CONNECTION_TIME
            if task_type == "flight":
                if crew.last_activity_aircraft_no and hasattr(task, 'aircraftNo') and task.aircraftNo == crew.last_activity_aircraft_no:
                    min_connection_this_task = MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT
                elif crew.last_activity_aircraft_no: 
                    min_connection_this_task = MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT
            elif task_type == "bus":
                min_connection_this_task = MIN_CONNECTION_TIME_BUS
            
            if connection_to_this_task_duration < min_connection_this_task:
                return False 

    # 如果是新FDP，检查前序休息 (Rule 3.2.1)
    if is_new_fdp:
        # 飞行值勤日开始前最小休息时间检查：12小时 - 修正版
        if crew.last_activity_end_time:  # 有前序任务
            # 休息开始时间：最后一个任务的结束时间
            rest_start_time = crew.last_activity_end_time
            # 休息结束时间：第一个任务的开始时间
            rest_end_time = task_start_time
            
            # 计算有效休息时间（考虑休息占位任务）
            effective_rest_hours = _calculate_effective_rest_hours(
                 rest_start_time, rest_end_time, crew.base, crew.schedule
             )
             
            # 检查是否有休息占位任务（isDuty=0）在休息期间
            has_rest_ground_duty = _has_rest_ground_duty_in_period(
                rest_start_time, rest_end_time, crew.schedule
            )
            
            # 如果有休息占位任务，则认为休息时间充足；否则检查实际休息时间
            # 修正：将小时数转换为timedelta进行比较
            effective_rest_time = timedelta(hours=effective_rest_hours)
            if not has_rest_ground_duty and effective_rest_time < MIN_REST_TIME_NORMAL:
                return False
        
        # 检查连续执勤天数 (Rule 3.4.2) - 简化：如果新FDP与上个FDP不在同一天，且上个FDP结束在外站
        if crew.last_fdp_end_time_for_cycle_check and task_start_time.date() > crew.last_fdp_end_time_for_cycle_check.date() and \
           crew.last_activity_end_location != crew.base:
            crew.consecutive_duty_days += (task_start_time.date() - crew.last_fdp_end_time_for_cycle_check.date()).days
        elif crew.last_fdp_end_time_for_cycle_check and task_start_time.date() == crew.last_fdp_end_time_for_cycle_check.date():
            pass # 同一天开始的新FDP，连续执勤天数不变
        else: # 第一个FDP，或者上个FDP在基地结束并有足够休息
            crew.consecutive_duty_days = 1 
        
        if crew.consecutive_duty_days > MAX_CONSECUTIVE_DUTY_DAYS_AWAY:
            return False # 连续执勤超限

    # 临时计算当前任务加入后FDP的状态
    temp_fdp_flight_tasks = crew.fdp_flight_tasks_count
    temp_fdp_total_tasks = crew.fdp_total_tasks_count
    temp_fdp_flight_time = crew.fdp_flight_time
    # temp_fdp_duty_time = crew.fdp_duty_time # 将在下面重新计算
    temp_fdp_start_for_duty_calc = crew.fdp_start_time
    temp_fdp_tasks_details_for_calc = list(crew.fdp_tasks_details) # 创建副本进行计算

    if is_new_fdp:
        temp_fdp_flight_tasks = 0
        temp_fdp_total_tasks = 0
        temp_fdp_flight_time = timedelta(0)
        temp_fdp_start_for_duty_calc = task_start_time
        temp_fdp_tasks_details_for_calc = []

    # 将当前任务加入临时FDP列表以计算执勤时间
    temp_fdp_tasks_details_for_calc.append({'type': task_type, 'std': task_start_time, 'sta': task_end_time, 'id': task.id if hasattr(task,'id') else None})

    if task_type == "flight":
        temp_fdp_flight_tasks += 1
    # GroundDuty 和 Bus 也计入总任务数 (Rule 3.1.1)
    temp_fdp_total_tasks += 1
    temp_fdp_flight_time += flight_duration

    # 计算飞行值勤日值勤时间：从第一个任务开始时间到最后一个飞行任务的到达时间
    last_flight_sta_in_temp_fdp = None
    for t_detail in reversed(temp_fdp_tasks_details_for_calc):
        if t_detail['type'] == 'flight':
            last_flight_sta_in_temp_fdp = t_detail['sta']
            break
    
    temp_fdp_duty_time = timedelta(0)
    if temp_fdp_start_for_duty_calc and last_flight_sta_in_temp_fdp: # FDP中有飞行任务
        # 飞行值勤开始时间为第一个任务的开始时间，飞行值勤结束时间为最后一个飞行任务的到达时间
        temp_fdp_duty_time = last_flight_sta_in_temp_fdp - temp_fdp_start_for_duty_calc
    elif temp_fdp_start_for_duty_calc and temp_fdp_tasks_details_for_calc: # FDP中无飞行任务，但有其他任务
        # 如果FDP完全由非飞行任务组成，则不是有效的飞行值勤日
        # 但为了计算，使用首任务到末任务的时间
        temp_fdp_duty_time = temp_fdp_tasks_details_for_calc[-1]['sta'] - temp_fdp_start_for_duty_calc
        
    # 4. FDP内任务数量限制 (Rule 3.1.1)
    if temp_fdp_flight_tasks > MAX_DAILY_FLIGHT_TASKS:
        return False
    if temp_fdp_total_tasks > MAX_DAILY_TOTAL_TASKS: # 包括飞行、地面、大巴
        return False

    # 5. FDP内累计飞行时间限制 (Rule 3.1.2)
    if temp_fdp_flight_time > MAX_DAILY_FLIGHT_TIME:
        return False

    # 6. FDP内累计值勤时间限制 (Rule 3.1.3) - 修正版
    # 分别检查单个飞行值勤日时长（≤12小时）和总飞行值勤时间（≤60小时）
    if temp_fdp_duty_time > MAX_DAILY_DUTY_TIME:  # 单个飞行值勤日不超过12小时
        return False
    
    # 检查总飞行值勤时间（≤60小时）
    if task_type == 'flight':  # 只有飞行任务才计入总飞行值勤时间
        current_total_flight_duty_hours = crew.total_flight_duty_time_in_period.total_seconds() / 3600.0
        potential_total_flight_duty_hours = current_total_flight_duty_hours + temp_fdp_duty_time.total_seconds() / 3600.0
        if potential_total_flight_duty_hours > MAX_TOTAL_FLIGHT_DUTY_TIME.total_seconds() / 3600.0:
            return False

    # 7. 过夜站限制 (Rule 3.2.3) - FDP结束和下一个FDP开始必须在基地或指定过夜站
    # 这个检查在assign_task_greedy中，当一个FDP实际结束时（即下一个任务开启新FDP或无任务可接）进行
    # 此处仅预判：如果当前任务是FDP的最后一个（之后是长休），且目的地不合规
    # 简化：暂时不在此处做严格预判，依赖assign_task_greedy中的逻辑

    # 8. FDP 内空飞结构检查 (规则 3.1.4)
    # 如果当前任务是飞行任务，而FDP状态是 'post_flight'，则不允许，因为飞行任务已经结束
    if task_type == 'flight' and crew.fdp_phase == 'post_flight':
        return False

    # 9. 飞行周期限制 (Rule 3.4.1)
    if is_new_fdp:
        current_task_date = task_start_time.date()
        temp_cycle_days_count = crew.current_cycle_days
        temp_cycle_start_date_val = crew.current_cycle_start_date

        if not temp_cycle_start_date_val: # 第一个FDP of the planning period for this crew
            # 检查飞行周期开始前是否在基地并有足够休息
            if crew.last_activity_end_time:  # 有前序活动
                # 检查是否在基地结束并有2个完整日历日休息
                if not (crew.last_activity_end_location == crew.base and \
                       (task_start_time - crew.last_activity_end_time).days >= MIN_CYCLE_REST_DAYS):
                    return False  # 飞行周期开始条件不满足
            temp_cycle_days_count = 1
        else:
            # 计算从周期开始到当前任务日期的天数
            temp_cycle_days_count = (current_task_date - temp_cycle_start_date_val).days + 1
        
        if temp_cycle_days_count > MAX_FLIGHT_CYCLE_DAYS:
            # 如果超期，需要检查是否在基地结束上个周期并有足够休息 - 修正版
            if not (crew.last_activity_end_location == crew.base and \
                      crew.last_activity_end_time and \
                      (task_start_time - crew.last_activity_end_time).days >= MIN_CYCLE_REST_DAYS):
                return False # 飞行周期可能超限
        
        # 飞行周期约束的额外检查
        if task_type == 'flight':  # 只有飞行任务才能开始或继续飞行周期
            # 检查飞行周期开始前的休息要求（2个完整日历日）
            if not temp_cycle_start_date_val and crew.last_activity_end_time:
                rest_days = (task_start_time.date() - crew.last_activity_end_time.date()).days
                if rest_days < MIN_CYCLE_REST_DAYS:
                    return False  # 飞行周期开始前休息不足
    
    # 9. 计划期内总飞行值勤时间限制 (Rule 3.5)
    # 应该累加的是FDP的实际值勤时间。此检查在assign_task_greedy中进行更新和检查。
    # 预估： (crew.total_flight_duty_time_in_period + temp_fdp_duty_time) > MAX_TOTAL_FLIGHT_DUTY_TIME
    # 这里的temp_fdp_duty_time是当前FDP如果加入此任务后的预估值勤时间，但total_flight_duty_time_in_period是已完成FDP的累积
    # 简化：暂时不在此处做严格预估，依赖assign_task_greedy

    return True


# 辅助函数：分配任务并更新机组状态
def assign_task_greedy(crew, task, task_type, start_date, crew_leg_matches_set=None, assigned_flights=None): # task可以是Flight, BusInfo, GroundDuty. Added start_date
    # 统一任务属性获取
    if task_type == "flight":
        task_start_time = task.std
        task_end_time = task.sta
        task_origin = task.depaAirport
        task_destination = task.arriAirport
        flight_duration = timedelta(minutes=task.flyTime)
    elif task_type == "bus":
        # 修正大巴任务的时间属性
        task_start_time = task.td if hasattr(task, 'td') else task.startTime
        task_end_time = task.ta if hasattr(task, 'ta') else task.endTime
        task_origin = task.depaAirport
        task_destination = task.arriAirport
        flight_duration = timedelta(0)
    else:  # ground_duty
        task_start_time = task.startTime
        task_end_time = task.endTime
        task_origin = task.airport
        task_destination = task.airport
        flight_duration = timedelta(0)

    # 更新 FDP 阶段
    if crew.fdp_phase == 'none': # 新FDP的第一个任务
        if task_type == 'flight':
            crew.fdp_phase = 'in_flight'
        else: # bus or ground duty
            crew.fdp_phase = 'pre_flight'
    elif crew.fdp_phase == 'pre_flight':
        if task_type == 'flight':
            crew.fdp_phase = 'in_flight'
    elif crew.fdp_phase == 'in_flight':
        if task_type != 'flight': # 飞行任务结束后接了地面或大巴
            crew.fdp_phase = 'post_flight'
    # 如果是 'post_flight'，则状态保持不变，因为只能接地面或大巴任务
    task_aircraft_no = task.aircraftNo if task_type == "flight" else None
    task_id_attr = task.id # Assuming all task objects have an 'id' attribute

    is_new_fdp = False
    previous_fdp_duty_time_to_add = timedelta(0)

    if not crew.fdp_start_time: 
        is_new_fdp = True
    elif crew.last_activity_end_time: 
        connection_or_rest_duration = task_start_time - crew.last_activity_end_time
        if connection_or_rest_duration >= MIN_REST_TIME_NORMAL:
            is_new_fdp = True
            # 上一个FDP结束，将其值勤时间加入总数
            previous_fdp_duty_time_to_add = crew.fdp_duty_time 
            crew.last_rest_end_time = task_start_time 
            crew.last_fdp_end_time_for_cycle_check = crew.last_activity_end_time # 记录上个FDP结束时间点

            # 检查飞行周期结束和重置 (Rule 3.4.1)
            if crew.current_cycle_start_date: 
                # 检查上一个FDP是否为飞行值勤日（包含飞行任务）
                last_fdp_has_flight = False
                if crew.fdp_tasks_details:
                    for task_detail in crew.fdp_tasks_details:
                        if task_detail['type'] == 'flight':
                            last_fdp_has_flight = True
                            break
                
                # 计算有效休息期：考虑休息占位任务
                effective_rest_time = _calculate_effective_rest_time(
                    crew.last_activity_end_time, task_start_time, crew.base, crew.schedule
                )
                
                # 飞行周期结束条件 - 修正版：
                # 1. 结束必须是飞行值勤日（有飞行航班任务）
                # 2. 必须回到基地（飞行任务结束在基地）
                # 3. 在基地有2个完整日历日休息（考虑休息占位任务）
                can_end_cycle = False
                
                if last_fdp_has_flight and effective_rest_time.days >= MIN_CYCLE_REST_DAYS:
                    # 检查最后一个飞行任务是否结束在基地
                    last_flight_ends_at_base = False
                    if crew.fdp_tasks_details:
                        for task_detail in reversed(crew.fdp_tasks_details):
                            if task_detail['type'] == 'flight':
                                last_flight_ends_at_base = (task_detail['dest'] == crew.base)
                                break
                    
                    if last_flight_ends_at_base and crew.last_activity_end_location == crew.base:
                        # 飞行任务结束在基地，可以结束周期
                        can_end_cycle = True
                
                if can_end_cycle:
                     crew.current_cycle_start_date = None # 重置周期
                     crew.current_cycle_start_time = None # 重置飞行周期开始时间
                     crew.current_cycle_end_time = None   # 重置飞行周期结束时间
                     crew.current_cycle_days = 0
                     crew.cycle_duration_hours = 0.0      # 重置飞行周期持续时间
                     crew.consecutive_duty_days = 0 # 在基地长休后重置连续执勤

    if is_new_fdp:
        # 累加前一个FDP的执勤时间 (如果有)
        crew.total_flight_duty_time_in_period += previous_fdp_duty_time_to_add
        if crew.total_flight_duty_time_in_period > MAX_TOTAL_FLIGHT_DUTY_TIME: # Rule 3.5 check
            pass # Or raise an error / mark as invalid roster

        crew.fdp_start_time = task_start_time
        crew.fdp_tasks_details = []
        crew.fdp_flight_tasks_count = 0
        crew.fdp_total_tasks_count = 0
        crew.fdp_flight_time = timedelta(0)
        # crew.fdp_duty_time is calculated below

        # 更新飞行周期开始 (Rule 3.4.1)
        if not crew.current_cycle_start_date: 
            crew.current_cycle_start_date = task_start_time.date()
            # 飞行周期开始时间为停止休息的时间（即当前任务开始时间）
            crew.current_cycle_start_time = task_start_time
            crew.current_cycle_days = 1
            crew.consecutive_duty_days = 1 # 新周期的第一天执勤
            crew.current_cycle_at_base = (task_origin == crew.base)
        else:
            crew.current_cycle_days = (task_start_time.date() - crew.current_cycle_start_date).days + 1
            if task_origin != crew.base:
                crew.current_cycle_at_base = False

    # 判断飞行任务是执飞还是置位
    is_positioning = False
    if task_type == "flight":
        # 检查机组是否有资格执飞此航班
        can_operate = crew_leg_matches_set is not None and (crew.crewId, task.id) in crew_leg_matches_set
        # 检查此航班是否已被其他机组执飞
        already_operated = assigned_flights is not None and task.id in assigned_flights
        
        if can_operate and not already_operated:
            # 机组执飞此航班
            is_positioning = False
            # 注意：只有在任务成功分配后才将航班添加到assigned_flights
        elif already_operated:
            # 机组作为乘客置位
            is_positioning = True
        else:
            # 这种情况不应该发生，因为can_assign_task_greedy已经检查过
            is_positioning = False
    
    # 添加任务到当前FDP
    task_detail = {
        'type': task_type, 
        'id': task_id_attr, 
        'std': task_start_time, 
        'sta': task_end_time, 
        'origin': task_origin, 
        'dest': task_destination
    }
    if task_type == "flight":
        task_detail['is_positioning'] = is_positioning
    
    crew.fdp_tasks_details.append(task_detail)
    
    if task_type == "flight":
        crew.fdp_flight_tasks_count += 1
        # 置位航班的飞行时间不计入机组的飞行时间
        if not is_positioning:
            crew.fdp_flight_time += flight_duration
    crew.fdp_total_tasks_count += 1 # All tasks count towards total FDP tasks

    # 更新飞行值勤日值勤时间：从第一个任务开始时间到最后一个飞行任务的到达时间
    last_flight_sta_in_current_fdp = None
    for t_detail in reversed(crew.fdp_tasks_details):
        if t_detail['type'] == 'flight':
            last_flight_sta_in_current_fdp = t_detail['sta']
            break
    
    if crew.fdp_start_time and last_flight_sta_in_current_fdp:
        # 飞行值勤开始时间为第一个任务的开始时间，飞行值勤结束时间为最后一个飞行任务的到达时间
        crew.fdp_duty_time = last_flight_sta_in_current_fdp - crew.fdp_start_time
    elif crew.fdp_start_time and crew.fdp_tasks_details: # FDP has no flights, e.g. only ground/bus
        # 如果没有飞行任务，则不是有效的飞行值勤日，但为了计算使用首任务到末任务的时间
        crew.fdp_duty_time = crew.fdp_tasks_details[-1]['sta'] - crew.fdp_start_time
    else:
        crew.fdp_duty_time = timedelta(0)

    # 设置Flight对象的is_positioning属性
    if task_type == "flight":
        task.is_positioning = is_positioning
        # 更新飞行周期结束时间为最后一个飞行任务的结束时间（包括置位航班）
        if crew.current_cycle_start_time:
            if crew.current_cycle_end_time is None or task_end_time > crew.current_cycle_end_time:
                crew.current_cycle_end_time = task_end_time
            # 计算飞行周期持续时间：从停止休息时间到最后一个飞行任务结束时间
            crew.cycle_duration_hours = (crew.current_cycle_end_time - crew.current_cycle_start_time).total_seconds() / 3600.0
    
    # 更新机组全局状态
    crew.schedule.append(task) 
    crew.current_location = task_destination
    crew.current_time = task_end_time 
    crew.last_activity_end_time = task_end_time
    crew.last_activity_end_location = task_destination
    crew.last_activity_aircraft_no = task_aircraft_no

    # 如果是执飞航班（非置位），将航班添加到已执飞集合中
    if task_type == "flight" and not is_positioning and assigned_flights is not None:
        assigned_flights.add(task.id)

    if task_type == "ground_duty":
        crew.is_on_ground_duty = True
        crew.current_ground_duty_end_time = task_end_time
    else: # Any non-ground duty task (flight, bus) ends ground duty status
        crew.is_on_ground_duty = False
        crew.current_ground_duty_end_time = None

def generate_initial_rosters_with_heuristic(
    flights: List[Flight], crews: List[Crew], bus_info: List[BusInfo], 
    ground_duties: List[GroundDuty], crew_leg_match_dict: dict, layover_stations=None
) -> List[Roster]:
    """
    使用与crew_scheduling_solver.py相同的贪心启发式算法生成初始解。
    """
    print("正在使用启发式算法生成初始解...")
    
    # 调试信息
    print(f"航班数量: {len(flights)}")
    print(f"机组数量: {len(crews)}")
    print(f"地面任务数量: {len(ground_duties)}")
    print(f"大巴任务数量: {len(bus_info)}")
    print(f"机组-航班匹配关系数量: {sum(len(flight_ids) for flight_ids in crew_leg_match_dict.values())}")
    
    # 设置开始日期 - 从flight.csv中动态读取最早航班日期
    from unified_config import UnifiedConfig
    planning_start_date = UnifiedConfig.get_planning_start_date()
    start_date = planning_start_date.date()
    
    # 构建crew_leg_matches_set
    crew_leg_matches_set = set()
    for crew_id, flight_ids in crew_leg_match_dict.items():
        for flight_id in flight_ids:
            crew_leg_matches_set.add((crew_id, flight_id))
    
    print(f"机组-航班匹配对数量: {len(crew_leg_matches_set)}")
    
    # 构建layover_stations_set (简化处理)
    layover_stations_set = set()
    
    initial_rosters = []
    all_tasks = []
    for f in flights: 
        # 飞行任务优先级最高
        all_tasks.append({'task_obj': f, 'type': 'flight', 'start_time': f.std, 'id': f.id, 'priority': 3})
    
    for gd in ground_duties: 
        # 地面任务，区分isDuty以优化休息期计算
        if hasattr(gd, 'isDuty'):
            if gd.isDuty == 0:
                # 休息占位任务优先级较高，有助于休息期计算
                priority = 1
            else:
                # 值勤占位任务优先级中等
                priority = 2
        else:
            # 未知类型地面任务优先级较低
            priority = 4
        all_tasks.append({'task_obj': gd, 'type': 'ground_duty', 'start_time': gd.startTime, 'id': ('gd', gd.id), 'priority': priority})
    
    for bi in bus_info: 
        # 大巴任务优先级较低，但在需要时应该被优先考虑
        all_tasks.append({'task_obj': bi, 'type': 'bus', 'start_time': bi.startTime, 'id': ('bus', bi.id), 'priority': 5})
    

    # 智能任务排序函数
    def smart_task_sort_key(task_info):
        """
        智能任务排序，考虑任务间的关联性
        """
        task_obj = task_info['task_obj']
        task_type = task_info['type']
        priority = task_info['priority']
        start_time = task_info['start_time']
        
        # 基础排序：优先级 + 时间
        base_key = (priority, start_time)
        
        # 对于休息占位任务，如果在基地且紧邻航班任务，提高优先级
        if task_type == 'ground_duty' and hasattr(task_obj, 'isDuty'):
            if task_obj.isDuty == 0:
                # 休息占位任务：查找时间相近的航班任务
                nearby_flights = [f for f in flights if abs((f.std - start_time).total_seconds()) < 3600]  # 1小时内
                if nearby_flights:
                    # 如果有相近的航班，稍微提高优先级
                    base_key = (priority - 1, start_time)
            elif task_obj.isDuty == 1:
                # 值勤占位任务：保持原优先级，但可能需要特殊处理
                pass
        
        # 对于大巴任务，如果能连接航班，提高优先级
        elif task_type == 'bus':
            # 查找可能连接的航班
            task_end_time = task_obj.ta if hasattr(task_obj, 'ta') else task_obj.endTime
            connecting_flights = [f for f in flights 
                                if f.depaAirport == task_obj.arriAirport and 
                                   f.std > task_end_time]
            if connecting_flights:
                # 如果能连接航班，提高优先级
                base_key = (priority - 0.5, start_time)
        
        return base_key
    
    # 使用智能排序
    sorted_tasks = sorted(all_tasks, key=smart_task_sort_key)
    unassigned_task_ids = {t['id'] for t in sorted_tasks}
    
    # 跟踪已被执飞的航班（用于置位）
    assigned_flights = set()

    assigned_tasks_count = 0
    for crew_idx, crew in enumerate(crews):
        # 初始化机组状态
        crew.schedule = []
        crew.current_location = crew.stayStation
        crew.current_time = datetime.combine(start_date, datetime.min.time())
        crew.last_rest_end_time = crew.current_time
        crew.last_activity_end_time = None
        crew.last_activity_end_location = crew.stayStation
        crew.last_activity_aircraft_no = None
        crew.fdp_start_time = None
        crew.fdp_tasks_details = [] # 存储FDP内任务的详细信息
        crew.fdp_flight_tasks_count = 0
        crew.fdp_total_tasks_count = 0
        crew.fdp_flight_time = timedelta(0)
        crew.fdp_duty_time = timedelta(0)
        crew.current_cycle_start_date = None
        crew.current_cycle_start_time = None  # 飞行周期开始时间（停止休息的时间）
        crew.current_cycle_end_time = None    # 飞行周期结束时间（最后一个飞行任务结束时间）
        crew.current_cycle_days = 0
        crew.current_cycle_at_base = (crew.stayStation == crew.base)
        crew.cycle_duration_hours = 0.0       # 飞行周期持续时间（小时）
        crew.total_flight_duty_time_in_period = timedelta(0)
        crew.is_on_ground_duty = False
        crew.current_ground_duty_end_time = None
        crew.consecutive_duty_days = 0 
        crew.last_fdp_end_time_for_cycle_check = None
        crew.fdp_phase = 'none'  # FDP 阶段: 'none', 'pre_flight', 'in_flight', 'post_flight'

        crew_assigned_count = 0
        crew_flight_count = 0
        crew_bus_count = 0
        crew_ground_count = 0
        
        while True:
            best_task_to_assign = None
            
            # 改进的任务选择策略：优先考虑能形成合理连接的任务
            for task_info in sorted_tasks:
                task_obj = task_info['task_obj']
                task_type = task_info['type']
                task_id = task_info['id']

                if task_id in unassigned_task_ids:
                    if (can_assign_task_greedy(crew, task_obj, task_type, crew_leg_matches_set, layover_stations_set, start_date, flights, assigned_flights) and
                        _validate_positioning_rules(crew, task_obj, task_type)):
                        best_task_to_assign = task_info
                        break
            
            if best_task_to_assign:
                task_type = best_task_to_assign['type']
                assign_task_greedy(crew, best_task_to_assign['task_obj'], task_type, start_date, crew_leg_matches_set, assigned_flights)
                unassigned_task_ids.remove(best_task_to_assign['id'])
                crew_assigned_count += 1
                assigned_tasks_count += 1
                
                # 统计不同类型任务的分配情况
                if task_type == 'flight':
                    crew_flight_count += 1
                elif task_type == 'bus':
                    crew_bus_count += 1
                elif task_type == 'ground_duty':
                    crew_ground_count += 1
            else:
                break
        
        if crew_idx < 5:  # 只打印前5个机组的详细信息
            print(f"机组 {crew.crewId} 分配了 {crew_assigned_count} 个任务 (航班:{crew_flight_count}, 大巴:{crew_bus_count}, 地面:{crew_ground_count})") 
        
        if crew.schedule:
            # 转换为Roster格式，使用评分系统计算正确的成本
            roster = Roster(crew_id=crew.crewId, duties=crew.schedule, cost=0)
            
            # 使用统一的评分系统计算成本
            if layover_stations is not None:
                scoring_system = ScoringSystem(flights, crews, layover_stations)
                # 使用统一的成本计算方法，初始解生成时global_duty_days_denominator为0，使用原始逻辑
                roster.cost = scoring_system.calculate_unified_roster_cost(roster, crew, global_duty_days_denominator=0)
            else:
                # 回退到简单的成本计算
                roster_cost = sum(getattr(task, 'cost', 0) for task in crew.schedule)
                roster.cost = roster_cost
            initial_rosters.append(roster)

    print(f"启发式算法成功生成 {len(initial_rosters)} 个初始排班方案。")
    unassigned_flight_ids = {uid for uid in unassigned_task_ids if not (isinstance(uid, tuple) and (uid[0] == 'bus' or uid[0] == 'gd'))}
    print(f"仍有 {len(unassigned_task_ids)} 个任务未被分配。")
    print(f"其中未分配的航班数量: {len(unassigned_flight_ids)}")
    print(f"已被执飞的航班数量: {len(assigned_flights)}")
    
    # 统计置位航班使用情况
    positioning_count = 0
    operating_count = 0
    for roster in initial_rosters:
        crew = next(c for c in crews if c.crewId == roster.crew_id)
        for task_detail in crew.fdp_tasks_details:
            if task_detail['type'] == 'flight':
                if task_detail.get('is_positioning', False):
                    positioning_count += 1
                else:
                    operating_count += 1
    
    print(f"执飞航班任务数量: {operating_count}")
    print(f"置位航班任务数量: {positioning_count}")
    
    # 输出初始解到CSV文件
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "initial_solution.csv")
    write_results_to_csv(initial_rosters, output_path)
    
    return initial_rosters


# def calculate_unified_roster_cost(roster, crews: List[Crew]) -> float:
#     """
#     [已弃用] 计算roster的统一成本
    
#     注意：此函数已被移至scoring_system.py中的ScoringSystem.calculate_unified_roster_cost方法
#     请使用新的统一评分系统以确保所有模块的计算逻辑一致
    
#     此函数保留仅为向后兼容，建议迁移到新的统一方法
#     """
#     if not roster.duties:
#         return 0.0
    
#     # 获取统一配置参数
#     optimization_params = UnifiedConfig.get_optimization_params()
#     flight_time_reward = optimization_params['flight_time_reward']
#     positioning_penalty_rate = optimization_params['positioning_penalty']
#     away_overnight_penalty_rate = optimization_params['away_overnight_penalty']
    
#     # 找到对应的机组
#     crew = None
#     for c in crews:
#         if c.crewId == roster.crew_id:
#             crew = c
#             break
    
#     if not crew:
#         return 0.0
    
#     # 计算各项成本
#     total_cost = 0.0
    
#     # 1. 飞行时间奖励（负值，减少成本）
#     flight_reward = 0.0
#     for duty in roster.duties:
#         if hasattr(duty, 'flightNo') and hasattr(duty, 'flyTime'):
#             # 只有执行航班才能获得飞行奖励
#             if not getattr(duty, 'is_positioning', False):
#                 flight_reward += flight_time_reward * (duty.flyTime / 60.0)
    
#     # 2. 置位惩罚
#     positioning_penalty = 0.0
#     for duty in roster.duties:
#         if hasattr(duty, 'flightNo'):
#             # 检查是否为置位航班
#             if getattr(duty, 'is_positioning', False):
#                 positioning_penalty += positioning_penalty_rate
#         elif hasattr(duty, 'id') and str(duty.id).startswith('Bus_'):
#             # 大巴置位任务
#             positioning_penalty += positioning_penalty_rate
    
#     # 3. 外站过夜惩罚
#     overnight_penalty = 0.0
#     sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
    
#     for i in range(len(sorted_duties) - 1):
#         current_duty = sorted_duties[i]
#         next_duty = sorted_duties[i + 1]
        
#         # 获取当前任务的结束地点和时间
#         current_end_airport = None
#         current_end_time = None
        
#         if hasattr(current_duty, 'arriAirport'):
#             current_end_airport = current_duty.arriAirport
#             current_end_time = getattr(current_duty, 'sta', getattr(current_duty, 'endTime', None))
#         elif hasattr(current_duty, 'endTime'):
#             current_end_time = current_duty.endTime
#             current_end_airport = getattr(current_duty, 'arriAirport', None)
        
#         # 获取下一个任务的开始时间
#         next_start_time = getattr(next_duty, 'std', getattr(next_duty, 'startTime', None))
        
#         # 检查外站过夜
#         if (current_end_airport and current_end_airport != crew.base and 
#             current_end_time and next_start_time):
            
#             rest_time = next_start_time - current_end_time
#             min_rest_hours = getattr(UnifiedConfig, 'MIN_REST_HOURS', 12)
#             if rest_time >= timedelta(hours=min_rest_hours):
#                 overnight_days = (next_start_time.date() - current_end_time.date()).days
#                 if overnight_days > 0:
#                     overnight_penalty += overnight_days * away_overnight_penalty_rate
    
#     # 计算总成本：惩罚项 - 奖励项
#     total_cost = positioning_penalty + overnight_penalty - flight_reward
    
#     return total_cost