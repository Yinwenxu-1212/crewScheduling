# file: initial_solution_generator.py

from datetime import datetime, timedelta
from typing import List, Dict
import os
from data_models import Flight, Crew, BusInfo, GroundDuty, Roster
from scoring_system import ScoringSystem
from results_writer import write_results_to_csv
from unified_config import UnifiedConfig

# FDP (Flight Duty Period) Rules - 从统一配置获取
FDP_RULES = {
    'max_fdp_hours': getattr(UnifiedConfig, 'MAX_DUTY_DAY_HOURS', 12),
    'max_flight_hours_in_fdp': getattr(UnifiedConfig, 'MAX_FLIGHT_TIME_IN_DUTY_HOURS', 8),
    'max_legs_in_fdp': getattr(UnifiedConfig, 'MAX_FLIGHTS_IN_DUTY', 6),
    'min_rest_period_hours': getattr(UnifiedConfig, 'MIN_REST_HOURS', 12),
}

# 定义排班规则常量 - 从统一配置获取连接时间约束
MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT = timedelta(minutes=getattr(UnifiedConfig, 'MIN_CONNECTION_TIME_FLIGHT_SAME_AIRCRAFT_MINUTES', 30))
MIN_CONNECTION_TIME_FLIGHT_DIFF_AIRCRAFT = timedelta(hours=getattr(UnifiedConfig, 'MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT_HOURS', 3))
MIN_CONNECTION_TIME_BUS = timedelta(hours=getattr(UnifiedConfig, 'MIN_CONNECTION_TIME_BUS_HOURS', 2))
DEFAULT_MIN_CONNECTION_TIME = timedelta(minutes=30) # 默认的最小连接时间，用于地面任务或其他未明确的情况
BRIEFING_TIME = timedelta(minutes=0)  # 飞行任务前简报时间
DEBRIEFING_TIME = timedelta(minutes=0) # 飞行任务后讲评时间

# FDP 和 周期规则常量 - 从统一配置获取或使用默认值
MAX_DAILY_FLIGHT_TASKS = getattr(UnifiedConfig, 'MAX_FLIGHTS_IN_DUTY', 11)
MAX_DAILY_TOTAL_TASKS = getattr(UnifiedConfig, 'MAX_TASKS_IN_DUTY', 13)
MAX_DAILY_FLIGHT_TIME = timedelta(hours=getattr(UnifiedConfig, 'MAX_FLIGHT_TIME_IN_DUTY_HOURS', 11))
MAX_DAILY_DUTY_TIME = timedelta(hours=getattr(UnifiedConfig, 'MAX_DUTY_DAY_HOURS', 15))
MAX_DUTY_PERIOD_SPAN = timedelta(hours=24) # Max span of any duty period (FDP or ground duty day) from first task start to last task end.
MIN_REST_TIME_NORMAL = timedelta(hours=getattr(UnifiedConfig, 'MIN_REST_HOURS', 10))
MIN_REST_TIME_LONG = timedelta(hours=48)   # 超过34小时的休息可重置周期
LAYOVER_STATIONS = set() # 将在加载数据时填充 (Rule 3.2.3)
MAX_CONSECUTIVE_DUTY_DAYS_AWAY = 7 # Rule 3.4.2: 在外站连续执勤（FDP）不超过7天
MIN_REST_DAYS_AT_BASE_FOR_CYCLE_RESET = timedelta(days=2) # Rule 3.4.1: 周期结束后在基地的休息时间至少为两个完整日历日

MAX_FLIGHT_CYCLE_DAYS = 5          # 飞行周期最大持续日历天数
MIN_CYCLE_REST_DAYS = 2            # 飞行周期结束后在基地的完整休息日历天数 (规则3.4.1)

MAX_TOTAL_FLIGHT_DUTY_TIME = timedelta(hours=60) # 计划期内总飞行值勤时间上限 (规则3.5)

# 辅助函数：检查占位任务与航班的连接关系
def check_ground_duty_flight_connection(crew, task, task_type):
    """
    检查占位任务与航班任务之间的特殊连接关系
    根据用户要求，占位任务（isDuty=0）之间不需要满足连接规则
    """
    if task_type != "ground_duty":
        return True
    
    # 检查占位任务是否为真正的占位（isDuty=0）
    if hasattr(task, 'isDuty') and task.isDuty == 0:
        # 占位任务之间不需要满足连接规则，直接返回True
        return True
            
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
def can_assign_task_greedy(crew, task, task_type, crew_leg_matches_set, layover_stations_set, start_date, all_flights=None): # task可以是Flight, BusInfo, GroundDuty
    # 根据用户要求，地面任务是必须要执行的，不需要考虑约束条件
    if task_type == "ground_duty":
        return True
    
    # 增强的占位任务连接检查
    if not check_ground_duty_flight_connection(crew, task, task_type):
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - 占位任务连接检查失败")
        return False
    
    # 增强的大巴任务连接检查
    if not check_bus_flight_connection(crew, task, task_type, all_flights):
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - 大巴任务连接检查失败")
        return False
    
    # 0. 置位规则检查 (大巴任务和航班置位)
    if task_type == 'bus':
        # 大巴置位允许在值勤日的开始或结束进行
        # 检查是否为大巴置位任务（通过任务ID判断）
        is_bus_positioning = hasattr(task, 'id') and task.id and str(task.id).startswith('Bus_')
        
        if is_bus_positioning:
            # 大巴置位只能在值勤日的开始或结束
            if crew.schedule:
                # 获取当前值勤日内的所有任务
                current_duty_day_tasks = []
                for task_item in crew.schedule:
                    # 检查任务是否在当前值勤日内（简化：检查是否在同一个FDP内）
                    if crew.fdp_start_time and hasattr(task_item, 'std'):
                        # 航班任务
                        if task_item.std >= crew.fdp_start_time:
                            current_duty_day_tasks.append(task_item)
                    elif crew.fdp_start_time and hasattr(task_item, 'startTime'):
                        # 地面任务或大巴任务
                        if task_item.startTime >= crew.fdp_start_time:
                            current_duty_day_tasks.append(task_item)
                
                # 检查当前值勤日内的飞行任务和大巴任务
                duty_day_flight_tasks = [t for t in current_duty_day_tasks 
                                       if hasattr(t, 'flightNo') or (hasattr(t, 'id') and t.id and t.id.startswith('Flt_'))]
                duty_day_bus_tasks = [t for t in current_duty_day_tasks 
                                    if hasattr(t, 'id') and t.id and str(t.id).startswith('Bus_')]
                
                # 如果已有飞行任务，大巴置位只能在值勤日结束时添加
                if duty_day_flight_tasks:
                    # 检查最后一个任务是否为飞行任务或大巴任务
                    last_task = crew.schedule[-1]
                    is_last_flight = hasattr(last_task, 'flightNo') or (hasattr(last_task, 'id') and last_task.id and last_task.id.startswith('Flt_'))
                    is_last_bus = hasattr(last_task, 'id') and last_task.id and str(last_task.id).startswith('Bus_')
                    
                    if not (is_last_flight or is_last_bus):
                        # 最后一个任务不是飞行任务或大巴任务，不能添加大巴置位
                        return False
                
                # 如果已有大巴置位任务，需要检查置位任务的连续性
                if duty_day_bus_tasks:
                    # 大巴置位任务只能在值勤日开始或结束连续出现
                    # 检查是否有其他任务在大巴置位任务之间
                    non_bus_tasks_between = False
                    first_bus_index = -1
                    last_bus_index = -1
                    
                    for i, task_item in enumerate(current_duty_day_tasks):
                        if task_item in duty_day_bus_tasks:
                            if first_bus_index == -1:
                                first_bus_index = i
                            last_bus_index = i
                    
                    # 检查大巴置位任务之间是否有其他任务
                    for i in range(first_bus_index + 1, last_bus_index):
                        if current_duty_day_tasks[i] not in duty_day_bus_tasks:
                            non_bus_tasks_between = True
                            break
                    
                    # 如果大巴置位任务之间有其他任务，则不能再添加大巴置位
                    if non_bus_tasks_between:
                        return False
                    
                    # 新的大巴置位任务只能在开始或结束位置添加
                    last_task = crew.schedule[-1]
                    is_last_bus_positioning = hasattr(last_task, 'id') and last_task.id and str(last_task.id).startswith('Bus_')
                    
                    if not is_last_bus_positioning:
                        # 如果最后一个任务不是大巴置位，检查是否可以在开始位置添加
                        # 这里简化处理：只允许在结束位置连续添加
                        return False
                
                # 如果没有飞行任务和大巴任务，大巴置位可以作为值勤日开始的任务
            # 如果机组没有任务，大巴置位可以作为第一个任务
        else:
            # 普通大巴任务的检查逻辑保持不变
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
    
    # 航班置位规则检查：航班置位只能在值勤日的开始或结束进行
    if task_type == 'flight':
        # 检查是否为置位航班（通过crew_leg_matches_set判断，如果不匹配则为置位）
        is_positioning_flight = (crew.crewId, task.id) not in crew_leg_matches_set
        
        if is_positioning_flight:
            # 飞行置位只能在值勤日的开始或结束
            if crew.schedule:
                # 获取当前值勤日内的所有任务
                current_duty_day_tasks = []
                for task_item in crew.schedule:
                    # 检查任务是否在当前值勤日内（简化：检查是否在同一个FDP内）
                    if crew.fdp_start_time and hasattr(task_item, 'std'):
                        # 航班任务
                        if task_item.std >= crew.fdp_start_time:
                            current_duty_day_tasks.append(task_item)
                    elif crew.fdp_start_time and hasattr(task_item, 'startTime'):
                        # 地面任务或大巴任务
                        if task_item.startTime >= crew.fdp_start_time:
                            current_duty_day_tasks.append(task_item)
                
                # 检查当前值勤日内的飞行任务
                duty_day_flight_tasks = [t for t in current_duty_day_tasks 
                                       if hasattr(t, 'flightNo') or (hasattr(t, 'id') and t.id and t.id.startswith('Flt_'))]
                
                if duty_day_flight_tasks:
                    # 检查现有航班任务的类型
                    positioning_tasks = []
                    operating_tasks = []
                    
                    for existing_task in duty_day_flight_tasks:
                        is_existing_positioning = (crew.crewId, existing_task.id) not in crew_leg_matches_set
                        if is_existing_positioning:
                            positioning_tasks.append(existing_task)
                        else:
                            operating_tasks.append(existing_task)
                    
                    # 如果既有置位又有执飞任务，则不允许再添加置位
                    if positioning_tasks and operating_tasks:
                        return False  # 不允许在混合任务中添加置位
                    
                    # 如果现有任务都是执飞，飞行置位只能在最后添加（值勤日结束）
                    if operating_tasks and not positioning_tasks:
                        # 检查是否可以在值勤日结束时添加置位
                        # 置位只能在所有执飞任务之后
                        last_task = crew.schedule[-1]
                        if not (hasattr(last_task, 'flightNo') or (hasattr(last_task, 'id') and last_task.id and last_task.id.startswith('Flt_'))):
                            # 最后一个任务不是飞行任务，不能添加飞行置位
                            return False
                    
                    # 如果现有任务都是置位，需要检查置位任务的位置
                    if positioning_tasks and not operating_tasks:
                        # 置位任务只能在值勤日开始或结束
                        # 如果已有置位任务，新的置位任务只能：
                        # 1. 在开始阶段连续添加
                        # 2. 在结束阶段连续添加
                        
                        # 检查是否有非飞行任务在置位任务之间
                        non_flight_tasks_between = False
                        first_positioning_index = -1
                        last_positioning_index = -1
                        
                        for i, task_item in enumerate(current_duty_day_tasks):
                            if task_item in positioning_tasks:
                                if first_positioning_index == -1:
                                    first_positioning_index = i
                                last_positioning_index = i
                        
                        # 检查置位任务之间是否有其他任务
                        for i in range(first_positioning_index + 1, last_positioning_index):
                            if current_duty_day_tasks[i] not in positioning_tasks:
                                non_flight_tasks_between = True
                                break
                        
                        # 如果置位任务之间有其他任务，则不能再添加置位
                        if non_flight_tasks_between:
                            return False
                        
                        # 新的置位任务只能在开始或结束位置添加
                        # 这里简化处理：如果最后一个任务是置位，可以继续添加
                        last_task = crew.schedule[-1]
                        if not (hasattr(last_task, 'flightNo') or (hasattr(last_task, 'id') and last_task.id and last_task.id.startswith('Flt_'))):
                            # 最后一个任务不是飞行任务，检查是否为置位
                            if hasattr(last_task, 'id') and last_task.id and last_task.id.startswith('Flt_'):
                                is_last_positioning = (crew.crewId, last_task.id) not in crew_leg_matches_set
                                if not is_last_positioning:
                                    return False  # 最后一个飞行任务不是置位，不能添加置位
                            else:
                                return False  # 最后一个任务不是飞行任务，不能添加置位
                
                # 如果没有飞行任务，飞行置位可以作为值勤日开始的任务
            # 如果机组没有任务，置位可以作为第一个任务

    # 占位任务的特殊检查
    if task_type == "ground_duty":
        # 检查是否为占位任务（isDuty=0）
        if hasattr(task, 'isDuty') and task.isDuty == 0:
            # 占位任务的特殊规则
            # 1. 占位任务不能在FDP的飞行阶段执行
            if crew.fdp_phase == 'in_flight':
                return False
            
            # 2. 占位任务位置检查 - 根据用户要求，所有地面任务都需要分配，放宽位置限制
            # 如果机组当前位置与任务位置不一致，允许通过调整机组位置来满足
            # if crew.current_location and crew.current_location != task.airport:
            #     return False
            
            # 3. 占位任务不应该与其他占位任务重叠
            if crew.is_on_ground_duty:
                # 检查时间是否重叠
                if crew.current_ground_duty_end_time and task.startTime < crew.current_ground_duty_end_time:
                    print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - 占位任务重叠：任务开始时间 {task.startTime} < 当前地面任务结束时间 {crew.current_ground_duty_end_time}")
                    return False

    # 1. 资格资质检查 (规则 10) - 仅针对飞行任务
    if task_type == "flight" and (crew.crewId, task.id) not in crew_leg_matches_set:
        return False # 机组与航班不匹配

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
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - 时间顺序检查失败：任务开始时间 {task_start_time} < 机组当前时间 {crew.current_time}")
        return False

    # 2. 地点衔接规则 (Rule 2.1, 2.2)
    if crew.last_activity_end_location != task_origin:
        if not crew.schedule and crew.stayStation != task_origin: # 第一个任务且不在历史停留地
            # 允许通过大巴或置位航班从基地出发 (Rule 2.2.1, 2.2.2)
            # 简化：如果任务是飞行或地面，且机组在基地，但任务不在基地，则不允许（除非有大巴/置位）
            # 当前贪心不主动创建大巴/置位来满足第一个任务的地点要求，除非任务本身就是大巴
            # 对于地面任务，根据用户要求放宽地点限制
            if task_type == 'ground_duty':
                pass # 地面任务允许在任何地点执行
            elif task_type != 'bus' and crew.stayStation == crew.base and task_origin != crew.base:
                 return False
            elif task_type == 'bus' and crew.stayStation == task.depaAirport: # 如果是巴士任务，且始发地匹配
                pass # 允许
            elif not (crew.stayStation == crew.base and task_origin in layover_stations_set) and not (crew.stayStation in layover_stations_set and task_origin == crew.base) : # 简化处理，需要更精细的置位逻辑
                 return False 
        elif crew.schedule: # 非第一个任务
            # 对于地面任务，放宽地点衔接限制
            if task_type == 'ground_duty':
                pass # 地面任务允许在任何地点执行
            else:
                return False # 直接连接失败

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
            
            # 增强的连接时间检查：考虑置位约束
            # 判断任务类型
            is_last_flight = hasattr(crew.schedule[-1], 'flightNo') or (hasattr(crew.schedule[-1], 'id') and crew.schedule[-1].id and crew.schedule[-1].id.startswith('Flt_'))
            is_current_flight = hasattr(task, 'flightNo') or (hasattr(task, 'id') and task.id and task.id.startswith('Flt_'))
            is_last_bus = hasattr(crew.schedule[-1], 'id') and crew.schedule[-1].id and str(crew.schedule[-1].id).startswith('Bus_')
            is_current_bus = hasattr(task, 'id') and task.id and str(task.id).startswith('Bus_')
            
            # 大巴置位：与相邻飞行任务及飞行置位任务间隔不小于2小时
            if (is_last_bus or is_current_bus) and connection_to_this_task_duration < timedelta(hours=2):
                return False
            
            # 航班飞行任务及飞行置位任务：不同机型间隔不小于3小时
            if is_last_flight and is_current_flight:
                last_aircraft = getattr(crew.schedule[-1], 'aircraftNo', None)
                current_aircraft = getattr(task, 'aircraftNo', None)
                
                if last_aircraft and current_aircraft and last_aircraft != current_aircraft:
                    if connection_to_this_task_duration < timedelta(hours=3):
                        return False 

    # 如果是新FDP，检查前序休息 (Rule 3.2.1)
    if is_new_fdp and crew.last_rest_end_time:
        actual_rest_before_fdp = task_start_time - crew.last_rest_end_time
        if actual_rest_before_fdp < MIN_REST_TIME_NORMAL:
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

    # 计算值勤时间 (Rule 3.1.3: FDP中首任务的计划离港时刻(STD)与该FDP中最后一个飞行任务的计划到港时刻(STA)之间的时间)
    last_flight_sta_in_temp_fdp = None
    for t_detail in reversed(temp_fdp_tasks_details_for_calc):
        if t_detail['type'] == 'flight':
            last_flight_sta_in_temp_fdp = t_detail['sta']
            break
    
    temp_fdp_duty_time = timedelta(0)
    if temp_fdp_start_for_duty_calc and last_flight_sta_in_temp_fdp: # FDP中有飞行任务
        temp_fdp_duty_time = last_flight_sta_in_temp_fdp - temp_fdp_start_for_duty_calc
    elif temp_fdp_start_for_duty_calc and temp_fdp_tasks_details_for_calc: # FDP中无飞行任务，但有其他任务
        # 规则未明确定义此种情况的FDP duty time，通常FDP围绕飞行任务展开
        # 简化：如果FDP完全由非飞行任务组成，则其执勤时间为首任务到末任务的时间
        temp_fdp_duty_time = temp_fdp_tasks_details_for_calc[-1]['sta'] - temp_fdp_start_for_duty_calc
        
    # 4. FDP内任务数量限制 (Rule 3.1.1)
    if temp_fdp_flight_tasks > MAX_DAILY_FLIGHT_TASKS:
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - FDP飞行任务数量超限：{temp_fdp_flight_tasks} > {MAX_DAILY_FLIGHT_TASKS}")
        return False
    if temp_fdp_total_tasks > MAX_DAILY_TOTAL_TASKS: # 包括飞行、地面、大巴
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - FDP总任务数量超限：{temp_fdp_total_tasks} > {MAX_DAILY_TOTAL_TASKS}")
        return False

    # 5. FDP内累计飞行时间限制 (Rule 3.1.2)
    if temp_fdp_flight_time > MAX_DAILY_FLIGHT_TIME:
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - FDP飞行时间超限：{temp_fdp_flight_time} > {MAX_DAILY_FLIGHT_TIME}")
        return False

    # 6. FDP内累计值勤时间限制 (Rule 3.1.3)
    if temp_fdp_duty_time > MAX_DAILY_DUTY_TIME:
        if task_type == "ground_duty":
            print(f"调试：机组 {crew.crewId} 无法分配地面任务 {task.id} - FDP值勤时间超限：{temp_fdp_duty_time} > {MAX_DAILY_DUTY_TIME}")
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
            temp_cycle_days_count = 1
        else:
            # 计算从周期开始到当前任务日期的天数
            temp_cycle_days_count = (current_task_date - temp_cycle_start_date_val).days + 1
        
        if temp_cycle_days_count > MAX_FLIGHT_CYCLE_DAYS:
            # 如果超期，需要检查是否在基地结束上个周期并有足够休息
            # This check is complex: requires knowing if the *previous* cycle ended at base with 2 days rest.
            # Simplified: if it's a new FDP and adding it makes cycle days > MAX, AND the crew is not starting this FDP at base after a long rest, it's a violation.
            # A more robust check would be in assign_task_greedy when a cycle actually completes or resets.
            # For now, if it looks like it will exceed, and the previous FDP didn't end at base with a cycle-ending rest, deny.
            if not (crew.last_activity_end_location == crew.base and \
                      crew.last_rest_end_time and \
                      (task_start_time - crew.last_fdp_end_time_for_cycle_check if crew.last_fdp_end_time_for_cycle_check else timedelta(0)) >= MIN_REST_DAYS_AT_BASE_FOR_CYCLE_RESET):
                return False # 飞行周期可能超限
    
    # 9. 计划期内总飞行值勤时间限制 (Rule 3.5)
    # 应该累加的是FDP的实际值勤时间。此检查在assign_task_greedy中进行更新和检查。
    # 预估： (crew.total_flight_duty_time_in_period + temp_fdp_duty_time) > MAX_TOTAL_FLIGHT_DUTY_TIME
    # 这里的temp_fdp_duty_time是当前FDP如果加入此任务后的预估值勤时间，但total_flight_duty_time_in_period是已完成FDP的累积
    # 简化：暂时不在此处做严格预估，依赖assign_task_greedy

    return True


# 辅助函数：分配任务并更新机组状态
def assign_task_greedy(crew, task, task_type, start_date, positioning_info=None, crew_leg_matches_set=None): # task可以是Flight, BusInfo, GroundDuty. Added start_date and positioning_info
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

    # 获取任务ID
    task_id_attr = task.id # Assuming all task objects have an 'id' attribute
    
    # 先检查是否为置位任务，在更新FDP阶段之前进行置位规则检查
    is_positioning = False
    if task_type == "flight":
        if positioning_info:
            # 如果有positioning_info，使用它来判断
            task_positioning_status = positioning_info.get(task_id_attr, 'operating')
            is_positioning = (task_positioning_status == 'positioning')
        elif crew_leg_matches_set:
            # 如果没有positioning_info但有crew_leg_matches_set，通过匹配关系判断
            is_positioning = (crew.crewId, task_id_attr) not in crew_leg_matches_set
        
        # 航班置位规则检查：置位只能在值勤日的开始或结束
        if is_positioning:
            # 如果机组已有任务，检查置位是否在合适位置
            if crew.schedule:
                # 检查现有航班任务类型
                existing_flight_tasks = [t for t in crew.schedule if hasattr(t, 'flightNo') or (hasattr(t, 'id') and t.id and t.id.startswith('Flt_'))]
                
                if existing_flight_tasks:
                    # 检查现有航班任务是否都是置位
                    all_existing_positioning = True
                    all_existing_operating = True
                    
                    for existing_task in existing_flight_tasks:
                        if positioning_info:
                            is_existing_positioning = positioning_info.get(existing_task.id, 'operating') == 'positioning'
                        elif crew_leg_matches_set:
                            is_existing_positioning = (crew.crewId, existing_task.id) not in crew_leg_matches_set
                        else:
                            is_existing_positioning = False
                            
                        if is_existing_positioning:
                            all_existing_operating = False
                        else:
                            all_existing_positioning = False
                    
                    # 如果既有置位又有执飞任务，则不允许再添加置位
                    if not all_existing_positioning and not all_existing_operating:
                        print(f"警告：机组 {crew.crewId} 已有混合任务，不能再分配置位航班 {task_id_attr}")
                        return  # 直接返回，不分配此任务
                    
                    # 如果现有任务都是执飞，只能在最后添加置位
                    if all_existing_operating:
                        # 检查是否还有非航班任务在最后
                        last_task = crew.schedule[-1]
                        if not (hasattr(last_task, 'flightNo') or (hasattr(last_task, 'id') and last_task.id and last_task.id.startswith('Flt_'))):
                            # 最后一个任务不是航班，不能添加置位
                            print(f"警告：机组 {crew.crewId} 最后任务不是航班，不能添加置位航班 {task_id_attr}")
                            return
    
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
                # 检查是否在基地完成周期性休息
                if crew.last_activity_end_location == crew.base and \
                   (crew.last_rest_end_time - crew.last_fdp_end_time_for_cycle_check if crew.last_fdp_end_time_for_cycle_check else timedelta(0)) >= MIN_REST_DAYS_AT_BASE_FOR_CYCLE_RESET:
                    crew.current_cycle_start_date = None # 重置周期
                    crew.current_cycle_days = 0
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
            crew.current_cycle_days = 1
            crew.consecutive_duty_days = 1 # 新周期的第一天执勤
            crew.current_cycle_at_base = (task_origin == crew.base)
        else:
            crew.current_cycle_days = (task_start_time.date() - crew.current_cycle_start_date).days + 1
            if task_origin != crew.base:
                crew.current_cycle_at_base = False

    # 置位检查已在前面完成
    
    # 添加任务到当前FDP
    crew.fdp_tasks_details.append({
        'type': task_type, 
        'id': task_id_attr, 
        'std': task_start_time, 
        'sta': task_end_time, 
        'origin': task_origin, 
        'dest': task_destination,
        'is_positioning': is_positioning
    })
    
    if task_type == "flight":
        crew.fdp_flight_tasks_count += 1
        # 置位任务的飞行时间不计入FDP执勤时间
        if not is_positioning:
            crew.fdp_flight_time += flight_duration
    
    crew.fdp_total_tasks_count += 1 # All tasks count towards total FDP tasks

    # 更新FDP值勤时间 (Rule 3.1.3)
    last_flight_sta_in_current_fdp = None
    for t_detail in reversed(crew.fdp_tasks_details):
        if t_detail['type'] == 'flight':
            last_flight_sta_in_current_fdp = t_detail['sta']
            break
    
    if crew.fdp_start_time and last_flight_sta_in_current_fdp:
        crew.fdp_duty_time = last_flight_sta_in_current_fdp - crew.fdp_start_time
    elif crew.fdp_start_time and crew.fdp_tasks_details: # FDP has no flights, e.g. only ground/bus
        crew.fdp_duty_time = crew.fdp_tasks_details[-1]['sta'] - crew.fdp_start_time
    else:
        crew.fdp_duty_time = timedelta(0)

    # 更新机组全局状态
    crew.schedule.append(task) 
    crew.current_location = task_destination
    crew.current_time = task_end_time 
    crew.last_activity_end_time = task_end_time
    crew.last_activity_end_location = task_destination
    crew.last_activity_aircraft_no = task_aircraft_no

    if task_type == "ground_duty":
        crew.is_on_ground_duty = True
        crew.current_ground_duty_end_time = task_end_time
    else: # Any non-ground duty task (flight, bus) ends ground duty status
        crew.is_on_ground_duty = False
        crew.current_ground_duty_end_time = None

def generate_initial_rosters_with_ground_first(
    flights: List[Flight], crews: List[Crew], bus_info: List[BusInfo], 
    ground_duties: List[GroundDuty], crew_leg_match_dict: dict, layover_stations=None
) -> List[Roster]:
    """
    新的初始解生成策略：先固定地面任务，再贪心寻找航班和大巴置位
    """
    print("正在使用地面任务优先的启发式算法生成初始解...")
    
    # 调试信息
    print(f"航班数量: {len(flights)}")
    print(f"机组数量: {len(crews)}")
    print(f"地面任务数量: {len(ground_duties)}")
    print(f"大巴任务数量: {len(bus_info)}")
    print(f"机组-航班匹配关系数量: {sum(len(flight_ids) for flight_ids in crew_leg_match_dict.values())}")
    
    # 设置开始日期 - 从统一配置获取
    planning_date_tuple = getattr(UnifiedConfig, 'PLANNING_START_DATE', (2025, 4, 29))
    start_date = datetime(*planning_date_tuple).date()
    
    # 构建crew_leg_matches_set
    crew_leg_matches_set = set()
    for crew_id, flight_ids in crew_leg_match_dict.items():
        for flight_id in flight_ids:
            crew_leg_matches_set.add((crew_id, flight_id))
    
    print(f"机组-航班匹配对数量: {len(crew_leg_matches_set)}")
    
    # 构建layover_stations_set (简化处理)
    layover_stations_set = set()
    
    # 第一步：为每个机组分配其必须执行的地面任务
    print("\n第一步：为每个机组分配必须执行的地面任务...")
    
    # 按机组ID分组地面任务
    crew_ground_tasks = {}
    for gd in ground_duties:
        crew_id = gd.crewId
        if crew_id not in crew_ground_tasks:
            crew_ground_tasks[crew_id] = []
        crew_ground_tasks[crew_id].append(gd)
    
    # 初始化所有机组
    crew_dict = {crew.crewId: crew for crew in crews}
    for crew in crews:
        crew.schedule = []
        crew.current_location = crew.stayStation
        crew.current_time = datetime.combine(start_date, datetime.min.time())
        crew.last_rest_end_time = crew.current_time
        crew.last_activity_end_time = None
        crew.last_activity_end_location = crew.stayStation
        crew.last_activity_aircraft_no = None
        crew.fdp_start_time = None
        crew.fdp_tasks_details = []
        crew.fdp_flight_tasks_count = 0
        crew.fdp_total_tasks_count = 0
        crew.fdp_flight_time = timedelta(0)
        crew.fdp_duty_time = timedelta(0)
        crew.current_cycle_start_date = None
        crew.current_cycle_days = 0
        crew.current_cycle_at_base = (crew.stayStation == crew.base)
        crew.total_flight_duty_time_in_period = timedelta(0)
        crew.is_on_ground_duty = False
        crew.current_ground_duty_end_time = None
        crew.consecutive_duty_days = 0
        crew.last_fdp_end_time_for_cycle_check = None
        crew.fdp_phase = 'none'
    
    # 为每个机组分配其地面任务
    assigned_ground_tasks = set()
    for crew_id, ground_tasks in crew_ground_tasks.items():
        if crew_id in crew_dict:
            crew = crew_dict[crew_id]
            # 按时间排序地面任务
            ground_tasks.sort(key=lambda x: x.startTime)
            
            print(f"为机组 {crew_id} 分配 {len(ground_tasks)} 个地面任务")
            
            for gd in ground_tasks:
                # 检查是否可以分配这个地面任务
                if can_assign_task_greedy(crew, gd, 'ground_duty', crew_leg_matches_set, layover_stations_set, start_date, flights):
                    assign_task_greedy(crew, gd, 'ground_duty', start_date, None, crew_leg_matches_set)
                    assigned_ground_tasks.add(('gd', gd.id))
                else:
                    print(f"警告：无法为机组 {crew_id} 分配地面任务 {gd.id}")
    
    print(f"成功分配了 {len(assigned_ground_tasks)} 个地面任务")
    
    # 第二步：贪心分配航班和大巴任务
    print("\n第二步：贪心分配航班和大巴任务...")
    
    # 准备未分配的航班和大巴任务
    flight_tasks = []
    bus_tasks = []
    
    for f in flights:
        flight_tasks.append({'task_obj': f, 'type': 'flight', 'start_time': f.std, 'id': f.id, 'priority': 1})
    
    for bi in bus_info:
        bus_tasks.append({'task_obj': bi, 'type': 'bus', 'start_time': bi.startTime, 'id': ('bus', bi.id), 'priority': 2})
    
    # 合并航班和大巴任务，按时间排序
    remaining_tasks = flight_tasks + bus_tasks
    remaining_tasks.sort(key=lambda x: x['start_time'])
    
    # 修改：分别跟踪航班和大巴的分配状态
    # 航班可以被多个机组分配（用于置位），大巴只能被一个机组分配
    unassigned_bus_ids = {t['id'] for t in remaining_tasks if t['type'] == 'bus'}
    assigned_tasks_count = len(assigned_ground_tasks)
    
    # 新增：跟踪每个航班被分配给了多少个机组（最多3个：1个执飞+2个置位）
    flight_assignment_count = {}
    for f in flights:
        flight_assignment_count[f.id] = 0
    
    MAX_CREWS_PER_FLIGHT = getattr(UnifiedConfig, 'MAX_CREWS_PER_FLIGHT', 6)  # 每个航班最多被分配给的机组数量

    # 贪心分配剩余任务
    for crew_idx, crew in enumerate(crews):
        crew_assigned_count = len([t for t in crew.schedule if hasattr(t, 'id')])
        crew_flight_count = 0
        crew_bus_count = 0
        crew_ground_count = len([t for t in crew.schedule if hasattr(t, 'airport')])  # 地面任务数量
        
        # 为当前机组寻找可分配的航班和大巴任务
        while True:
            best_task_to_assign = None
            best_score = float('-inf')
            
            # 智能任务选择：优先考虑能与现有任务良好衔接的任务
            for task_info in remaining_tasks:
                task_obj = task_info['task_obj']
                task_type = task_info['type']
                task_id = task_info['id']
                
                # 修改：航班可以被多个机组分配，大巴只能被一个机组分配
                can_assign = True
                if task_type == 'bus' and task_id in unassigned_bus_ids:
                    can_assign = True
                elif task_type == 'bus' and task_id not in unassigned_bus_ids:
                    can_assign = False  # 大巴已被分配
                elif task_type == 'flight':
                    # 新增：检查航班分配数量约束
                    if flight_assignment_count.get(task_id, 0) >= MAX_CREWS_PER_FLIGHT:
                        can_assign = False  # 航班已达到最大分配数量
                    else:
                        can_assign = True  # 航班可以分配
                
                if can_assign:
                    if can_assign_task_greedy(crew, task_obj, task_type, crew_leg_matches_set, layover_stations_set, start_date, flights):
                        # 计算任务的适配分数，传入航班分配计数器以实现优先级调整
                        score = calculate_task_assignment_score(crew, task_obj, task_type, flights, flight_assignment_count)
                        if score > best_score:
                            best_score = score
                            best_task_to_assign = task_info
            
            if best_task_to_assign:
                task_type = best_task_to_assign['type']
                task_id = best_task_to_assign['id']
                assign_task_greedy(crew, best_task_to_assign['task_obj'], task_type, start_date, None, crew_leg_matches_set)
                
                # 修改：只有大巴任务需要从未分配列表中移除
                if task_type == 'bus':
                    unassigned_bus_ids.remove(task_id)
                elif task_type == 'flight':
                    # 新增：更新航班分配计数器
                    flight_assignment_count[task_id] += 1
                
                crew_assigned_count += 1
                assigned_tasks_count += 1
                
                # 统计不同类型任务的分配情况
                if task_type == 'flight':
                    crew_flight_count += 1
                elif task_type == 'bus':
                    crew_bus_count += 1
            else:
                break
        
        if crew_idx < 5:  # 只打印前5个机组的详细信息
            print(f"机组 {crew.crewId} 总共分配了 {crew_assigned_count} 个任务 (航班:{crew_flight_count}, 大巴:{crew_bus_count}, 地面:{crew_ground_count})")
    
    # 第三步：处理置位逻辑和生成最终的排班方案
    print("\n第三步：处理置位逻辑和生成最终的排班方案...")
    
    # 处理航班置位逻辑
    flight_crew_assignments = process_flight_positioning(crews, flights)
    
    # 重新计算机组的FDP状态，考虑置位任务的飞行时间不计入执勤时间
    for crew in crews:
        if crew.schedule:
            recalculate_crew_fdp_with_positioning(crew, flight_crew_assignments.get(crew.crewId, {}), start_date)
    
    initial_rosters = []
    for crew in crews:
        if crew.schedule:
            # 转换为Roster格式，使用评分系统计算正确的成本
            roster = Roster(crew_id=crew.crewId, duties=crew.schedule, cost=0)
            
            # 添加置位标记信息
            roster.positioning_info = flight_crew_assignments.get(crew.crewId, {})
            
            # 如果提供了layover_stations，使用评分系统计算正确的成本
            if layover_stations is not None:
                scoring_system = ScoringSystem(flights, crews, layover_stations)
                # 使用calculate_roster_cost_with_dual_prices方法，传入空的对偶价格
                cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                    roster, crew, {}, 0.0
                )
                roster.cost = cost_details['total_cost']
            else:
                # 回退到简单的成本计算
                roster_cost = sum(getattr(task, 'cost', 0) for task in crew.schedule)
                roster.cost = roster_cost
            initial_rosters.append(roster)
    
    print(f"地面任务优先算法成功生成 {len(initial_rosters)} 个初始排班方案。")
    
    # 统计已分配的航班
    assigned_flight_ids = set()
    for crew in crews:
        for task in crew.schedule:
            if hasattr(task, 'id') and isinstance(task.id, str) and task.id.startswith('Flt_'):
                assigned_flight_ids.add(task.id)
    
    # 统计未分配的航班
    all_flight_ids = {f.id for f in flights}
    unassigned_flight_ids = all_flight_ids - assigned_flight_ids
    
    print(f"航班分配统计：")
    print(f"  总航班数量: {len(all_flight_ids)}")
    print(f"  已分配航班数量: {len(assigned_flight_ids)}")
    print(f"  未分配航班数量: {len(unassigned_flight_ids)}")
    print(f"仍有 {len(unassigned_bus_ids)} 个大巴任务未被分配。")
    
    # 新增：统计航班分配约束的效果
    flights_with_multiple_crews = 0
    flights_at_max_capacity = 0
    total_crew_assignments = 0
    
    for flight_id, count in flight_assignment_count.items():
        if count > 0:
            total_crew_assignments += count
            if count > 1:
                flights_with_multiple_crews += 1
            if count >= MAX_CREWS_PER_FLIGHT:
                flights_at_max_capacity += 1
    
    print(f"\n航班分配约束效果：")
    print(f"  每个航班最多允许 {MAX_CREWS_PER_FLIGHT} 个机组选中")
    print(f"  被多个机组选中的航班数量: {flights_with_multiple_crews}")
    print(f"  达到最大分配数量的航班: {flights_at_max_capacity}")
    print(f"  总的机组-航班分配数量: {total_crew_assignments}")
    print(f"  平均每个已分配航班被 {total_crew_assignments/len(assigned_flight_ids):.2f} 个机组选中")
    
    # 新增：详细的航班覆盖率分析
    coverage_rate = len(assigned_flight_ids) / len(all_flight_ids) * 100
    print(f"\n航班覆盖率优化效果：")
    print(f"  航班覆盖率: {coverage_rate:.2f}% ({len(assigned_flight_ids)}/{len(all_flight_ids)})")
    
    # 按分配数量统计航班分布
    assignment_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
    for flight_id in all_flight_ids:
        count = flight_assignment_count.get(flight_id, 0)
        if count <= 3:
            assignment_distribution[count] += 1
        else:
            assignment_distribution[3] += 1  # 3+归为一类
    
    print(f"  航班分配分布:")
    print(f"    未分配航班: {assignment_distribution[0]} 个")
    print(f"    被1个机组选中: {assignment_distribution[1]} 个")
    print(f"    被2个机组选中: {assignment_distribution[2]} 个")
    print(f"    被3+个机组选中: {assignment_distribution[3]} 个")
    
    # 计算分配效率
    if total_crew_assignments > 0:
        efficiency = len(assigned_flight_ids) / total_crew_assignments * 100
        print(f"  分配效率: {efficiency:.2f}% (覆盖航班数/总分配次数)")
    
    if unassigned_flight_ids:
        print(f"未分配的航班ID: {sorted(list(unassigned_flight_ids))[:10]}{'...' if len(unassigned_flight_ids) > 10 else ''}")
    
    # 输出初始解到CSV文件
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "initial_solution.csv")
    write_results_to_csv(initial_rosters, output_path)
    
    return initial_rosters

def process_flight_positioning(crews, flights):
    """处理航班置位逻辑：为多机组共享的航班选择执飞机组"""
    from collections import defaultdict
    
    # 统计每个航班被分配给了哪些机组
    flight_assignments = defaultdict(list)
    
    for crew in crews:
        for task in crew.schedule:
            if hasattr(task, 'id') and task.id and task.id.startswith('Flt_'):
                flight_assignments[task.id].append(crew.crewId)
    
    # 为每个机组记录置位信息
    crew_positioning_info = defaultdict(dict)
    
    for flight_id, assigned_crews in flight_assignments.items():
        if len(assigned_crews) > 1:
            # 多个机组分配了同一航班，选择第一个作为执飞机组
            operating_crew = assigned_crews[0]
            
            for crew_id in assigned_crews:
                if crew_id == operating_crew:
                    crew_positioning_info[crew_id][flight_id] = 'operating'  # 执飞
                else:
                    crew_positioning_info[crew_id][flight_id] = 'positioning'  # 置位
        else:
            # 只有一个机组，默认为执飞
            crew_positioning_info[assigned_crews[0]][flight_id] = 'operating'
    
    return crew_positioning_info

def recalculate_crew_fdp_with_positioning(crew, positioning_info, start_date):
    """重新计算机组的FDP状态，考虑置位任务的飞行时间不计入执勤时间"""
    from datetime import timedelta
    
    # 保存原始任务列表
    original_schedule = crew.schedule.copy()
    
    # 重置机组状态
    crew.schedule = []
    crew.current_location = crew.stayStation
    crew.current_time = datetime.combine(start_date, datetime.min.time())
    crew.last_rest_end_time = crew.current_time
    crew.last_activity_end_time = None
    crew.last_activity_end_location = crew.stayStation
    crew.last_activity_aircraft_no = None
    crew.fdp_start_time = None
    crew.fdp_tasks_details = []
    crew.fdp_flight_tasks_count = 0
    crew.fdp_total_tasks_count = 0
    crew.fdp_flight_time = timedelta(0)
    crew.fdp_duty_time = timedelta(0)
    crew.current_cycle_start_date = None
    crew.current_cycle_days = 0
    crew.current_cycle_at_base = (crew.stayStation == crew.base)
    crew.total_flight_duty_time_in_period = timedelta(0)
    crew.is_on_ground_duty = False
    crew.current_ground_duty_end_time = None
    crew.consecutive_duty_days = 0
    crew.last_fdp_end_time_for_cycle_check = None
    crew.fdp_phase = 'none'
    
    # 重新处理每个任务
    for task in original_schedule:
        # 确定任务类型 - 更准确的判断逻辑
        if hasattr(task, 'flightNo') or (hasattr(task, 'id') and task.id and task.id.startswith('Flt_')):
            task_type = 'flight'
        elif hasattr(task, 'depaAirport') and hasattr(task, 'arriAirport') and hasattr(task, 'td'):
            # BusInfo对象有depaAirport, arriAirport, td属性
            task_type = 'bus'
        elif hasattr(task, 'airport') and hasattr(task, 'isDuty'):
            # GroundDuty对象有airport, isDuty属性
            task_type = 'ground_duty'
        else:
            # 默认根据ID前缀判断
            if hasattr(task, 'id') and task.id:
                if task.id.startswith('Bus_'):
                    task_type = 'bus'
                else:
                    task_type = 'ground_duty'
            else:
                task_type = 'ground_duty'
        
        # 重新分配任务，传递置位信息
        assign_task_greedy(crew, task, task_type, start_date, positioning_info)

def calculate_task_assignment_score(crew, task, task_type, all_flights, flight_assignment_count=None):
    """
    计算任务分配的适配分数，用于智能选择最佳任务
    新增：优先分配未被覆盖的航班，降低已被多个机组选中航班的优先级
    """
    score = 0
    
    # 基础分数：任务类型优先级
    if task_type == 'flight':
        score += 100  # 航班任务优先级最高
        
        # 新增：航班覆盖率优先级调整
        if flight_assignment_count is not None:
            task_id = getattr(task, 'id', None)
            if task_id:
                current_assignment_count = flight_assignment_count.get(task_id, 0)
                
                # 优先分配未被覆盖的航班
                if current_assignment_count == 0:
                    score += 1000  # 大幅提高未覆盖航班的优先级
                elif current_assignment_count == 1:
                    score += 500  # 适度提高只被一个机组选中的航班优先级
                elif current_assignment_count == 2:
                    score -= 50   # 降低已被两个机组选中的航班优先级
                else:
                    score -= 100  # 大幅降低已被多个机组选中的航班优先级
                    
    elif task_type == 'bus':
        score += 50   # 大巴任务优先级中等
    
    # 地点衔接分数
    if crew.schedule:
        last_task = crew.schedule[-1]
        if hasattr(last_task, 'arriAirport'):
            last_location = last_task.arriAirport
        elif hasattr(last_task, 'airport'):
            last_location = last_task.airport
        else:
            last_location = crew.current_location
        
        if task_type == 'flight' and last_location == task.depaAirport:
            score += 30  # 地点完美衔接
        elif task_type == 'bus' and last_location == task.depaAirport:
            score += 20  # 大巴地点衔接
    
    # 时间衔接分数
    if crew.current_time:
        if task_type == 'flight':
            task_start = task.std
        else:  # bus
            task_start = task.startTime
        
        time_gap = (task_start - crew.current_time).total_seconds() / 3600  # 转换为小时
        
        if 2 <= time_gap <= 6:  # 理想的时间间隔
            score += 20
        elif 1 <= time_gap <= 12:  # 可接受的时间间隔
            score += 10
        elif time_gap > 24:  # 时间间隔过长
            score -= 10
    
    # 大巴任务的特殊评分：检查是否能连接后续航班
    if task_type == 'bus':
        bus_dest = task.arriAirport
        bus_end_time = task.endTime
        
        # 查找大巴到达后可能的航班
        connecting_flights = [f for f in all_flights 
                            if f.depaAirport == bus_dest and 
                               f.std > bus_end_time and 
                               (f.std - bus_end_time).total_seconds() / 3600 <= 6]  # 6小时内
        
        if connecting_flights:
            score += 25  # 大巴能连接后续航班
    
    return score

# 保持原有函数作为备用
def calculate_unified_roster_cost(roster, crews: List[Crew]) -> float:
    """
    计算roster的统一成本，与主问题和子问题保持一致
    使用统一配置的参数
    """
    if not roster.duties:
        return 0.0
    
    # 获取统一配置参数
    optimization_params = UnifiedConfig.get_optimization_params()
    flight_time_reward = optimization_params['flight_time_reward']
    positioning_penalty_rate = optimization_params['positioning_penalty']
    away_overnight_penalty_rate = optimization_params['away_overnight_penalty']
    
    # 找到对应的机组
    crew = None
    for c in crews:
        if c.crewId == roster.crew_id:
            crew = c
            break
    
    if not crew:
        return 0.0
    
    # 计算各项成本
    total_cost = 0.0
    
    # 1. 飞行时间奖励（负值，减少成本）
    flight_reward = 0.0
    for duty in roster.duties:
        if hasattr(duty, 'flightNo') and hasattr(duty, 'flyTime'):
            # 只有执行航班才能获得飞行奖励
            if not getattr(duty, 'is_positioning', False):
                flight_reward += flight_time_reward * (duty.flyTime / 60.0)
    
    # 2. 置位惩罚
    positioning_penalty = 0.0
    for duty in roster.duties:
        if hasattr(duty, 'flightNo'):
            # 检查是否为置位航班
            if getattr(duty, 'is_positioning', False):
                positioning_penalty += positioning_penalty_rate
        elif hasattr(duty, 'id') and str(duty.id).startswith('Bus_'):
            # 大巴置位任务
            positioning_penalty += positioning_penalty_rate
    
    # 3. 外站过夜惩罚
    overnight_penalty = 0.0
    sorted_duties = sorted(roster.duties, key=lambda x: getattr(x, 'std', getattr(x, 'startTime', datetime.min)))
    
    for i in range(len(sorted_duties) - 1):
        current_duty = sorted_duties[i]
        next_duty = sorted_duties[i + 1]
        
        # 获取当前任务的结束地点和时间
        current_end_airport = None
        current_end_time = None
        
        if hasattr(current_duty, 'arriAirport'):
            current_end_airport = current_duty.arriAirport
            current_end_time = getattr(current_duty, 'sta', getattr(current_duty, 'endTime', None))
        elif hasattr(current_duty, 'endTime'):
            current_end_time = current_duty.endTime
            current_end_airport = getattr(current_duty, 'arriAirport', None)
        
        # 获取下一个任务的开始时间
        next_start_time = getattr(next_duty, 'std', getattr(next_duty, 'startTime', None))
        
        # 检查外站过夜
        if (current_end_airport and current_end_airport != crew.base and 
            current_end_time and next_start_time):
            
            rest_time = next_start_time - current_end_time
            min_rest_hours = getattr(UnifiedConfig, 'MIN_REST_HOURS', 12)
            if rest_time >= timedelta(hours=min_rest_hours):
                overnight_days = (next_start_time.date() - current_end_time.date()).days
                if overnight_days > 0:
                    overnight_penalty += overnight_days * away_overnight_penalty_rate
    
    # 计算总成本：惩罚项 - 奖励项
    total_cost = positioning_penalty + overnight_penalty - flight_reward
    
    return total_cost


def generate_initial_rosters_with_heuristic(
    flights: List[Flight], crews: List[Crew], bus_info: List[BusInfo], 
    ground_duties: List[GroundDuty], crew_leg_match_dict: dict, layover_stations=None
) -> List[Roster]:
    """
    原有的启发式算法生成初始解（保持向后兼容）
    """
    # 调用新的地面任务优先算法
    return generate_initial_rosters_with_ground_first(
        flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
    )
