# file: initial_solution_generator.py

from datetime import datetime, timedelta
from typing import List, Dict
import os
from data_models import Flight, Crew, BusInfo, GroundDuty, Roster
from scoring_system import ScoringSystem
from results_writer import write_results_to_csv

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

# 辅助函数：检查任务是否可以分配给机组 (现在也处理其他类型任务)
def can_assign_task_greedy(crew, task, task_type, crew_leg_matches_set, layover_stations_set, start_date): # task可以是Flight, BusInfo, GroundDuty
 # 0. 置位规则检查 (大巴任务)
    if task_type == 'bus':
        # 大巴只能在FDP的开始（第一个任务）或结束（最后一个任务）时进行
        # 'pre_flight' 意味着FDP已开始但尚未有飞行任务
        # 'post_flight' 意味着FDP的飞行任务已全部结束
        # 'none' 意味着这是一个全新的FDP的第一个任务
        if crew.fdp_phase not in ['none', 'pre_flight', 'post_flight']:
            return False

    # 1. 资格资质检查 (规则 10) - 仅针对飞行任务
    if task_type == "flight" and (crew.crewId, task.id) not in crew_leg_matches_set:
        return False # 机组与航班不匹配

    task_start_time = task.std if task_type == "flight" else (task.td if task_type == "bus" else task.startTime)
    task_end_time = task.sta if task_type == "flight" else (task.ta if task_type == "bus" else task.endTime)
    task_origin = task.depaAirport if task_type == "flight" else (task.depaAirport if task_type == "bus" else task.airport)
    task_destination = task.arriAirport if task_type == "flight" else (task.arriAirport if task_type == "bus" else task.airport)
    flight_duration = timedelta(minutes=task.flyTime) if task_type == "flight" else timedelta(0)

    # 1. 检查时间顺序：任务开始时间必须晚于或等于机组当前可用时间
    if crew.current_time and task_start_time < crew.current_time:
        return False

    # 2. 地点衔接规则 (Rule 2.1, 2.2)
    if crew.last_activity_end_location != task_origin:
        if not crew.schedule and crew.stayStation != task_origin: # 第一个任务且不在历史停留地
            # 允许通过大巴或置位航班从基地出发 (Rule 2.2.1, 2.2.2)
            # 简化：如果任务是飞行或地面，且机组在基地，但任务不在基地，则不允许（除非有大巴/置位）
            # 当前贪心不主动创建大巴/置位来满足第一个任务的地点要求，除非任务本身就是大巴
            if task_type != 'bus' and crew.stayStation == crew.base and task_origin != crew.base:
                 return False
            elif task_type == 'bus' and crew.stayStation == task.depaAirport: # 如果是巴士任务，且始发地匹配
                pass # 允许
            elif not (crew.stayStation == crew.base and task_origin in layover_stations_set) and not (crew.stayStation in layover_stations_set and task_origin == crew.base) : # 简化处理，需要更精细的置位逻辑
                 return False 
        elif crew.schedule: # 非第一个任务
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
        return False
    if temp_fdp_total_tasks > MAX_DAILY_TOTAL_TASKS: # 包括飞行、地面、大巴
        return False

    # 5. FDP内累计飞行时间限制 (Rule 3.1.2)
    if temp_fdp_flight_time > MAX_DAILY_FLIGHT_TIME:
        return False

    # 6. FDP内累计值勤时间限制 (Rule 3.1.3)
    if temp_fdp_duty_time > MAX_DAILY_DUTY_TIME:
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
def assign_task_greedy(crew, task, task_type, start_date): # task可以是Flight, BusInfo, GroundDuty. Added start_date
    task_start_time = task.std if task_type == "flight" else (task.startTime if task_type == "bus" else task.startTime)
    task_end_time = task.sta if task_type == "flight" else (task.endTime if task_type == "bus" else task.endTime)
    task_origin = task.depaAirport if task_type == "flight" else (task.depaAirport if task_type == "bus" else task.airport)
    task_destination = task.arriAirport if task_type == "flight" else (task.arriAirport if task_type == "bus" else task.airport)
    flight_duration = timedelta(minutes=task.flyTime) if task_type == "flight" else timedelta(0)

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

    # 添加任务到当前FDP
    crew.fdp_tasks_details.append({'type': task_type, 'id': task_id_attr, 'std': task_start_time, 'sta': task_end_time, 'origin': task_origin, 'dest': task_destination})
    if task_type == "flight":
        crew.fdp_flight_tasks_count += 1
    crew.fdp_total_tasks_count += 1 # All tasks count towards total FDP tasks
    crew.fdp_flight_time += flight_duration

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

def generate_initial_rosters_with_heuristic(
    flights: List[Flight], crews: List[Crew], bus_info: List[BusInfo], 
    ground_duties: List[GroundDuty], crew_leg_match_dict: dict, layover_stations=None
) -> List[Roster]:
    """
    使用与crew_scheduling_solver.py相同的贪心启发式算法生成初始解。
    """
    print("正在使用启发式算法生成初始解...")
    
    # 设置开始日期
    start_date = datetime(2025, 5, 28).date()
    
    # 构建crew_leg_matches_set
    crew_leg_matches_set = set()
    for crew_id, flight_ids in crew_leg_match_dict.items():
        for flight_id in flight_ids:
            crew_leg_matches_set.add((crew_id, flight_id))
    
    # 构建layover_stations_set (简化处理)
    layover_stations_set = set()
    
    initial_rosters = []
    all_tasks = []
    for f in flights: all_tasks.append({'task_obj': f, 'type': 'flight', 'start_time': f.std, 'id': f.id, 'priority': 1})
    for gd in ground_duties: all_tasks.append({'task_obj': gd, 'type': 'ground_duty', 'start_time': gd.startTime, 'id': ('gd', gd.id), 'priority': 2})
    for bi in bus_info: all_tasks.append({'task_obj': bi, 'type': 'bus', 'start_time': bi.startTime, 'id': ('bus', bi.id), 'priority': 3})

    # 优先分配飞行任务，然后是地面任务，最后是大巴
    # 在相同类型任务中，按开始时间排序
    sorted_tasks = sorted(all_tasks, key=lambda t: (t['priority'], t['start_time']))
    unassigned_task_ids = {t['id'] for t in sorted_tasks}

    for crew in crews:
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
        crew.current_cycle_days = 0
        crew.current_cycle_at_base = (crew.stayStation == crew.base)
        crew.total_flight_duty_time_in_period = timedelta(0)
        crew.is_on_ground_duty = False
        crew.current_ground_duty_end_time = None
        crew.consecutive_duty_days = 0 
        crew.last_fdp_end_time_for_cycle_check = None
        crew.fdp_phase = 'none'  # FDP 阶段: 'none', 'pre_flight', 'in_flight', 'post_flight'

        while True:
            best_task_to_assign = None
            for task_info in sorted_tasks:
                task_obj = task_info['task_obj']
                task_type = task_info['type']
                task_id = task_info['id']

                if task_id in unassigned_task_ids:
                    if can_assign_task_greedy(crew, task_obj, task_type, crew_leg_matches_set, layover_stations_set, start_date):
                        best_task_to_assign = task_info
                        break
            
            if best_task_to_assign:
                assign_task_greedy(crew, best_task_to_assign['task_obj'], best_task_to_assign['type'], start_date)
                unassigned_task_ids.remove(best_task_to_assign['id'])
            else:
                break 
        
        if crew.schedule:
            # 转换为Roster格式，使用评分系统计算正确的成本
            roster = Roster(crew_id=crew.crewId, duties=crew.schedule, cost=0)
            
            # 如果提供了layover_stations，使用评分系统计算成本
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

    print(f"启发式算法成功生成 {len(initial_rosters)} 个初始排班方案。")
    unassigned_flight_ids = {uid for uid in unassigned_task_ids if not (isinstance(uid, tuple) and (uid[0] == 'bus' or uid[0] == 'gd'))}
    print(f"仍有 {len(unassigned_task_ids)} 个任务未被分配。")
    print(f"其中未分配的航班数量: {len(unassigned_flight_ids)}")
    
    # 输出初始解到CSV文件
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, "initial_solution.csv")
    write_results_to_csv(initial_rosters, output_path)
    
    return initial_rosters
