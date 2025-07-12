# utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
try:
    from . import config
except ImportError:
    import config

# 休息占位任务处理函数
def is_rest_ground_duty(task):
    """
    判断是否为休息占位任务
    根据isDuty字段判断：isDuty=0表示休息占位任务，isDuty=1表示工作占位任务
    """
    if task.get('type') != 'ground_duty':
        return False
    
    # 根据isDuty字段判断：0表示休息，1表示工作
    return task.get('isDuty', 1) == 0

def calculate_rest_period(prev_task_end, next_task_start):
    """
    计算两个任务之间的休息时间（小时）
    """
    if prev_task_end and next_task_start:
        rest_duration = next_task_start - prev_task_end
        return rest_duration.total_seconds() / 3600.0
    return 0

def get_rest_quality_bonus(rest_hours):
    """
    根据休息时间长度给予奖励
    """
    if rest_hours >= config.MIN_REST_PERIOD_HOURS:
        return config.REST_PERIOD_BONUS
    return 0

class DataHandler:
    """数据加载和预处理"""
    def __init__(self, path=config.DATA_PATH):
        self.path = path
        self.data = self._load_and_preprocess()

    def _load_and_preprocess(self):
        print("Loading and preprocessing data...")
        data_files = {
            'flights': 'flight.csv', 'crews': 'crew.csv',
            'crew_leg_match': 'crewLegMatch.csv', 'ground_duties': 'groundDuty.csv',
            'bus_info': 'busInfo.csv', 'layover_stations': 'layoverStation.csv'
        }
        data = {}
        for name, filename in data_files.items():
            file_path = os.path.join(self.path, filename)
            data[name] = pd.read_csv(file_path)
        
        data['ground_duties']['type'] = 'ground_duty'
        time_cols = {'flights': ['std', 'sta'], 'ground_duties': ['startTime', 'endTime'], 'bus_info': ['td', 'ta']}
        for name, cols in time_cols.items():
            for col in cols:
                data[name][col] = pd.to_datetime(data[name][col])
        
        self.crew_leg_map = data['crew_leg_match'].groupby('crewId')['legId'].apply(set).to_dict()
        data['ground_duties'] = data['ground_duties'].sort_values(['crewId', 'startTime'])
        self.crew_ground_duties = data['ground_duties'].groupby('crewId').apply(lambda x: x.to_dict('records')).to_dict()
        self.layover_airports = set(data['layover_stations']['airport'])
        
        self.tasks_df = self._unify_tasks(data)
        data['unified_tasks'] = self.tasks_df
        print("Data loaded and preprocessed.")
        return data

    def _unify_tasks(self, data):
        flights = data['flights'].copy()
        flights.rename(columns={'id': 'taskId', 'std': 'startTime', 'sta': 'endTime'}, inplace=True)
        flights['type'] = 'flight'
        
        buses = data['bus_info'].copy()
        buses.rename(columns={'id': 'taskId', 'td': 'startTime', 'ta': 'endTime'}, inplace=True)
        buses['type'] = 'positioning_bus'
        
        # 添加占位任务（ground_duties）
        ground_duties = data['ground_duties'].copy()
        ground_duties.rename(columns={'id': 'taskId', 'startTime': 'startTime', 'endTime': 'endTime'}, inplace=True)
        ground_duties['type'] = 'ground_duty'
        # 占位任务的出发地和到达地都是同一个机场
        ground_duties['depaAirport'] = ground_duties['airport']
        ground_duties['arriAirport'] = ground_duties['airport']
        
        unified = pd.concat([flights, buses, ground_duties], ignore_index=True)
        unified = unified.fillna({'flyTime': 0})
        return unified


def calculate_complete_calendar_days(end_time, start_time):
    """
    计算两个时间点之间的完整日历日数量
    完整日历日：从end_time的次日00:00到start_time的前日23:59:59
    """
    from datetime import timedelta
    
    # 获取结束时间的次日和开始时间的前日
    end_date = end_time.date()
    start_date = start_time.date()
    
    # 计算完整日历日数量
    # 如果start_date <= end_date，说明没有完整的日历日
    if start_date <= end_date:
        return 0
    
    # 计算完整日历日数量
    complete_days = (start_date - end_date).days - 1
    return max(0, complete_days)

def can_start_new_flight_cycle(last_duty_end_time, last_duty_end_location, 
                                current_task_start_time, crew_base):
    """判断是否可以开始新的飞行周期"""
    
    # 条件1：必须从基地出发
    if last_duty_end_location != crew_base:
        return False, "上一个值勤日未在基地结束，无法开始新飞行周期"
    
    # 条件2：必须有2个完整日历日的休息
    rest_days = calculate_complete_calendar_days(
        last_duty_end_time, 
        current_task_start_time
    )
    
    if rest_days < 2:
        return False, f"休息不足2个完整日历日（只有{rest_days}天）"
    
    # 条件3：休息期间必须一直在基地
    # 由于上一个值勤在基地结束，且中间没有任务，可以认为一直在基地
    
    return True, "可以开始新飞行周期"

def is_valid_flight_cycle(cycle_duties):
    """检查飞行周期是否有效"""
    if not cycle_duties:
        return False
    
    # 检查是否包含飞行任务
    has_flight = False
    for duty in cycle_duties:
        for task in duty:
            if task.get('type') == 'flight':
                has_flight = True
                break
        if has_flight:
            break
    
    return has_flight

def flatten_cycle(cycle_duties):
    """将周期中的值勤日列表扁平化为任务列表"""
    flattened = []
    for duty in cycle_duties:
        flattened.extend(duty)
    return flattened

def identify_duties_and_cycles(roster, ground_duties, crew_base=None):
    """
    核心辅助函数：从一个机长的完整排班中识别出值勤日和飞行周期。
    修正版本：正确处理飞行周期的基地休息要求和2个完整日历日休息，考虑休息占位任务
    返回:
    - duties: 一个列表，每个元素是一个代表值勤日的任务列表。 e.g., [[t1, t2], [t3]]
    - cycles: 一个列表，每个元素是一个代表飞行周期的任务列表。
    """
    if not roster and not ground_duties:
        return [], []
        
    all_tasks = sorted(roster + ground_duties, key=lambda x: x['startTime'])
    
    # 识别值勤日 (Duties) - 修正版本
    duties = []
    if not all_tasks: return [], []
    
    current_duty = [all_tasks[0]]
    for i in range(1, len(all_tasks)):
        prev_task, curr_task = all_tasks[i-1], all_tasks[i]
        
        # 计算任务间隔时间
        interval = curr_task['startTime'] - prev_task['endTime']
        
        # 值勤日断开条件：
        # 1. 间隔超过12小时
        # 2. 当前任务是休息占位任务
        should_break_duty = False
        
        if interval >= timedelta(hours=12):
            should_break_duty = True
        elif is_rest_ground_duty(curr_task):
            should_break_duty = True
        
        if should_break_duty:
            duties.append(current_duty)
            current_duty = [curr_task]
        else:
            current_duty.append(curr_task)
    
    if current_duty:
        duties.append(current_duty)

    # 识别飞行周期 (Flight Cycles) - 修正版本
    cycles = []
    if not duties: return [], []
    
    # 如果没有提供crew_base，尝试从任务中推断
    if crew_base is None:
        # 假设第一个任务的出发地是基地（这是一个简化假设）
        crew_base = duties[0][0].get('depaAirport', 'UNKNOWN')
    
    current_cycle = None
    last_duty_end_location = crew_base  # 假设开始时在基地
    
    for i, duty in enumerate(duties):
        duty_start_location = duty[0].get('depaAirport')
        duty_end_location = duty[-1].get('arriAirport')
        
        # 判断是否包含飞行任务
        has_flight = any(t.get('type') == 'flight' for t in duty)
        
        if i == 0:
            # 第一个值勤日
            if crew_base == duty_start_location:
                # 从基地出发，可以开始飞行周期
                current_cycle = [duty] if has_flight else []
            else:
                # 不从基地出发，不能开始飞行周期
                current_cycle = []
        else:
            prev_duty = duties[i-1]
            prev_end_time = prev_duty[-1]['endTime']
            prev_end_location = prev_duty[-1].get('arriAirport')
            
            # 检查是否满足新飞行周期条件
            can_start, reason = can_start_new_flight_cycle(
                prev_end_time,
                prev_end_location,
                duty[0]['startTime'],
                crew_base
            )
            
            if can_start:
                # 可以开始新周期
                if current_cycle and is_valid_flight_cycle(current_cycle):
                    cycles.append(flatten_cycle(current_cycle))
                current_cycle = [duty] if has_flight else []
            else:
                # 继续当前周期（如果有的话）
                if current_cycle is not None:
                    current_cycle.append(duty)
        
        last_duty_end_location = duty_end_location
    
    # 处理最后的周期
    if current_cycle and is_valid_flight_cycle(current_cycle):
        cycles.append(flatten_cycle(current_cycle))
    
    return duties, cycles


# utils.py (修正后的 RuleChecker)

class RuleChecker:
    def __init__(self, data_handler: DataHandler):
        self.dh = data_handler

    def check_full_roster(self, roster, crew_info):
        """检查一个完整的机长排班，返回总违规次数"""
        if not roster:
            return 0
            
        violations = 0
        # ground_duties 应该从 data_handler 获取，而不是 roster
        ground_duties = self.dh.crew_ground_duties.get(crew_info['crewId'], [])
        
        # 获取机组基地信息
        crew_base = crew_info.get('base') or crew_info.get('stayStation')
        
        # identify_duties_and_cycles 的输入应该是 assignable_tasks，并传入crew_base
        duties, cycles = identify_duties_and_cycles(sorted_tasks, ground_duties, crew_base)
        
        # Rule 2: 地点衔接规则 - 第一个任务必须从stayStation开始
        if roster and roster[0].get('depaAirport') != crew_info.get('stayStation'):
            violations += 1
        
        total_flight_duty_time = 0

        for duty in duties:
            flight_duty_tasks = [t for t in duty if t.get('type') in ['flight', 'positioning_flight', 'positioning_bus']]
            if not flight_duty_tasks:
                continue

            # 检查是否为飞行值勤日（包含飞行任务）
            has_flight_task = any(t.get('type') == 'flight' for t in flight_duty_tasks)
            
            # Rule 1: 置位规则
            pos_indices = [i for i, t in enumerate(flight_duty_tasks) if 'positioning' in t.get('type','')]
            is_pos_in_middle = any(0 < i < len(flight_duty_tasks) - 1 for i in pos_indices)
            if is_pos_in_middle:
                 violations += 1
            
            # Rule 3: 最小连接时间
            for i in range(len(flight_duty_tasks) - 1):
                t1, t2 = flight_duty_tasks[i], flight_duty_tasks[i+1]
                interval = t2['startTime'] - t1['endTime']
                if 'bus' in t1.get('type','') or 'bus' in t2.get('type',''):
                    if interval < timedelta(hours=2): violations += 1
                elif t1.get('aircraftNo') != t2.get('aircraftNo'):
                    if interval < timedelta(hours=3): violations += 1
            
            # Rule 4: 任务数量限制
            if sum(1 for t in flight_duty_tasks if t.get('type') == 'flight') > 4: violations += 1
            if len(flight_duty_tasks) > 6: violations += 1

            # Rule 5: 最大飞行时间
            if sum(t.get('flyTime', 0) for t in flight_duty_tasks if t.get('type') == 'flight' and not t.get('is_positioning', False)) / 60.0 > 8: violations += 1

            # Rule 6: 最大飞行值勤时间（修正版）
            # 飞行值勤时间 = 飞行值勤日的总时长（从第一个任务开始到最后一个任务结束）
            if has_flight_task:  # 只有飞行值勤日才计算飞行值勤时间
                duty_start = flight_duty_tasks[0]['startTime']
                duty_end = flight_duty_tasks[-1]['endTime']  # 修正：使用最后一个任务的结束时间
                flight_duty_duration = (duty_end - duty_start).total_seconds() / 3600.0
                if flight_duty_duration > 12: violations += 1
                total_flight_duty_time += flight_duty_duration
        
        # --- 错误修正开始 ---
        # Rule 8: 飞行周期限制
        for cycle in cycles:
            # 'cycle' 本身就是一个扁平的任务列表，代表一个飞行周期内的所有任务
            flight_tasks_in_cycle = [t for t in cycle if t.get('type') in ['flight', 'positioning_flight', 'positioning_bus']]
            if flight_tasks_in_cycle:
                start_date = flight_tasks_in_cycle[0]['startTime'].date()
                end_date = flight_tasks_in_cycle[-1]['endTime'].date()
                # 飞行周期最多持续四个日历日
                if (end_date - start_date).days > 3:
                    violations += 1
        # --- 错误修正结束 ---

        # Rule 7: 最小休息时间规则 - 值勤日之间至少12小时休息
        for i in range(len(duties) - 1):
            if duties[i] and duties[i+1]:
                rest_time = duties[i+1][0]['startTime'] - duties[i][-1]['endTime']
                if rest_time < timedelta(hours=12):
                    violations += 1
        
        # Rule 9: 总飞行值勤时间限制
        if total_flight_duty_time > 60: violations += 1
        
        return violations
    
def calculate_final_score(roster_plan, data_handler):
    """根据竞赛说明计算最终得分 (已更新)"""
    dh = data_handler
    flights_df, crews_df = dh.data['flights'], dh.data['crews']
    rule_checker = RuleChecker(dh)
    
    total_fly_hours, total_duty_calendar_days, overnight_stays, positioning_count, total_violations = 0, 0, 0, 0, 0
    
    # --- 核心修改：只记录被"执飞"的航班 ---
    covered_flight_ids = set()
    covered_ground_duties = 0  # 新增：统计覆盖的占位任务数量

    for crew_id, tasks in roster_plan.items():
        if not tasks: continue
        
        assignable_tasks = [t for t in tasks if t.get('type') != 'ground_duty']
        if not assignable_tasks: continue

        sorted_tasks = sorted(assignable_tasks, key=lambda x: x['startTime'])
        crew_info = crews_df[crews_df['crewId'] == crew_id].iloc[0].to_dict()
        ground_duties = dh.crew_ground_duties.get(crew_id, [])
        duties, _ = identify_duties_and_cycles(sorted_tasks, ground_duties)
        
        for duty in duties:
            if not duty: continue
            start_date, end_date = duty[0]['startTime'].date(), duty[-1]['endTime'].date()
            total_duty_calendar_days += (end_date - start_date).days + 1

        for i in range(len(sorted_tasks) - 1):
             t1, t2 = sorted_tasks[i], sorted_tasks[i+1]
             arr_airport = t1.get('arriAirport')
             if arr_airport and arr_airport != crew_info['base']:
                 overnight_days = (t2['startTime'].date() - t1['endTime'].date()).days
                 if overnight_days > 0: overnight_stays += overnight_days
        
        for task in sorted_tasks:
            task_type = task.get('type', '')
            if task_type == 'flight' and not task.get('is_positioning', False):
                # 只有执飞任务才累加飞行小时和计入覆盖
                total_fly_hours += task.get('flyTime', 0) / 60.0
                covered_flight_ids.add(task['taskId'])
            elif 'positioning' in task_type:
                positioning_count += 1
            elif task_type == 'ground_duty':
                covered_ground_duties += 1
                
        total_violations += rule_checker.check_full_roster(sorted_tasks, crew_info)

    # 用总航班数减去“被执飞”的航班数
    uncovered_flights_count = len(flights_df) - len(covered_flight_ids)
    
    avg_daily_fly_time = (total_fly_hours / total_duty_calendar_days) if total_duty_calendar_days > 0 else 0
    
    # 计算占位任务覆盖奖励
    ground_duty_bonus = covered_ground_duties * config.GROUND_DUTY_COVERAGE_REWARD
    
    score = (avg_daily_fly_time * config.SCORE_FLY_TIME_MULTIPLIER +
             uncovered_flights_count * config.PENALTY_UNCOVERED_FLIGHT +
             overnight_stays * config.PENALTY_OVERNIGHT_STAY_AWAY_FROM_BASE +
             positioning_count * config.PENALTY_POSITIONING +
             total_violations * config.PENALTY_RULE_VIOLATION +
             ground_duty_bonus)
             
    return score
