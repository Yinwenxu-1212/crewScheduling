# utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import config

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
            'bus_info': 'businfo.csv', 'layover_stations': 'layoverStation.csv'
        }
        data = {}
        for name, filename in data_files.items():
            file_path = os.path.join(self.path, filename)
            data[name] = pd.read_csv(file_path)
        
        data['ground_duties']['type'] = 'groundDuty'
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
        
        unified = pd.concat([flights, buses], ignore_index=True)
        unified['flyTime'].fillna(0, inplace=True)
        return unified


def identify_duties_and_cycles(roster, ground_duties):
    """
    核心辅助函数：从一个机长的完整排班中识别出值勤日和飞行周期。
    返回:
    - duties: 一个列表，每个元素是一个代表值勤日的任务列表。 e.g., [[t1, t2], [t3]]
    - cycles: 一个列表，每个元素是一个代表飞行周期的任务列表。
    """
    if not roster and not ground_duties:
        return [], []
        
    all_tasks = sorted(roster + ground_duties, key=lambda x: x['startTime'])
    
    # 识别值勤日 (Duties)
    duties = []
    if not all_tasks: return [], []
    
    current_duty = [all_tasks[0]]
    for i in range(1, len(all_tasks)):
        prev_task, curr_task = all_tasks[i-1], all_tasks[i]
        
        # 休息占位会断开值勤日
        is_rest_duty = prev_task.get('isDuty', 1) == 0 or curr_task.get('isDuty', 1) == 0
        rest_time = curr_task['startTime'] - prev_task['endTime']

        if rest_time < timedelta(hours=12) and not is_rest_duty:
            current_duty.append(curr_task)
        else:
            duties.append(current_duty)
            current_duty = [curr_task]
    if current_duty:
        duties.append(current_duty)

    # 识别飞行周期 (Flight Cycles)
    cycles = []
    if not duties: return [], []

    current_cycle = duties[0] # current_cycle现在是一个扁平的任务列表
    for i in range(1, len(duties)):
        last_duty_end_time = duties[i-1][-1]['endTime']
        current_duty_start_time = duties[i][0]['startTime']
        
        # 检查两个值勤日之间是否有至少2天的休息
        # 简化计算：如果休息时间超过48小时，就认为是新周期
        if (current_duty_start_time - last_duty_end_time) >= timedelta(hours=48):
             cycles.append(current_cycle)
             current_cycle = duties[i]
        else:
             current_cycle.extend(duties[i]) # 将新值勤日的任务合并进来
    if current_cycle:
        cycles.append(current_cycle)
    
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
        
        # identify_duties_and_cycles 的输入应该是 assignable_tasks
        duties, cycles = identify_duties_and_cycles(roster, ground_duties)
        
        total_flight_duty_time = 0

        for duty in duties:
            flight_duty_tasks = [t for t in duty if t.get('type') in ['flight', 'positioning_flight', 'positioning_bus']]
            if not flight_duty_tasks:
                continue

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
            if sum(t.get('flyTime', 0) for t in flight_duty_tasks) / 60.0 > 8: violations += 1

            # Rule 6: 最大飞行值勤时间
            duty_start = flight_duty_tasks[0]['startTime']
            flight_tasks_in_duty = [t for t in flight_duty_tasks if 'flight' in t.get('type','')]
            if flight_tasks_in_duty:
                duty_end = flight_tasks_in_duty[-1]['endTime']
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

        # Rule 9: 总飞行值勤时间限制
        if total_flight_duty_time > 60: violations += 1
        
        return violations
    
def calculate_final_score(roster_plan, data_handler):
    """根据竞赛说明计算最终得分 (已更新)"""
    dh = data_handler
    flights_df, crews_df = dh.data['flights'], dh.data['crews']
    rule_checker = RuleChecker(dh)
    
    total_fly_hours, total_duty_calendar_days, overnight_stays, positioning_count, total_violations = 0, 0, 0, 0, 0
    
    # --- 核心修改：只记录被“执飞”的航班 ---
    covered_flight_ids = set()

    for crew_id, tasks in roster_plan.items():
        if not tasks: continue
        
        assignable_tasks = [t for t in tasks if t.get('type') != 'groundDuty']
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
            if task_type == 'flight':
                # 只有执飞任务才累加飞行小时和计入覆盖
                total_fly_hours += task.get('flyTime', 0) / 60.0
                covered_flight_ids.add(task['taskId'])
            elif 'positioning' in task_type:
                positioning_count += 1
                
        total_violations += rule_checker.check_full_roster(sorted_tasks, crew_info)

    # 用总航班数减去“被执飞”的航班数
    uncovered_flights_count = len(flights_df) - len(covered_flight_ids)
    
    avg_daily_fly_time = (total_fly_hours / total_duty_calendar_days) if total_duty_calendar_days > 0 else 0
    
    score = (avg_daily_fly_time * config.SCORE_FLY_TIME_MULTIPLIER +
             uncovered_flights_count * config.PENALTY_UNCOVERED_FLIGHT +
             overnight_stays * config.PENALTY_OVERNIGHT_STAY_AWAY_FROM_BASE +
             positioning_count * config.PENALTY_POSITIONING +
             total_violations * config.PENALTY_RULE_VIOLATION)
             
    return score
