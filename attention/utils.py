# utils.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from . import config

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

        # 时间格式转换
        time_cols = {
            'flights': ['std', 'sta'],
            'ground_duties': ['startTime', 'endTime'],
            'bus_info': ['td', 'ta']
        }
        for name, cols in time_cols.items():
            for col in cols:
                data[name][col] = pd.to_datetime(data[name][col])
        
        # 建立快速查找字典
        self.crew_leg_map = data['crew_leg_match'].groupby('crewId')['legId'].apply(set).to_dict()
        data['ground_duties'] = data['ground_duties'].sort_values(['crewId', 'startTime'])
        self.crew_ground_duties = data['ground_duties'].groupby('crewId').apply(lambda x: x.to_dict('records')).to_dict()
        self.layover_airports = set(data['layover_stations']['airport'])
        
        # 将所有可分配任务统一格式化，便于后续处理
        self.tasks_df = self._unify_tasks(data)
        data['unified_tasks'] = self.tasks_df
        
        print("Data loaded and preprocessed.")
        return data

    def _unify_tasks(self, data):
        """将航班和巴士信息统一到一个DataFrame中"""
        flights = data['flights'].copy()
        flights['type'] = 'flight'
        flights.rename(columns={
            'id': 'taskId', 'std': 'startTime', 'sta': 'endTime', 
            'depaAirport': 'depaAirport', 'arriAirport': 'arriAirport'}, inplace=True)
        
        buses = data['bus_info'].copy()
        buses['type'] = 'positioning_bus'
        buses.rename(columns={
            'id': 'taskId', 'td': 'startTime', 'ta': 'endTime',
            'depaAirport': 'depaAirport', 'arriAirport': 'arriAirport'}, inplace=True)
        
        # 飞行置位是特殊的航班，我们在决策时动态创建
        # 这里只合并基础任务
        unified = pd.concat([
            flights[['taskId', 'type', 'startTime', 'endTime', 'depaAirport', 'arriAirport', 'fleet', 'aircraftNo', 'flyTime']],
            buses[['taskId', 'type', 'startTime', 'endTime', 'depaAirport', 'arriAirport']]
        ], ignore_index=True)
        unified['flyTime'].fillna(0, inplace=True)
        return unified


def identify_duties_and_cycles(roster, ground_duties):
    """
    核心辅助函数：从一个机长的完整排班中识别出值勤日和飞行周期
    roster: 按时间排序的任务列表
    """
    if not roster:
        return [], []

    all_tasks = sorted(roster + ground_duties, key=lambda x: x['startTime'])
    
    # 识别值勤日 (Duties)
    duties = []
    current_duty = []
    for i, task in enumerate(all_tasks):
        if not current_duty:
            current_duty.append(task)
        else:
            last_task_in_duty = current_duty[-1]
            rest_time = task['startTime'] - last_task_in_duty['endTime']
            # 根据规则7，休息时间小于12小时则认为是连续值勤
            if rest_time < timedelta(hours=12):
                current_duty.append(task)
            else:
                duties.append(current_duty)
                current_duty = [task]
    if current_duty:
        duties.append(current_duty)

    # 识别飞行周期 (Flight Cycles)
    # 规则8: 飞行周期开始前需要至少2个完整日历日的休息
    cycles = []
    current_cycle = []
    for i, duty in enumerate(duties):
        if not current_cycle:
             current_cycle.append(duty)
        else:
            last_duty_in_cycle = current_cycle[-1]
            last_task_in_last_duty = last_duty_in_cycle[-1]
            
            # 计算两个值勤日之间的休息天数
            rest_start = last_task_in_last_duty['endTime']
            rest_end = duty[0]['startTime']
            
            # 计算跨越的午夜数量
            full_days_rest = (rest_end.date() - rest_start.date()).days
            if rest_end.time() < rest_start.time():
                 full_days_rest -= 1
            
            if full_days_rest >= 2:
                cycles.append(current_cycle)
                current_cycle = [duty]
            else:
                current_cycle.append(duty)
    if current_cycle:
        cycles.append(current_cycle)
    
    return duties, cycles

class RuleChecker:
    """
    一个完整的规则检查器，用于验证排班计划的可行性。
    """
    def __init__(self, data_handler):
        self.dh = data_handler

    def check_full_roster(self, roster, crew_info):
        """检查一个完整的机长排班，返回总违规次数"""
        if not roster:
            return 0
            
        violations = 0
        sorted_roster = sorted(roster, key=lambda x: x['startTime'])
        ground_duties = self.dh.crew_ground_duties.get(crew_info['crewId'], [])
        
        duties, cycles = identify_duties_and_cycles(sorted_roster, ground_duties)
        
        total_flight_duty_time = 0

        for duty in duties:
            # 过滤出飞行值勤日（包含飞行或置位任务）
            flight_duty_tasks = [t for t in duty if t.get('type') in ['flight', 'positioning_flight', 'positioning_bus']]
            if not flight_duty_tasks:
                continue

            # 规则1: 置位规则
            positioning_tasks = [t for t in flight_duty_tasks if 'positioning' in t.get('type','')]
            if positioning_tasks:
                if not (positioning_tasks[0] == flight_duty_tasks[0] or positioning_tasks[-1] == flight_duty_tasks[-1]):
                    violations += 1 # 假设一个值勤日最多只有一个置位
            
            # 规则3: 最小连接时间
            for i in range(len(flight_duty_tasks) - 1):
                task1, task2 = flight_duty_tasks[i], flight_duty_tasks[i+1]
                interval = task2['startTime'] - task1['endTime']
                if 'bus' in task1.get('type','') or 'bus' in task2.get('type',''):
                    if interval < timedelta(hours=2): violations += 1
                elif task1.get('aircraftNo') != task2.get('aircraftNo'):
                    if interval < timedelta(hours=3): violations += 1

            # 规则4: 任务数量限制
            flight_tasks_count = sum(1 for t in flight_duty_tasks if t.get('type') == 'flight')
            if flight_tasks_count > 4: violations += flight_tasks_count - 4
            if len(flight_duty_tasks) > 6: violations += len(flight_duty_tasks) - 6

            # 规则5: 最大飞行时间
            duty_fly_time = sum(t.get('flyTime', 0) for t in flight_duty_tasks if t.get('type') == 'flight') / 60.0
            if duty_fly_time > 8: violations += 1

            # 规则6: 最大飞行值勤时间
            duty_start = flight_duty_tasks[0]['startTime']
            duty_end = [t for t in flight_duty_tasks if 'flight' in t.get('type','')][-1]['endTime']
            flight_duty_duration = (duty_end - duty_start).total_seconds() / 3600.0
            if flight_duty_duration > 12: violations += 1
            total_flight_duty_time += flight_duty_duration

        # 规则8: 飞行周期限制
        for cycle in cycles:
            cycle_start_date = cycle[0][0]['startTime'].date()
            cycle_end_date = cycle[-1][-1]['endTime'].date()
            if (cycle_end_date - cycle_start_date).days >= 4:
                violations += 1

        # 规则9: 总飞行值勤时间限制
        if total_flight_duty_time > 60: violations += 1
        
        # 其他规则在增量检查中处理，这里可以再次校验
        # 规则2, 10, 11, 12
        
        return violations

def calculate_final_score(roster_plan, data_handler):
    """根据竞赛说明计算最终得分"""
    dh = data_handler
    flights_df = dh.data['flights']
    crews_df = dh.data['crews']
    rule_checker = RuleChecker(dh)
    
    total_fly_hours = 0
    total_duty_calendar_days = 0
    overnight_stays = 0
    positioning_count = 0
    total_violations = 0
    
    assigned_flight_ids = set()

    for crew_id, tasks in roster_plan.items():
        if not tasks:
            continue
        
        # 确保任务按时间排序
        sorted_tasks = sorted(tasks, key=lambda x: x.get('startTime', x.get('endTime')))
        
        crew_info = crews_df[crews_df['crewId'] == crew_id].iloc[0].to_dict()
        ground_duties = dh.crew_ground_duties.get(crew_id, [])
        
        # 识别值勤日以计算总值勤日历日
        duties, _ = identify_duties_and_cycles(sorted_tasks, ground_duties)
        
        for duty in duties:
            if not duty: continue
            start_date = duty[0]['startTime'].date()
            end_date = duty[-1]['endTime'].date()
            total_duty_calendar_days += (end_date - start_date).days + 1

        # 遍历任务以计算外站过夜、飞行小时等
        for i in range(len(sorted_tasks) - 1):
             task1, task2 = sorted_tasks[i], sorted_tasks[i+1]
             
             # --- 错误修正开始 ---
             # 只有当task1是飞行或置位任务时，它才有'arriAirport'，才能计算外站过夜
             task1_arrival_airport = task1.get('arriAirport')
             if task1_arrival_airport is None:
                 # 如果是占位任务，它的地点是'airport'
                 task1_arrival_airport = task1.get('airport')

             if task1_arrival_airport and task1_arrival_airport != crew_info['base']:
                 overnight_days = (task2['startTime'].date() - task1['endTime'].date()).days
                 if overnight_days > 0:
                     overnight_stays += overnight_days
             # --- 错误修正结束 ---
        
        for task in sorted_tasks:
            task_type = task.get('type')
            if task_type == 'flight':
                total_fly_hours += task['flyTime'] / 60.0
                assigned_flight_ids.add(task['taskId'])
            elif task_type and 'positioning' in task_type:
                positioning_count += 1
                
        # 传入给规则检查器的应该是可分配任务，不含固有占位
        assignable_tasks = [t for t in sorted_tasks if 'groundDuty' not in t.get('type', '')]
        total_violations += rule_checker.check_full_roster(assignable_tasks, crew_info)

    uncovered_flights_count = len(flights_df) - len(assigned_flight_ids)
    
    avg_daily_fly_time = (total_fly_hours / total_duty_calendar_days) if total_duty_calendar_days > 0 else 0
    
    score = (avg_daily_fly_time * config.SCORE_FLY_TIME_MULTIPLIER +
             uncovered_flights_count * config.PENALTY_UNCOVERED_FLIGHT +
             overnight_stays * config.PENALTY_OVERNIGHT_STAY +
             positioning_count * config.PENALTY_POSITIONING +
             total_violations * config.PENALTY_RULE_VIOLATION)
             
    return score
