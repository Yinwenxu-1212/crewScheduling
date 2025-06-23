# file: subproblem_solver.py

import heapq
import itertools
from datetime import datetime, timedelta, date
from data_models import Crew, Flight, BusInfo, GroundDuty, Node, Roster, RestPeriod
from typing import List, Dict, Set, Optional
import csv
import os

# --- 全局变量，用于ML数据收集 ---
TRAINING_DATA_FILE = 'subproblem_training_data.csv'
write_header = not os.path.exists(TRAINING_DATA_FILE)
csv_writer = None
csv_file = None
CSV_HEADER = [
    'cost', 'current_airport_hash', 'current_time_weekday', 'current_time_hour',
    'duty_start_hour', 'duty_flight_time', 'duty_flight_count', 'duty_task_count',
    'total_flight_hours', 'total_positioning', 'total_away_overnights', 'total_calendar_days',
    'next_task_is_flight', 'next_task_fly_time', 'connection_hours',
    'next_task_dual_price',
    'label'
]
REWARD_PER_FLIGHT_HOUR = -1000
PENALTY_PER_AWAY_OVERNIGHT = 0.5
PENALTY_PER_POSITIONING = 0.5

# --- 规则常数 ---
MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT = timedelta(hours=3)
MIN_CONNECTION_TIME_BUS = timedelta(hours=2)
MAX_DUTY_DAY_HOURS = 12.0
MIN_REST_HOURS = 12.0
MAX_FLIGHTS_IN_DUTY = 4
MAX_TASKS_IN_DUTY = 6
MAX_FLIGHT_TIME_IN_DUTY_HOURS = 8.0

class Label:
    """标号类，包含所有用于计算得分和进行优势判断的资源"""
    def __init__(self, cost: float, path: List, node: Node,
                 duty_start_time: Optional[datetime], duty_flight_time: float,
                 duty_flight_count: int, duty_task_count: int,
                 total_flight_hours: float, total_positioning: int,
                 total_away_overnights: int, total_calendar_days: Set[date],
                 has_flown_in_duty: bool,used_task_ids: Optional[Set[str]],
                 tie_breaker: int):
        self.cost, self.path, self.node = cost, path, node
        self.duty_start_time, self.duty_flight_time = duty_start_time, duty_flight_time
        self.duty_flight_count, self.duty_task_count = duty_flight_count, duty_task_count
        self.total_flight_hours, self.total_positioning = total_flight_hours, total_positioning
        self.total_away_overnights, self.total_calendar_days = total_away_overnights, total_calendar_days
        self.has_flown_in_duty = has_flown_in_duty 
        self.used_task_ids = used_task_ids if used_task_ids is not None else set()
        self.tie_breaker = tie_breaker

    def __lt__(self, other):
        if abs(self.cost - other.cost) > 1e-6: return self.cost < other.cost
        return self.tie_breaker < other.tie_breaker
    def dominates(self, other: 'Label') -> bool:
        if self.cost > other.cost: return False
        if self.total_positioning > other.total_positioning: return False
        if self.total_away_overnights > other.total_away_overnights: return False
        if len(self.total_calendar_days) > len(other.total_calendar_days): return False
        if self.total_flight_hours < other.total_flight_hours: return False
        if self.cost < other.cost or self.total_flight_hours > other.total_flight_hours:
            return True
        return False

def is_conflicting(
    task_start_time: datetime, task_end_time: datetime, task_airport: str,
    ground_duties: List[GroundDuty]
) -> bool:
    """
    检查一个新任务是否与给定的、不可安排的地面任务列表冲突。
    """
    for duty in ground_duties:
        # 只有当任务和地面任务在同一个机场时，才需要检查时间冲突
        if task_airport == duty.airport:
            # 检查时间区间是否重叠
            if max(task_start_time, duty.startTime) < min(task_end_time, duty.endTime):
                return True
    return False

def find_positioning_tasks(from_airport, to_airport, earliest_start, all_bus, all_ddh):
    result = []
    for b in all_bus:
        if b.depaAirport == from_airport and b.arriAirport == to_airport and b.startTime >= earliest_start:
            result.append(b)
    for f in all_ddh:
        if f.depaAirport == from_airport and f.arriAirport == to_airport and f.std >= earliest_start:
            result.append(f)
    return result

def solve_subproblem_for_crew(
    crew: Crew, all_flights: List[Flight], all_bus_info: List[BusInfo],
    all_crew_ground_duties: List[GroundDuty], dual_prices: Dict[str, float],
    layover_stations: Dict[str, dict], crew_leg_match_dict: Dict[str, List[str]],
    crew_sigma_dual: float  # 添加第8个参数
) -> List[Roster]:
    
    global write_header, csv_writer, csv_file
    if csv_writer is None:
        csv_file = open(TRAINING_DATA_FILE, 'a', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER)
        if write_header:
            csv_writer.writeheader()
            write_header = False

    print(f"  [子问题] 开始为机组 {crew.crewId} 求解 (数据收集中)...")

    flights_by_id = {f.id: f for f in all_flights}
    qualified_flight_ids = crew_leg_match_dict.get(crew.crewId, [])
    crew_flights = [flights_by_id[fid] for fid in qualified_flight_ids if fid in flights_by_id]
     # --- 【GroundDuty】区分处理“值勤占位”和“休息占位” ---
    # “值勤占位”(isDuty=True)是可连接的任务
    schedulable_ground_duties = [gd for gd in all_crew_ground_duties if gd.isDuty and gd.crewId == crew.crewId]
    # “休息占位”(isDuty=False)或其他机长的任务，是需要避开的障碍
    conflicting_ground_duties = [gd for gd in all_crew_ground_duties if not gd.isDuty and gd.crewId == crew.crewId]

    all_tasks = [{'task': f, 'type': 'flight'} for f in crew_flights] + \
                [{'task': b, 'type': 'bus'} for b in all_bus_info] + \
                [{'task': gd, 'type': 'ground_duty'} for gd in schedulable_ground_duties]
    
    start_node = Node(airport=crew.stayStation, time=datetime(2025, 5, 28, 0, 0))
    tie_breaker = itertools.count()
    initial_label = Label(cost=0, path=[], node=start_node, duty_start_time=None,
                        duty_flight_time=0, duty_flight_count=0, duty_task_count=0,
                        total_flight_hours=0, total_positioning=0, total_away_overnights=0,
                        total_calendar_days=set(), has_flown_in_duty=False,
                        used_task_ids=set(),  
                        tie_breaker=next(tie_breaker))

    pq = [initial_label]
    visited: Dict[str, List[Label]] = {}
    found_rosters = []
    processed_labels_count = 0
    
    parent_map = {initial_label.tie_breaker: None}
    temp_feature_store = {}

    while pq:
        current_label = heapq.heappop(pq)
        processed_labels_count += 1

        if processed_labels_count % 10000 == 0:
            print(f"    [子问题进度] 机组 {crew.crewId}: 已处理 {processed_labels_count} 个标号...")

        current_node_airport = current_label.node.airport
        is_dominated = False
        if current_node_airport in visited:
            for existing_label in visited[current_node_airport]:
                if existing_label.dominates(current_label):
                    is_dominated = True; break
        if is_dominated: continue
        if current_node_airport not in visited: visited[current_node_airport] = []
        visited[current_node_airport] = [l for l in visited[current_node_airport] if not current_label.dominates(l)]
        visited[current_node_airport].append(current_label)
        
        # --- 值勤日必须在可过夜机场结束 ---
        if current_label.path and current_label.node.airport in layover_stations:
            rest_start_time = current_label.node.time
            rest_end_time = rest_start_time + timedelta(hours=MIN_REST_HOURS)
            if not is_conflicting(rest_start_time, rest_end_time, current_label.node.airport, conflicting_ground_duties):
                new_cost = current_label.cost + PENALTY_PER_AWAY_OVERNIGHT
                rest_period = RestPeriod(rest_start_time, rest_end_time, current_label.node.airport)
                new_path = current_label.path + [rest_period]
                new_node = Node(current_label.node.airport, rest_end_time)
                new_label = Label(cost=new_cost, path=new_path, node=new_node, duty_start_time=None,
                                duty_flight_time=0, duty_flight_count=0, duty_task_count=0,
                                total_flight_hours=current_label.total_flight_hours,
                                total_positioning=current_label.total_positioning,
                                total_away_overnights=current_label.total_away_overnights + 1,
                                total_calendar_days=current_label.total_calendar_days,
                                has_flown_in_duty=False, 
                                used_task_ids=current_label.used_task_ids,  # ✅ 加上这一行
                                tie_breaker=next(tie_breaker))
                heapq.heappush(pq, new_label)
                parent_map[new_label.tie_breaker] = (current_label.tie_breaker, None)

        for task_info in all_tasks:
            task = task_info['task']
            is_flight = task_info['type'] == 'flight'
            is_ground_duty = task_info['type'] == 'ground_duty'

            if hasattr(task, 'id') and task.id in current_label.used_task_ids:
                continue  # 跳过已经加入路径的任务

            if is_flight:
                task_dep_station, task_arr_station = task.depaAirport, task.arriAirport
                task_start_time, task_end_time = task.std, task.sta
            elif task_info['type'] == 'bus':  # Bus
                task_dep_station, task_arr_station = task.depaAirport, task.arriAirport
                task_start_time, task_end_time = task.startTime, task.endTime
            elif task_info['type'] == 'ground_duty':  # GroundDuty
                task_dep_station, task_arr_station = task.airport, task.airport
                task_start_time, task_end_time = task.startTime, task.endTime

            if task_dep_station != current_label.node.airport:
                positioning_tasks = find_positioning_tasks(
                    from_airport=current_label.node.airport,
                    to_airport=task_dep_station,
                    earliest_start=current_label.node.time,
                    all_bus=all_bus_info,
                    all_ddh=[f for f in crew_flights if f.flightNo.startswith("DH")]
                )

                for pos_task in positioning_tasks:
                    pos_start = pos_task.startTime if isinstance(pos_task, BusInfo) else pos_task.std
                    pos_end = pos_task.endTime if isinstance(pos_task, BusInfo) else pos_task.sta

                    # 连接时间是否足够 + 无冲突
                    if is_conflicting(pos_start, pos_end, pos_task.depaAirport, conflicting_ground_duties): continue
                    if is_conflicting(pos_end, task_start_time, task_dep_station, conflicting_ground_duties): continue
                    if pos_end > task_start_time - timedelta(hours=2): continue

                    # 构建路径: 当前 path + [置位, 任务]
                    new_path = current_label.path + [pos_task, task]
                    new_node = Node(airport=task_arr_station, time=task_end_time)
                    new_days = current_label.total_calendar_days.union({task_start_time.date(), task_end_time.date()})
                    new_cost = current_label.cost + PENALTY_PER_POSITIONING
                    if is_flight:
                        new_cost += REWARD_PER_FLIGHT_HOUR * (task.flyTime / 60.0)
                        new_cost -= dual_prices.get(task.id, 0)

                    heapq.heappush(pq, Label(
                        cost=new_cost,
                        path=new_path,
                        node=new_node,
                        duty_start_time=task_start_time if current_label.duty_start_time is None else current_label.duty_start_time,
                        duty_flight_time=current_label.duty_flight_time + (task.flyTime if is_flight else 0),
                        duty_flight_count=current_label.duty_flight_count + (1 if is_flight else 0),
                        duty_task_count=current_label.duty_task_count + 2,
                        total_flight_hours=current_label.total_flight_hours + (task.flyTime / 60.0 if is_flight else 0),
                        total_positioning=current_label.total_positioning + 1,
                        total_away_overnights=current_label.total_away_overnights,
                        total_calendar_days=new_days,
                        has_flown_in_duty=current_label.has_flown_in_duty or is_flight,
                        used_task_ids=current_label.used_task_ids.union({task.id}),
                        tie_breaker=next(tie_breaker)
                    ))
                continue

            # --- 值勤日必须从可过夜机场开始 ---
            if current_label.duty_start_time is None and current_label.path:
                # 如果即将开始一个新值勤日（即刚结束一次休息），那么当前机场必须是合法的过夜站
                if current_label.node.airport not in layover_stations:
                    continue
            
            last_task = current_label.path[-1] if current_label.path and isinstance(current_label.path[-1], Flight) else None
            min_connect_time = MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT if is_flight and last_task and task.aircraftNo != last_task.aircraftNo else MIN_CONNECTION_TIME_BUS if not is_flight else timedelta(hours=0)
            if (task_start_time - current_label.node.time) < min_connect_time: continue
            
            if is_conflicting(task_start_time, task_end_time, task_dep_station, conflicting_ground_duties): continue

            if current_label.duty_start_time is None:
                new_duty_start_time, new_duty_flight_time, new_duty_flight_count, new_duty_task_count = task_start_time, 0, 0, 0
                new_has_flown_in_duty = is_flight
            else:
                new_duty_start_time, new_duty_flight_time, new_duty_flight_count, new_duty_task_count = current_label.duty_start_time, current_label.duty_flight_time, current_label.duty_flight_count, current_label.duty_task_count
                new_has_flown_in_duty = current_label.has_flown_in_duty or is_flight

            if (task_end_time - new_duty_start_time).total_seconds() / 3600 > MAX_DUTY_DAY_HOURS: continue
            if (new_duty_task_count + 1) > MAX_TASKS_IN_DUTY: continue
            
            cost_increase = 0
            new_total_flight_hours, new_total_positioning = current_label.total_flight_hours, current_label.total_positioning
            new_cal_days = set()
            d = task_start_time.date()
            while d <= task_end_time.date():
                if d not in current_label.total_calendar_days: new_cal_days.add(d)
                d += timedelta(days=1)
            
            if is_flight:
                if (new_duty_flight_count + 1) > MAX_FLIGHTS_IN_DUTY: continue
                if (new_duty_flight_time + task.flyTime) / 60 > MAX_FLIGHT_TIME_IN_DUTY_HOURS: continue
                cost_increase += REWARD_PER_FLIGHT_HOUR * (task.flyTime / 60.0)
                cost_increase -= dual_prices.get(task.id, 0)
                new_duty_flight_time += task.flyTime
                new_duty_flight_count += 1
                new_total_flight_hours += task.flyTime / 60.0
            else: # Bus
                # --- 置位任务成本为0，仅作为连接手段 ---
                cost_increase += 0
                new_total_positioning += 1

            new_duty_task_count += 1
            
            features = {
                'cost': current_label.cost, 'current_airport_hash': hash(current_label.node.airport) % 1000,
                'current_time_weekday': current_label.node.time.weekday(), 'current_time_hour': current_label.node.time.hour,
                'duty_start_hour': current_label.duty_start_time.hour if current_label.duty_start_time else -1,
                'duty_flight_time': current_label.duty_flight_time, 'duty_flight_count': current_label.duty_flight_count,
                'duty_task_count': current_label.duty_task_count, 'total_flight_hours': current_label.total_flight_hours,
                'total_positioning': current_label.total_positioning, 'total_away_overnights': current_label.total_away_overnights,
                'total_calendar_days': len(current_label.total_calendar_days),
                'next_task_is_flight': 1 if is_flight else 0,
                'next_task_fly_time': task.flyTime if is_flight else 0,
                'connection_hours': (task_start_time - current_label.node.time).total_seconds() / 3600,
                'next_task_dual_price': dual_prices.get(task.id, 0) if is_flight else 0,
                'label': 0 
            }
            
            new_cost = current_label.cost + cost_increase
            next_node = Node(airport=task_arr_station, time=task_end_time)
            new_path = current_label.path + [task]
            
            new_label = Label(cost=new_cost, path=new_path, node=next_node,
                              duty_start_time=new_duty_start_time, duty_flight_time=new_duty_flight_time,
                              duty_flight_count=new_duty_flight_count, duty_task_count=new_duty_task_count,
                              total_flight_hours=new_total_flight_hours, total_positioning=new_total_positioning,
                              total_away_overnights=current_label.total_away_overnights,
                              total_calendar_days=current_label.total_calendar_days.union(new_cal_days),
                              has_flown_in_duty=new_has_flown_in_duty, # <--- 传入新状态
                              used_task_ids=current_label.used_task_ids.union({task.id}),
                              tie_breaker=next(tie_breaker))
            heapq.heappush(pq, new_label)
            parent_map[new_label.tie_breaker] = (current_label.tie_breaker, features)

            if next_node.airport == crew.base and new_cost < -1e-6:
                print(f"      [子问题发现!] 机组 {crew.crewId}: 找到负成本方案! Reduced Cost: {new_cost:.2f}")
                final_calendar_days_count = len(new_label.total_calendar_days) or 1
                avg_flight_hours = new_label.total_flight_hours / final_calendar_days_count
                true_score = (avg_flight_hours * 1000) - (new_label.total_away_overnights * PENALTY_PER_AWAY_OVERNIGHT) - (new_label.total_positioning * PENALTY_PER_POSITIONING)
                found_rosters.append(Roster(crew.crewId, new_path, -true_score))
                
                temp_label_tb = new_label.tie_breaker
                while temp_label_tb is not None:
                    parent_info = parent_map.get(temp_label_tb)
                    if parent_info and parent_info[1] is not None:
                        parent_tb, f = parent_info
                        f['label'] = 1
                        temp_label_tb = parent_tb
                    else: break
    
    for tb, parent_info in parent_map.items():
        if parent_info:
            _, features = parent_info
            if features:
                temp_feature_store[tb] = features
    
    for features in temp_feature_store.values():
        csv_writer.writerow(features)
    if csv_file: csv_file.flush()

    print(f"  [子问题] 机组 {crew.crewId} 求解完毕。找到 {len(found_rosters)} 个有效方案。")
    
    unique_rosters = []
    seen_rosters = set()
    for r in found_rosters:
        roster_tuple = tuple(getattr(d, 'id', str(d)) for d in r.duties)
        if roster_tuple not in seen_rosters:
            unique_rosters.append(r); seen_rosters.add(roster_tuple)
            
    return unique_rosters
