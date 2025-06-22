# file: initial_solution_generator.py

from datetime import datetime, timedelta
from typing import List, Dict
from data_models import Flight, Crew, BusInfo, GroundDuty, Roster

# --- 规则常数，与subproblem_solver保持一致 ---
MAX_DUTY_DAY_HOURS = 12.0
MAX_FLIGHTS_IN_DUTY = 4
MAX_TASKS_IN_DUTY = 6
MIN_CONNECTION_TIME_BUS = timedelta(hours=2)
MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT = timedelta(hours=3)

def get_task_info(task):
    """一个辅助函数，用于从不同类型的任务对象中安全地提取通用信息"""
    if isinstance(task, Flight):
        return task.depaAirport, task.arriAirport, task.std, task.sta
    elif isinstance(task, BusInfo):
        return task.depaAirport, task.arriAirport, task.startTime, task.endTime
    return None, None, None, None

def is_roster_valid(roster: Roster, new_task, crew_ground_duties: list, crew_stay_station: str) -> bool:
    """
    一个更完整的验证函数，确保生成的初始解尽可能合法。
    """
    new_task_dep, _, new_task_start, new_task_end = get_task_info(new_task)
    if new_task_start is None: return False

    # 规则11: 与地面任务冲突
    if any(max(new_task_start, gd.startTime) < min(new_task_end, gd.endTime) for gd in crew_ground_duties):
        return False
        
    last_task = roster.duties[-1] if roster.duties else None

    if not last_task:
        # 这是第一个任务，必须从 stayStation 出发 (规则2)
        return new_task_dep == crew_stay_station
    
    # 如果不是第一个任务，检查过程衔接
    _, last_task_arr, _, last_task_end = get_task_info(last_task)
    if new_task_dep != last_task_arr:
        return False

    # 规则3: 最小衔接时间
    min_connect = MIN_CONNECTION_TIME_BUS if isinstance(new_task, BusInfo) else (MIN_CONNECTION_TIME_FLIGHT_DIFFERENT_AIRCRAFT if isinstance(new_task, Flight) and isinstance(last_task, Flight) and new_task.aircraftNo != last_task.aircraftNo else timedelta(0))
    if (new_task_start - last_task_end) < min_connect:
        return False

    # 检查值勤日约束
    # 一个简单的启发式：如果与上个任务间隔超过12小时，就认为是新值勤日
    if (new_task_start - last_task_end).total_seconds() / 3600 >= 12.0:
        duty_tasks_in_day = [new_task]
    else:
        # 这里的追溯逻辑可以更复杂，暂时简化为从roster开头算起
        duty_tasks_in_day = roster.duties + [new_task] 

    # 规则6: 最大值勤时长
    duty_start_time = get_task_info(duty_tasks_in_day[0])[2]
    duty_end_time = get_task_info(duty_tasks_in_day[-1])[3]
    if (duty_end_time - duty_start_time).total_seconds() / 3600 > MAX_DUTY_DAY_HOURS:
        return False

    # 规则4: 任务数量
    if len(duty_tasks_in_day) > MAX_TASKS_IN_DUTY: return False
    if sum(1 for t in duty_tasks_in_day if isinstance(t, Flight)) > MAX_FLIGHTS_IN_DUTY: return False
        
    return True

def generate_initial_rosters_with_heuristic(
    flights: List[Flight], crews: List[Crew], bus_info: List[BusInfo], 
    ground_duties: List[GroundDuty], crew_leg_match_dict: dict
) -> List[Roster]:
    """
    使用修正后的快速贪心启发式算法生成初始解。
    """
    print("正在使用启发式算法生成初始解...")
    
    # 将所有可分配的任务放入一个池子
    all_assignable_tasks = [(f.id, 'flight', f) for f in flights] + [(b.id, 'bus', b) for b in bus_info]
    unassigned_tasks = {(tid, ttype) for tid, ttype, _ in all_assignable_tasks}
    
    initial_rosters = []
    
    # 遍历每一位机长
    for crew in crews:
        current_roster = Roster(crew_id=crew.crewId, duties=[], cost=0)
        crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]
        
        while True:
            best_next_task_info = None
            min_connection_time = float('inf')
            
            # 寻找可以衔接在当前排班表末尾的、最优的下一个任务
            
            # 遍历所有待选任务
            for task_id, task_type, task_obj in all_assignable_tasks:
                if (task_id, task_type) not in unassigned_tasks: continue

                # 检查资质
                if task_type == 'flight':
                    if crew.crewId not in crew_leg_match_dict.get(task_id, []):
                        continue

                # 检查合法性
                if is_roster_valid(current_roster, task_obj, crew_specific_gds, crew.stayStation):
                    
                    if current_roster.duties:
                        _, _, _, last_task_end = get_task_info(current_roster.duties[-1])
                    else: # 第一个任务
                        last_task_end = datetime(2025, 5, 28, 0, 0)
                    
                    _, _, new_task_start, _ = get_task_info(task_obj)
                    
                    connection_time = (new_task_start - last_task_end).total_seconds()
                    
                    if connection_time < min_connection_time:
                        min_connection_time = connection_time
                        best_next_task_info = (task_obj, task_type)
            
            if best_next_task_info:
                best_task, best_task_type = best_next_task_info
                current_roster.duties.append(best_task)
                unassigned_tasks.remove((best_task.id, best_task_type))
            else:
                break
                
        if current_roster.duties:
            current_roster.cost = -1000 * sum(f.flyTime / 60.0 for f in current_roster.duties if isinstance(f, Flight))
            initial_rosters.append(current_roster)

    print(f"启发式算法成功生成 {len(initial_rosters)} 个初始排班方案。")
    print(f"仍有 {len(unassigned_tasks)} 个任务未被分配。")
    return initial_rosters