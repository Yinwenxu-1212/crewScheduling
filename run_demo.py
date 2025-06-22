# file: run_demo.py
# 职责: 创建一个迷你的、可控的测试场景，用于快速调试和验证整个列生成框架。

import time
from datetime import datetime, timedelta
from data_models import Flight, Crew, BusInfo, GroundDuty, Node, Roster, RestPeriod
from master_problem import MasterProblem
from subproblem_solver import solve_subproblem_for_crew
from results_writer import write_results_to_csv

def create_demo_data():
    """手动创建一个小型的、但逻辑完整的测试数据集"""
    print("--- Creating Demo Data ---")

    # 1. 创建几个航班对象
    flights = [
        # 一个可以由C001执行的、从基地出发的合法路径： F1 -> REST -> F3
        Flight(id='F1', flightNo='MU5101', depaAirport='SHA', arriAirport='PEK', std='2025/05/29 08:00', sta='2025/05/29 10:00', fleet='A320', aircraftNo='B-6001', flyTime=120),
        Flight(id='F2', flightNo='MU5102', depaAirport='PEK', arriAirport='SHA', std='2025/05/29 12:00', sta='2025/05/29 14:00', fleet='A320', aircraftNo='B-6001', flyTime=120),
        Flight(id='F3', flightNo='CA1832', depaAirport='PEK', arriAirport='SHA', std='2025/05/30 13:00', sta='2025/05/30 15:00', fleet='A320', aircraftNo='B-6002', flyTime=120),
        # 一个由C002执行的、从外站开始的航班
        Flight(id='F4', flightNo='3U8886', depaAirport='CTU', arriAirport='PEK', std='2025/05/29 09:00', sta='2025/05/29 11:30', fleet='A321', aircraftNo='B-6003', flyTime=150),
        # 一个额外的航班，用于测试未覆盖的情况
        Flight(id='F5', flightNo='HU7605', depaAirport='PEK', arriAirport='CAN', std='2025/05/29 18:00', sta='2025/05/29 21:00', fleet='B737', aircraftNo='B-7001', flyTime=180),
    ]

    # 2. 创建几个机长对象
    crews = [
        Crew(crewId='C001', base='SHA', stayStation='SHA'),
        Crew(crewId='C002', base='PEK', stayStation='CTU'),
    ]

    # 3. 创建资质匹配
    crew_leg_match_dict = {
        'C001': ['F1', 'F2', 'F3'],
        'C002': ['F4']
    }

    # 4. 创建巴士和其他数据
    bus_info = [
        BusInfo(id='B1', depaAirport='PEK', arriAirport='TSN', td='2025/05/29 15:00', ta='2025/05/29 17:00')
    ]
    ground_duties = [] # 暂时没有地面任务，以简化测试
    layover_stations = { # 假设北京和上海是可过夜机场
        'PEK': {},
        'SHA': {}
    }

    print("--- Demo Data Created ---")
    return flights, crews, bus_info, ground_duties, layover_stations, crew_leg_match_dict


def run_demo():
    """运行整个列生成框架，但使用小型的Demo数据"""
    
    # --- 1. 获取Demo数据 ---
    flights, crews, bus_info, ground_duties, layover_stations, crew_leg_match_dict = create_demo_data()

    # --- 2. 初始化主问题 ---
    # 【重要】将机组信息也传入主问题，以便创建机长唯一性约束
    master_problem = MasterProblem(flights=flights, crews=crews)
    
    # 在这个Demo中，我们不使用启发式初始解，而是用最简单的单航班方案，以更好地观察列生成的效果
    print("正在生成简单的初始解...")
    initial_rosters = []
    for flight in flights:
        for crew_id in crew_leg_match_dict:
            if flight.id in crew_leg_match_dict[crew_id]:
                cost = -1000 * (flight.flyTime / 60.0)
                initial_rosters.append(Roster(crew_id, [flight], cost))
                break # 分配给第一个能飞的机长即可
    
    print(f"成功生成 {len(initial_rosters)} 个初始排班方案。")
    print("将初始解添加至主问题...")
    for roster in initial_rosters:
        master_problem.add_roster_column(roster)

    # --- 3. 列生成循环 ---
    for i in range(10): # Demo中，我们迭代几轮就足够了
        print(f"\n--- 列生成迭代 {i+1} ---")
        
        print("求解主问题LP松弛...")
        duals = master_problem.solve_lp()
        if duals is None:
            print("主问题LP求解失败，终止。")
            break
        print(f"对偶价格获取成功: {duals}")
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        for crew in crews:
            crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]
            
            new_rosters = solve_subproblem_for_crew(
                crew, flights, bus_info, crew_specific_gds, 
                duals, layover_stations, crew_leg_match_dict
            )
            
            if new_rosters:
                print(f"  > 为机组 {crew.crewId} 找到了 {len(new_rosters)} 个新方案！")
                for r in new_rosters: master_problem.add_roster_column(r)
                new_rosters_found_count += len(new_rosters)
        
        print(f"本轮找到并添加了 {new_rosters_found_count} 个新排班方案。")
        if new_rosters_found_count == 0 and i > 0:
            print("\n未找到更多有价值的排班方案，列生成结束。")
            break

    # --- 4. 最终求解 ---
    print("\n列生成结束，正在求解最终的整数规划问题...")
    final_model = master_problem.solve_bip()
    if final_model.SolCount > 0:
        selected_rosters = master_problem.get_selected_rosters()
        print(f"\n成功找到解，包含 {len(selected_rosters)} 个排班方案。")
        for idx, roster in enumerate(selected_rosters):
            path_str = " -> ".join([t.id if hasattr(t, 'id') else 'REST' for t in roster.duties])
            print(f"  - 方案 {idx+1} (机长 {roster.crew_id}): {path_str}")
    else:
        print("\n未能找到可行的整数解。")


if __name__ == "__main__":
    run_demo()