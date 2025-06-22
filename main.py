# file: main.py

import time
from data_loader import load_all_data
from master_problem import MasterProblem
from subproblem_solver import solve_subproblem_for_crew
from results_writer import write_results_to_csv
from gurobipy import GRB
from data_models import Roster

# 从新文件中导入初始解生成器
from initial_solution_generator import generate_initial_rosters_with_heuristic

def main():
    # --- 1. 设置 ---
    start_time = time.time()
    TIME_LIMIT_SECONDS = 1 * 3600 + 55 * 60 
    data_path = 'data/'
    MAX_ITERATIONS = 500

    # --- 2. 数据加载与预处理 ---
    print("正在加载所有数据...")
    all_data = load_all_data(data_path)
    if not all_data:
        print("数据加载失败，程序退出。")
        return

    flights = all_data["flights"]
    crews = all_data["crews"]
    bus_info = all_data["bus_info"]
    ground_duties = all_data["ground_duties"]
    crew_leg_match_list = all_data["crew_leg_matches"]
    layover_stations = all_data["layover_stations"]
    
    print("正在预处理机长-航班资质数据...")
    crew_leg_match_dict = {}
    for match in crew_leg_match_list:
        flight_id, crew_id = match.flightId, match.crewId
        if crew_id not in crew_leg_match_dict:
            crew_leg_match_dict[crew_id] = []
        crew_leg_match_dict[crew_id].append(flight_id)
        
    # --- 3. 调用新的启发式函数生成初始解 ---
    master_problem = MasterProblem(flights=flights, crews=crews)
    initial_rosters = generate_initial_rosters_with_heuristic(
        flights, crews, bus_info, ground_duties, crew_leg_match_dict
    )
    
    if not initial_rosters:
        print("错误：启发式算法未能生成任何初始解。程序退出。")
        return
        
    print("将初始解添加至主问题...")
    for roster in initial_rosters:
        master_problem.add_roster_column(roster)
    
    # --- 4. 列生成循环 ---
    for i in range(MAX_ITERATIONS):
        iteration_start_time = time.time()
        print(f"\n--- 列生成迭代 {i+1} ---")
        
        elapsed_time = time.time() - start_time
        print(f"已用时: {elapsed_time / 60:.2f} 分钟。")                                                                                                           
        if elapsed_time > TIME_LIMIT_SECONDS:
            print("已达到时间限制，终止列生成，进入最终求解阶段。")
            break

        print("求解主问题LP松弛...")
        duals = master_problem.solve_lp()
        if duals is None:
            print("主问题LP求解失败，终止列生成。")
            break
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        for crew in crews:
            crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]        
            
            # --- 正确传递全部7个参数 ---
            new_rosters = solve_subproblem_for_crew(
                crew, flights, bus_info, crew_specific_gds, 
                duals, layover_stations, crew_leg_match_dict
            )

            if new_rosters:
                for r in new_rosters[:5]: 
                    master_problem.add_roster_column(r)
                new_rosters_found_count += len(new_rosters)
        
        print(f"本轮找到并添加了 {new_rosters_found_count} 个新排班方案。")
        
        if new_rosters_found_count == 0 and i > 0: 
            print("\n未找到更多有价值的排班方案，列生成结束。")
            break
        
        iteration_time = time.time() - iteration_start_time
        print(f"本轮迭代耗时: {iteration_time:.2f} 秒。")

    # --- 5. 求解最终整数规划问题 ---
    print("\n列生成结束，正在求解最终的整数规划问题...")
    final_model = master_problem.solve_bip()

    # 使用 SolCount > 0 来检查是否找到了解
    if final_model.SolCount > 0:
        selected_rosters = master_problem.get_selected_rosters()
        print(f"\n成功找到解，包含 {len(selected_rosters)} 个排班方案。")
        output_file = "output/rosterResult.csv"
        write_results_to_csv(selected_rosters, output_file)
        print(f"最终结果已写入文件: {output_file}")
    else:
        print("\n在时间限制内未能找到可行的整数解。")


if __name__ == "__main__":
    main()