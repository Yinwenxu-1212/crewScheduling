# file: main.py

import time
from datetime import datetime
from data_loader import load_all_data
from master_problem import MasterProblem
from results_writer import write_results_to_csv
from gurobipy import GRB
from data_models import Roster, Flight
from scoring_system import ScoringSystem

# 从新文件中导入初始解生成器
from initial_solution_generator import generate_initial_rosters_with_heuristic

# 在导入部分添加（如果文件存在的话）
try:
    from attention_guided_subproblem_solver import solve_subproblem_for_crew_with_attention
    ATTENTION_AVAILABLE = True
    print("Attention guidance successfully imported")
except ImportError as e:
    print(f"ImportError details: {e}")
    ATTENTION_AVAILABLE = False
except Exception as e:
    print(f"Other error during import: {e}")
    ATTENTION_AVAILABLE = False

def main():
    # --- 1. 设置 ---
    start_time = time.time()
    TIME_LIMIT_SECONDS = 1 * 3600 + 55 * 60 
    data_path = 'data/'
    MAX_ITERATIONS = 10

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
        flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
    )
    
    if not initial_rosters:
        print("错误：启发式算法未能生成任何初始解。程序退出。")
        return
        
    print("将初始解添加至主问题...")
    for roster in initial_rosters:
        master_problem.add_roster_column(roster)
    
    # --- 4. 列生成循环 ---
    # 在列生成循环外部初始化全局方案记录
    global_roster_signatures = set()
    
    def get_roster_signature(roster):
        # 只考虑任务ID，忽略顺序和时间的微小差异
        duty_ids = sorted([duty.id for duty in roster.duties])
        return f"{roster.crew_id}_{hash(tuple(duty_ids))}"
    
    # --- 4. 列生成循环 ---
    for i in range(MAX_ITERATIONS):
        iteration_start_time = time.time()
        print(f"\n--- 列生成迭代 {i+1} ---")
        
        # (时间检查逻辑保持不变)
        elapsed_time = time.time() - start_time
        print(f"已用时: {elapsed_time / 60:.2f} 分钟。")
        if elapsed_time > TIME_LIMIT_SECONDS:
            print("已达到时间限制，终止列生成，进入最终求解阶段。")
            break

        print("求解主问题LP松弛...")
        # --- 【修改】接收两种对偶价格 ---
        pi_duals, sigma_duals = master_problem.solve_lp()
        if pi_duals is None:
            print("主问题LP求解失败，终止列生成。")
            break
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        for crew in crews:
            crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]
            
            # --- 【修改】获取这位机长专属的sigma价格 ---
            crew_sigma_dual = sigma_duals.get(crew.crewId, 0.0)

            # --- 【修改】将sigma作为第8个参数传入子问题 ---
            new_rosters = solve_subproblem_for_crew_with_attention(
                crew, flights, bus_info, crew_specific_gds, 
                pi_duals, layover_stations, crew_leg_match_dict,
                crew_sigma_dual 
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

    # --- 5. 计算初始解质量 ---
    print("\n正在评估初始解质量...")
    
    # 使用新的评分系统计算详细得分
    scoring_system = ScoringSystem(flights, crews, layover_stations)
    score_details = scoring_system.calculate_total_score(initial_rosters)
    
    # 计算简单的成本统计（用于兼容性）
    initial_cost = sum(roster.cost for roster in initial_rosters)
    covered_flights = set()
    for roster in initial_rosters:
        for duty in roster.duties:
            if isinstance(duty, Flight):
                covered_flights.add(duty.id)
    
    uncovered_flights_count = len(flights) - len(covered_flights)
    initial_total_cost = initial_cost + uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY
    
    # 输出详细的评分信息
    print(f"=== 初始解详细评分 ===")
    print(f"总航班数: {len(flights)}")
    print(f"覆盖航班数: {len(covered_flights)}")
    print(f"未覆盖航班数: {score_details['uncovered_flights']}")
    print(f"值勤日日均飞时: {score_details['avg_daily_fly_time']:.2f} 小时")
    print(f"飞时得分: {score_details['fly_time_score']:.2f}")
    print(f"未覆盖航班惩罚: {score_details['uncovered_penalty']:.2f}")
    print(f"新增过夜站点惩罚: {score_details['new_layover_penalty']:.2f}")
    print(f"外站过夜惩罚: {score_details['away_overnight_penalty']:.2f}")
    print(f"置位惩罚: {score_details['positioning_penalty']:.2f}")
    print(f"违规惩罚: {score_details['violation_penalty']:.2f}")
    print(f"总得分: {score_details['total_score']:.2f}")
    print(f"=== 兼容性统计 ===")
    print(f"初始解排班成本: {initial_cost:.2f}")
    print(f"初始解总成本: {initial_total_cost:.2f}")
    
    # --- 6. 求解最终整数规划问题 ---
    print("\n列生成结束，正在求解最终的整数规划问题...")
    final_model = master_problem.solve_bip()

    # 使用 SolCount > 0 来检查是否找到了解
    final_solution_found = False
    if final_model.SolCount > 0:
        selected_rosters = master_problem.get_selected_rosters()
        if selected_rosters:  # 确保选择了方案
            final_cost = final_model.ObjVal
            print(f"\n最终解成本: {final_cost:.2f}, 包含 {len(selected_rosters)} 个排班方案。")
            
            # 比较解的质量
            if final_cost <= initial_total_cost:
                print(f"最终解优于初始解 (改善: {initial_total_cost - final_cost:.2f})")
                output_file = "output/rosterResult.csv"
                write_results_to_csv(selected_rosters, output_file)
                print(f"最终结果已写入文件: {output_file}")
                final_solution_found = True
            else:
                print(f"最终解劣于初始解 (恶化: {final_cost - initial_total_cost:.2f})，将使用初始解")
        else:
            print("\n最终解未选择任何排班方案")
    else:
        print("\n在时间限制内未能找到可行的整数解。")
    
    # --- 7. 回退到初始解 ---
    if not final_solution_found:
        print("\n使用初始解作为最终输出...")
        output_file = "output/rosterResult.csv"
        write_results_to_csv(initial_rosters, output_file)
        print(f"初始解已写入文件: {output_file}")
        print(f"初始解统计: 成本 {initial_cost:.2f}, 未覆盖航班 {uncovered_flights_count} 个")


if __name__ == '__main__':
    if not ATTENTION_AVAILABLE:
        print("Error: Attention module not available!")
        exit(1)
    
    # 直接运行完整的列生成算法，使用attention求解器
    main()