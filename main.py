# file: main.py

import time
import csv
from datetime import datetime
from data_loader import load_all_data
from master_problem import MasterProblem
from results_writer import write_results_to_csv
from gurobipy import GRB
from data_models import Roster, Flight
from scoring_system import ScoringSystem
from initial_solution_generator import generate_initial_rosters_with_heuristic

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
    MAX_ITERATIONS = 5

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
    print("\n开始列生成过程...")
    previous_obj_val = float('inf')  # 初始化上一轮目标函数值
    
    # 在列生成循环外部初始化全局方案记录
    global_roster_signatures = set()
    
    def get_roster_signature(roster):
        # 只考虑任务ID，忽略顺序和时间的微小差异
        duty_ids = sorted([duty.id for duty in roster.duties])
        return f"{roster.crew_id}_{hash(tuple(duty_ids))}"
    
    # 修改列生成循环
    for i in range(MAX_ITERATIONS):  # 改为大写的MAX_ITERATIONS
        iteration_start_time = time.time()
        print(f"\n=== 列生成第 {i+1} 轮 ===")
        
        # 求解主问题LP松弛
        pi_duals, sigma_duals, current_obj = master_problem.solve_lp()
        
        if pi_duals is None:
            print("主问题求解失败，退出列生成。")
            break
        
        # 跟踪目标函数变化
        if i > 0:
            obj_change = current_obj - previous_obj
            if obj_change > 1e-6:
                print(f"警告：目标函数增加了 {obj_change:.6f}，违反了列生成的单调性！")
            else:
                print(f"目标函数变化：{obj_change:.6f}")
        previous_obj = current_obj
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        
        # 创建评分系统用于详细成本分析
        scoring_system = ScoringSystem(flights, crews, layover_stations)
        
        for crew in crews:
            crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]
            crew_sigma_dual = sigma_duals.get(crew.crewId, 0.0)
            
            new_rosters = solve_subproblem_for_crew_with_attention(
                crew, flights, bus_info, crew_specific_gds, 
                pi_duals, layover_stations, crew_leg_match_dict,
                crew_sigma_dual 
            )
            
            if new_rosters:
                valuable_count = 0
                # print(f"\n=== 机组 {crew.crewId} 的Roster详细分析 ===")
                for idx, r in enumerate(new_rosters):
                    # 获取详细的成本分解
                    cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                        r, crew, pi_duals, crew_sigma_dual
                    )
                    
                    reduced_cost = cost_details['reduced_cost']
                    status = "[有价值]" if reduced_cost < -1e-6 else "[无价值]"
                    
                    # print(f"\n  Roster {idx+1} {status}:")
                    # print(f"    总成本 (total_cost): {cost_details['total_cost']:.6f}")
                    # print(f"    Reduced Cost: {reduced_cost:.6f}")
                    # print(f"    成本分解:")
                    # print(f"      - 飞行奖励 (flight_reward): {cost_details['flight_reward']:.6f}")
                    # print(f"      - 对偶价格收益 (dual_price_total): {cost_details['dual_price_total']:.6f}")
                    # print(f"      - 置位惩罚 (positioning_penalty): {cost_details['positioning_penalty']:.6f}")
                    # print(f"      - 外站过夜惩罚 (overnight_penalty): {cost_details['overnight_penalty']:.6f}")
                    # print(f"      - 其他成本 (other_costs): {cost_details['other_costs']:.6f}")
                    # print(f"      - 机组对偶系数 (crew_sigma_dual): {cost_details['crew_sigma_dual']:.6f}")
                    # print(f"    统计信息:")
                    # print(f"      - 航班数量: {cost_details['flight_count']}")
                    # print(f"      - 总飞行时间: {cost_details['total_flight_hours']:.2f}小时")
                    # print(f"      - 值勤天数: {cost_details['duty_days']}")
                    # print(f"      - 日均飞行时间: {cost_details['avg_daily_flight_hours']:.2f}小时")
                    # print(f"      - 置位次数: {cost_details['positioning_count']}")
                    # print(f"      - 外站过夜次数: {cost_details['overnight_count']}")
                    
                    if reduced_cost < -1e-6:
                        master_problem.add_roster_column(r)
                        new_rosters_found_count += 1
                        valuable_count += 1
                        
                # print(f"\n  机组 {crew.crewId}: 共有 {valuable_count} 个有价值的roster被添加到主问题")
            else:
                # print(f"  机组 {crew.crewId}: 未找到任何roster")
                pass
                    
        # 显示每轮列生成后的最优解变化
        print(f"\n=== 第 {i+1} 轮列生成结果 ===")
        print(f"本轮新增有价值roster数量: {new_rosters_found_count}")
        
        # 求解当前主问题获取最优解
        pi_duals, sigma_duals, current_obj_val = master_problem.solve_lp()
        if current_obj_val is not None:  # 求解成功
            print(f"当前主问题最优目标函数值: {current_obj_val:.6f}")
            
            # 如果不是第一轮，显示目标函数值的变化
            if i > 0:
                improvement = previous_obj_val - current_obj_val
                print(f"相比上轮的改善: {improvement:.6f}")
                if improvement < 1e-6:
                    print("目标函数值改善微小，可能接近最优解")
            
            previous_obj_val = current_obj_val
        else:
            print("当前主问题求解失败")
        
        if new_rosters_found_count == 0 and i > 0:
            print("\n未找到更多有价值的排班方案，列生成结束。")
            break

    # --- 5. 计算初始解质量 ---
    print("\n正在评估初始解质量...")
    
    # 使用与列生成一致的成本计算方式
    # 计算每个roster的成本（与列生成过程中master_problem使用的方式一致）
    initial_cost = sum(roster.cost for roster in initial_rosters)
    covered_flights = set()
    for roster in initial_rosters:
        for duty in roster.duties:
            if isinstance(duty, Flight):
                covered_flights.add(duty.id)
    
    uncovered_flights_count = len(flights) - len(covered_flights)
    initial_total_cost = initial_cost + uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY
    
    # 输出与列生成一致的成本信息
    print(f"=== 初始解成本分析（与列生成一致） ===")
    print(f"总航班数: {len(flights)}")
    print(f"覆盖航班数: {len(covered_flights)}")
    print(f"未覆盖航班数: {uncovered_flights_count}")
    print(f"排班方案数: {len(initial_rosters)}")
    print(f"排班成本总和: {initial_cost:.2f}")
    print(f"未覆盖航班惩罚: {uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY:.2f}")
    print(f"初始解总成本: {initial_total_cost:.2f}")
    
    # 调试：分析roster成本的分布
    print(f"\n=== Roster成本调试信息 ===")
    roster_costs = [roster.cost for roster in initial_rosters]
    print(f"Roster成本范围: [{min(roster_costs):.2f}, {max(roster_costs):.2f}]")
    print(f"平均Roster成本: {sum(roster_costs)/len(roster_costs):.2f}")
    positive_costs = [c for c in roster_costs if c > 0]
    negative_costs = [c for c in roster_costs if c < 0]
    print(f"正成本Roster数量: {len(positive_costs)}")
    print(f"负成本Roster数量: {len(negative_costs)}")
    if negative_costs:
        print(f"负成本原因: 飞时奖励({-sum(roster_costs)/1000:.2f}小时*1000) > 各种惩罚")
    
    # 可选：显示详细的评分系统结果（仅供参考）
    scoring_system = ScoringSystem(flights, crews, layover_stations)
    score_details = scoring_system.calculate_total_score(initial_rosters)
    print(f"\n=== 详细评分系统结果（仅供参考） ===")
    print(f"值勤日日均飞时: {score_details['avg_daily_fly_time']:.2f} 小时")
    print(f"飞时得分: {score_details['fly_time_score']:.2f}")
    print(f"总得分: {score_details['total_score']:.2f}")
    
    # --- 6. 求解最终整数规划问题 ---
    print("\n列生成结束，正在求解最终的整数规划问题...")
    final_model = master_problem.solve_bip()

    # 调试：显示目标函数值的详细组成
    print(f"\n=== 目标函数调试信息 ===")
    if final_model.SolCount > 0:
        obj_val = final_model.ObjVal
        print(f"最终目标函数值: {obj_val:.2f}")
        
        # 分解目标函数
        roster_cost_sum = 0
        uncovered_penalty_sum = 0
        
        for roster, var in master_problem.roster_vars.items():
            if var.X > 0.5:  # 被选中的roster
                roster_cost_sum += roster.cost * var.X
                
        for flight_id, var in master_problem.uncovered_vars.items():
            if var.X > 0.5:  # 未覆盖的航班
                uncovered_penalty_sum += master_problem.UNCOVERED_FLIGHT_PENALTY * var.X
        
        print(f"目标函数组成:")
        print(f"  - 选中Roster成本总和: {roster_cost_sum:.2f}")
        print(f"  - 未覆盖航班惩罚: {uncovered_penalty_sum:.2f}")
        print(f"  - 总计: {roster_cost_sum + uncovered_penalty_sum:.2f}")
        print(f"  - 验证: 与目标函数值差异 = {abs(obj_val - (roster_cost_sum + uncovered_penalty_sum)):.6f}")
    else:
        print("未找到可行解")

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
    main()