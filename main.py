# file: main.py

import time
import csv
import os
from datetime import datetime
from coverage_validator import CoverageValidator, print_coverage_summary
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
    MAX_ITERATIONS = 5  # 增加列生成迭代次数以提高解的质量
    
    # 设置日志文件
    debug_dir = "debug"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    log_file_path = os.path.join(debug_dir, f"roster_cost_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_file = open(log_file_path, 'w', encoding='utf-8')
    log_file.write(f"=== Roster成本调试日志 ===\n")
    log_file.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    log_file.flush()
    
    def log_debug(message: str):
        """写入调试信息到日志文件"""
        log_file.write(f"{message}\n")
        log_file.flush()

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
    no_improvement_rounds = 0  # 连续无改进轮数计数
    convergence_count = 0  # 目标函数改善微小的连续轮数
    
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
        log_debug(f"\n=== 列生成第 {i+1} 轮开始 ===\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 求解主问题LP松弛（不输出详细调试信息）
        pi_duals, sigma_duals, current_obj = master_problem.solve_lp(verbose=False, use_multiple_starts=True, num_starts=3)
        
        if pi_duals is None:
            print("主问题求解失败，退出列生成。")
            break
        
        # 跟踪目标函数变化
        if i > 0:
            obj_change = current_obj - previous_obj_val
            if obj_change > 1e-6:
                print(f"警告：目标函数增加了 {obj_change:.6f}，违反了列生成的单调性！")
            else:
                print(f"目标函数变化：{obj_change:.6f}")
        # 注意：previous_obj_val 将在后面的详细求解后更新
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        
        # 创建评分系统用于详细成本分析
        scoring_system = ScoringSystem(flights, crews, layover_stations)
        
        for crew in crews:
            crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]
            crew_sigma_dual = sigma_duals.get(crew.crewId, 0.0)
            
            # 获取当前的lambda值
            current_lambda = master_problem.get_current_lambda()
            new_rosters = solve_subproblem_for_crew_with_attention(
                crew, flights, bus_info, crew_specific_gds, 
                pi_duals, layover_stations, crew_leg_match_dict,
                crew_sigma_dual, iteration_round=i, external_log_func=log_debug, lambda_k=current_lambda
            )
            
            if new_rosters:
                valuable_count = 0
                # print(f"\n=== 机组 {crew.crewId} 的Roster详细分析 ===")
                for idx, r in enumerate(new_rosters):
                    # 获取当前的lambda值
                    current_lambda = master_problem.get_current_lambda()
                    # 调试：检查对偶价格数值
                    if idx == 0:  # 只对第一个roster进行调试
                        flight_dual_sum = sum(pi_duals.get(duty.id, 0.0) for duty in r.duties if isinstance(duty, Flight))
                        log_debug(f"调试 - 机组 {crew.crewId} Roster {idx+1}: 航班对偶价格总和={flight_dual_sum:.6f}")
                        
                        # 输出前3个航班的对偶价格
                        flight_duties = [duty for duty in r.duties if isinstance(duty, Flight)]
                        for i, flight in enumerate(flight_duties[:3]):
                            dual_val = pi_duals.get(flight.id, 0.0)
                            log_debug(f"  航班 {flight.id}: 对偶价格={dual_val:.6f}")
                    
                    # 获取详细的成本分解
                    cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                        r, crew, pi_duals, crew_sigma_dual, current_lambda
                    )
                    
                    reduced_cost = cost_details['reduced_cost']
                    status = "[有价值]" if reduced_cost < -1e-6 else "[无价值]"
                    
                    if reduced_cost < -1e-6:
                        valuable_count += 1
                        # 简化日志输出，只记录关键信息
                        # 只记录前3个有价值的roster的详细信息，其余只记录摘要
                        if valuable_count <= 3:
                            log_debug(f"\n机组 {crew.crewId} - Roster {idx+1} {status}:")
                            log_debug(f"  Reduced Cost: {reduced_cost:.6f}, 航班数: {cost_details['flight_count']}, 飞行时间: {cost_details['total_flight_hours']:.2f}h")
                        elif valuable_count <= 10:
                            log_debug(f"机组 {crew.crewId} - Roster {idx+1}: RC={reduced_cost:.3f}, 航班={cost_details['flight_count']}, 时间={cost_details['total_flight_hours']:.1f}h")
                        # 超过10个有价值roster后不再记录详细信息
                        
                        master_problem.add_roster_column(r)
                        new_rosters_found_count += 1
                        
                # print(f"\n  机组 {crew.crewId}: 共有 {valuable_count} 个有价值的roster被添加到主问题")
            else:
                # print(f"  机组 {crew.crewId}: 未找到任何roster")
                pass
                    
        # 显示每轮列生成后的最优解变化
        print(f"\n=== 第 {i+1} 轮列生成结果 ===")
        print(f"本轮新增有价值roster数量: {new_rosters_found_count}")
        
        # 求解当前主问题获取最优解（输出详细调试信息）
        pi_duals, sigma_duals, current_obj_val = master_problem.solve_lp(verbose=True, use_multiple_starts=True, num_starts=3)
        if current_obj_val is not None:  # 求解成功
            print(f"当前主问题最优目标函数值: {current_obj_val:.6f}")
            
            # 如果不是第一轮，显示目标函数值的变化
            if i > 0:
                improvement = previous_obj_val - current_obj_val
                print(f"相比上轮的改善: {improvement:.6f}")
                
                # 基于目标函数改善判断收敛
                if improvement < 1e-4:  # 改善小于阈值
                    convergence_count += 1
                    print(f"目标函数改善微小，连续{convergence_count}轮")
                else:
                    convergence_count = 0
            
            previous_obj_val = current_obj_val
        else:
            print("当前主问题求解失败")
        
        # 基本收敛条件
        if new_rosters_found_count == 0:
            no_improvement_rounds += 1
            print(f"本轮未找到有价值roster，连续{no_improvement_rounds}轮无改进")
        else:
            no_improvement_rounds = 0
        
        # 简单收敛判断
        if no_improvement_rounds >= 3 and i > 0:
            print(f"\n连续3轮未找到有价值的排班方案，列生成结束。")
            break
        elif convergence_count >= 3 and i > 1:
            print(f"\n目标函数连续3轮改善微小，列生成收敛。")
            break
        elif i >= MAX_ITERATIONS - 1:
            print("\n达到最大迭代次数，列生成结束。")
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
    print(f"=== 初始解成本分析 ===")
    print(f"总航班数: {len(flights)}")
    print(f"覆盖航班数: {len(covered_flights)}")
    print(f"未覆盖航班数: {uncovered_flights_count}")
    print(f"排班方案数: {len(initial_rosters)}")
    print(f"排班成本总和: {initial_cost:.2f}")
    print(f"未覆盖航班惩罚: {uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY:.2f}")
    print(f"初始解总成本: {initial_total_cost:.2f}")
    
    # 验证初始解航班覆盖率
    print(f"\n=== 初始解航班覆盖率验证 ===")
    validator = CoverageValidator(min_coverage_rate=0.8)
    initial_coverage_result = validator.validate_coverage(flights, initial_rosters)
    print(validator.get_coverage_report(initial_coverage_result))
    
    if not initial_coverage_result['is_valid']:
        print("\n⚠️  警告：初始解不满足80%航班覆盖率要求！")
        print("程序将继续运行，但最终结果可能不符合竞赛要求。")
        suggestions = validator.suggest_improvements(initial_coverage_result)
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("\n✅ 初始解满足航班覆盖率要求")
    
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
            
            # 验证航班覆盖率
            print("\n=== 最终解航班覆盖率验证 ===")
            validator = CoverageValidator(min_coverage_rate=0.8)
            coverage_result = validator.validate_coverage(flights, selected_rosters)
            print(validator.get_coverage_report(coverage_result))
            
            if not coverage_result['is_valid']:
                print("\n⚠️  警告：最终解不满足80%航班覆盖率要求！")
                print("根据竞赛规则，此解决方案可能被判定为无效。")
                suggestions = validator.suggest_improvements(coverage_result)
                for suggestion in suggestions:
                    print(suggestion)
            
            # 比较解的质量
            if final_cost <= initial_total_cost and coverage_result['is_valid']:
                print(f"\n✅ 最终解优于初始解且满足覆盖率要求 (改善: {initial_total_cost - final_cost:.2f})")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"output/rosterResult_{timestamp}.csv"
                write_results_to_csv(selected_rosters, output_file)
                print(f"最终结果已写入文件: {output_file}")
                final_solution_found = True
            elif coverage_result['is_valid']:
                print(f"\n⚠️  最终解满足覆盖率但劣于初始解 (恶化: {final_cost - initial_total_cost:.2f})")
                print("将检查初始解的覆盖率后决定使用哪个解")
            else:
                print(f"\n❌ 最终解不满足覆盖率要求，将使用初始解")
        else:
            print("\n最终解未选择任何排班方案")
    else:
        print("\n在时间限制内未能找到可行的整数解。")
    
    # --- 7. 回退到初始解 ---
    if not final_solution_found:
        print("\n使用初始解作为最终输出...")
        
        # 初始解的覆盖率验证已在前面完成，这里直接使用结果
        if not initial_coverage_result['is_valid']:
            print("\n❌ 警告：初始解不满足80%航班覆盖率要求！")
            print("根据竞赛规则，此解决方案可能被判定为无效。")
        else:
            print("\n✅ 使用满足覆盖率要求的初始解作为最终输出")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"output/rosterResult_{timestamp}.csv"
        write_results_to_csv(initial_rosters, output_file)
        print(f"初始解已写入文件: {output_file}")
        print(f"初始解统计: 成本 {initial_cost:.2f}, 未覆盖航班 {uncovered_flights_count} 个")
        print(f"覆盖率: {initial_coverage_result['coverage_rate']:.1%}")
    
    # 关闭日志文件
    log_debug(f"\n=== 程序结束 ===\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()
    print(f"\n调试日志已保存到: {log_file_path}")


if __name__ == '__main__':
    if not ATTENTION_AVAILABLE:
        print("Error: Attention module not available!")
        exit(1)
    main()