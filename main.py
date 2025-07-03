# file: main.py
# 值勤日平均飞行时间奖励的权重100
# 未覆盖航班惩罚（从5增加到300）

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
from ground_duty_validator import GroundDutyValidator

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
    MAX_ITERATIONS = 3  # 大幅增加列生成迭代次数以提高覆盖率
    
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
    master_problem = MasterProblem(flights=flights, crews=crews, ground_duties=ground_duties)
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
        
        # 注意：目标函数变化将在详细求解后计算，确保使用一致的目标函数值
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        
        # 创建评分系统用于详细成本分析
        scoring_system = ScoringSystem(flights, crews, layover_stations)
        
        # 添加总体调试信息
        current_lambda = master_problem.get_current_lambda()
        log_debug(f"\n=== 第 {i+1} 轮列生成调试信息 ===")
        log_debug(f"当前 Lambda 值: {current_lambda:.6f}")
        
        # 分析对偶价格分布
        dual_values = list(pi_duals.values())
        if dual_values:
            log_debug(f"对偶价格统计: 最小={min(dual_values):.3f}, 最大={max(dual_values):.3f}, 平均={sum(dual_values)/len(dual_values):.3f}")
            positive_duals = [d for d in dual_values if d > 0]
            log_debug(f"正对偶价格数量: {len(positive_duals)}/{len(dual_values)}")
        
        # 分析机组对偶价格
        sigma_values = list(sigma_duals.values())
        if sigma_values:
            log_debug(f"机组对偶价格统计: 最小={min(sigma_values):.3f}, 最大={max(sigma_values):.3f}, 平均={sum(sigma_values)/len(sigma_values):.3f}")
        
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
                for idx, r in enumerate(new_rosters):
                    # 获取当前的lambda值
                    current_lambda = master_problem.get_current_lambda()
                    
                    # 获取详细的成本分解
                    cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                        r, crew, pi_duals, crew_sigma_dual, current_lambda
                    )
                    
                    reduced_cost = cost_details['reduced_cost']
                    
                    if reduced_cost < -1e-4:
                        valuable_count += 1
                        master_problem.add_roster_column(r)
                        new_rosters_found_count += 1
                
                # 只记录关键统计信息
                if valuable_count > 0:
                    log_debug(f"机组 {crew.crewId}: 生成 {len(new_rosters)} 个roster，其中 {valuable_count} 个有价值")
                        
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
            
            # 跟踪目标函数变化（使用一致的目标函数值）
            if i > 0 and previous_obj_val != float('inf'):
                obj_change = current_obj_val - previous_obj_val
                if obj_change > 1e-6:
                    print(f"警告：目标函数增加了 {obj_change:.6f}，不满足列生成的单调性！")
                else:
                    print(f"目标函数变化：{obj_change:.6f}")
            
            # 如果不是第一轮，显示目标函数值的变化
            if i > 0:
                improvement = current_obj_val - previous_obj_val
                print(f"相比上轮的改善: {improvement:.6f}")
                
                # 基于目标函数改善判断收敛
                if improvement < 1e-6:  # 降低收敛阈值，允许更多迭代
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
    
    # 使用与Dinkelbach算法一致的目标函数值计算方式
    # 计算初始解的原始问题目标函数值（日均飞时得分）
    total_flight_hours = 0.0
    total_duty_days = 0.0
    total_penalties = 0.0
    covered_flights = set()
    
    for roster in initial_rosters:
        if hasattr(roster, 'metrics'):
            metrics = roster.metrics
            total_flight_hours += metrics['total_flight_hours']
            total_duty_days += metrics['duty_days']
            total_penalties += (0.5 * metrics['positioning_count'] +
                               0.5 * metrics['away_overnight_days'] +
                               10 * metrics['new_layover_stations'])
        
        for duty in roster.duties:
            if isinstance(duty, Flight):
                covered_flights.add(duty.id)
    
    uncovered_flights_count = len(flights) - len(covered_flights)
    total_penalties += uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY
    
    # 计算初始解的原始问题目标函数值（日均飞时得分）
    if total_duty_days > 0:
        initial_objective_value = (100 * total_flight_hours - total_penalties) / total_duty_days
    else:
        initial_objective_value = 0.0
    
    # 输出初始解的目标函数值信息
    print(f"=== 初始解目标函数分析 ===")
    print(f"总航班数: {len(flights)}")
    print(f"覆盖航班数: {len(covered_flights)}")
    print(f"未覆盖航班数: {uncovered_flights_count}")
    print(f"航班覆盖率: {len(covered_flights)/len(flights)*100:.1f}%")
    print(f"排班方案数: {len(initial_rosters)}")
    print(f"总飞行时间: {total_flight_hours:.2f} 小时")
    print(f"总值勤天数: {total_duty_days:.0f} 天")
    print(f"日均飞行时间: {total_flight_hours/total_duty_days if total_duty_days > 0 else 0:.2f} 小时")
    print(f"总惩罚项: {total_penalties:.2f}")
    print(f"初始解目标函数值（日均飞时得分）: {initial_objective_value:.2f}")
    
    # 验证初始解航班覆盖率
    print(f"\n=== 初始解航班覆盖率验证 ===")
    validator = CoverageValidator(min_coverage_rate=0.8)
    initial_coverage_result = validator.validate_coverage(flights, initial_rosters)
    print(validator.get_coverage_report(initial_coverage_result))
    
    # 验证初始解占位任务规则
    print(f"\n=== 初始解占位任务规则验证 ===")
    ground_duty_validator = GroundDutyValidator(ground_duties)
    initial_ground_duty_result = ground_duty_validator.validate_solution(initial_rosters)
    print(ground_duty_validator.get_validation_report(initial_ground_duty_result))
    
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
        print(f"负成本原因: 飞时奖励(100*飞行时间) > 各种惩罚")
    print(f"总Roster成本: {sum(roster_costs):.2f}")
    
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
            
            # 验证占位任务规则
            print("\n=== 占位任务规则验证 ===")
            ground_duty_validator = GroundDutyValidator(ground_duties, crews)
            ground_duty_result = ground_duty_validator.validate_solution(selected_rosters)
            print(ground_duty_validator.get_validation_report(ground_duty_result))
            
            if not coverage_result['is_valid']:
                print("\n⚠️  警告：最终解不满足80%航班覆盖率要求！")
                print("根据竞赛规则，此解决方案可能被判定为无效。")
                suggestions = validator.suggest_improvements(coverage_result)
                for suggestion in suggestions:
                    print(suggestion)
            
            # 计算最终解的目标函数值（使用与初始解一致的计算方式）
            final_summary = master_problem.get_solution_summary()
            final_objective_value = final_summary.get('final_score', 0.0)
            
            print(f"\n=== 最终解目标函数分析 ===")
            print(f"最终解目标函数值（日均飞时得分）: {final_objective_value:.2f}")
            print(f"初始解目标函数值（日均飞时得分）: {initial_objective_value:.2f}")
            
            # 比较解的质量（注意：这是最大化问题，目标函数值越大越好）
            # 修改逻辑：即使占位任务没有全部覆盖也可以输出最终解，只要航班覆盖率满足要求
            final_solution_valid = coverage_result['is_valid']  # 只要求航班覆盖率满足要求
            
            if final_objective_value >= initial_objective_value and final_solution_valid:
                print(f"\n✅ 最终解优于初始解且满足航班覆盖率要求 (改善: {final_objective_value - initial_objective_value:.2f})")
                if not ground_duty_result['is_valid']:
                    print("⚠️  注意：最终解未完全覆盖所有占位任务，但仍可输出")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"output/rosterResult_{timestamp}.csv"
                write_results_to_csv(selected_rosters, output_file)
                print(f"最终结果已写入文件: {output_file}")
                final_solution_found = True
            elif final_solution_valid:
                print(f"\n⚠️  最终解满足航班覆盖率要求但劣于初始解 (恶化: {initial_objective_value - final_objective_value:.2f})")
                if not ground_duty_result['is_valid']:
                    print("⚠️  注意：最终解未完全覆盖所有占位任务，但仍可输出")
                print("将检查初始解的约束满足情况后决定使用哪个解")
            else:
                print(f"\n❌ 最终解不满足航班覆盖率要求，将使用初始解")
                if not ground_duty_result['is_valid']:
                    print("⚠️  注意：最终解也未完全覆盖所有占位任务")
        else:
            print("\n最终解未选择任何排班方案")
    else:
        print("\n在时间限制内未能找到可行的整数解。")
    
    # --- 7. 回退到初始解 ---
    if not final_solution_found:
        print("\n使用初始解作为最终输出...")
        
        # 初始解的验证已在前面完成，这里直接使用结果
        # 修改逻辑：即使占位任务没有全部覆盖也可以输出初始解，只要航班覆盖率满足要求
        initial_solution_valid = initial_coverage_result['is_valid']  # 只要求航班覆盖率满足要求
        if not initial_solution_valid:
            print(f"\n❌ 警告：初始解不满足航班覆盖率要求！")
            print("根据竞赛规则，此解决方案可能被判定为无效。")
        else:
            print("\n✅ 使用满足航班覆盖率要求的初始解作为最终输出")
            if not initial_ground_duty_result['is_valid']:
                print("⚠️  注意：初始解未完全覆盖所有占位任务，但仍可输出")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"output/rosterResult_{timestamp}.csv"
        write_results_to_csv(initial_rosters, output_file)
        print(f"初始解已写入文件: {output_file}")
        print(f"初始解统计: 目标函数值（日均飞时得分） {initial_objective_value:.2f}, 未覆盖航班 {uncovered_flights_count} 个")
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