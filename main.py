#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机组排班优化系统主程序
Crew Scheduling Optimization System Main Module

基于列生成算法和注意力机制的机组排班优化解决方案。
该系统使用线性目标函数进行优化，支持复杂的航空业务约束。

Author: Crew Scheduling Team
Email: 2151102@tongji.edu.cn
Version: 2.0.0
Date: 2025-01-09
"""

import csv
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 项目模块导入
from constraint_checker import UnifiedConstraintChecker
from coverage_validator import CoverageValidator
from data_loader import load_all_data
from data_models import Flight, Roster
from ground_duty_validator import GroundDutyValidator
from initial_solution_generator import generate_initial_rosters_with_heuristic
from master_problem import MasterProblem
from results_writer import write_results_to_csv
from scoring_system import ScoringSystem
from unified_config import UnifiedConfig

# 第三方库导入
try:
    from gurobipy import GRB
except ImportError:
    print("警告: Gurobi未安装，某些功能可能不可用")
    GRB = None

# 可选模块导入
OPTIMIZATION_AVAILABLE = False
ATTENTION_AVAILABLE = False

try:
    from attention_guided_subproblem_solver import solve_subproblem_for_crew_with_attention
    ATTENTION_AVAILABLE = True
    print("✅ 注意力引导模块加载成功")
except ImportError as e:
    print(f"⚠️  注意力引导模块导入失败: {e}")
    ATTENTION_AVAILABLE = False
except Exception as e:
    print(f"❌ 注意力引导模块加载错误: {e}")
    ATTENTION_AVAILABLE = False

def main() -> None:
    """
    机组排班优化系统主函数
    
    执行完整的机组排班优化流程：
    1. 数据加载与预处理
    2. 初始解生成
    3. 列生成算法优化
    4. 整数规划求解
    5. 结果验证与输出
    
    Returns:
        None
        
    Raises:
        SystemExit: 当关键模块不可用或数据加载失败时
    """
    # --- 0. 算法版本 ---
    print("=== 机组排班优化系统 ===")
    print("使用简化线性目标函数版本")
    print("目标函数: 覆盖率奖励 + 飞行时间奖励 - 各种惩罚项")
    
    use_simple_objective = True
    
    # --- 1. 设置 ---
    start_time = time.time()
    TIME_LIMIT_SECONDS = UnifiedConfig.TIME_LIMIT_SECONDS
    data_path = UnifiedConfig.DATA_PATH
    MAX_ITERATIONS = UnifiedConfig.MAX_COLUMN_GENERATION_ITERATIONS
    
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
    UnifiedConfig.GDS = ground_duties # 便于类传参
    
    print("正在预处理机长-航班资质数据...")
    crew_leg_match_dict = {}
    for match in crew_leg_match_list:
        flight_id, crew_id = match.flightId, match.crewId
        if crew_id not in crew_leg_match_dict:
            crew_leg_match_dict[crew_id] = []
        crew_leg_match_dict[crew_id].append(flight_id)
        
    # --- 3. 初始化主问题求解器 ---
    master_problem = MasterProblem(flights, crews, ground_duties, layover_stations)
    
    print("\n=== 线性目标函数参数 ===")
    print(f"飞行时间奖励: {master_problem.FLIGHT_TIME_REWARD}")
    print(f"未覆盖航班惩罚: {master_problem.UNCOVERED_FLIGHT_PENALTY}")
    print(f"占位任务惩罚: {master_problem.POSITIONING_PENALTY}")
    print(f"过夜惩罚: {master_problem.AWAY_OVERNIGHT_PENALTY}")
    print(f"新过站惩罚: {master_problem.NEW_LAYOVER_PENALTY}")
    
    # --- 4. 调用新的启发式函数生成初始解 ---
    initial_rosters = generate_initial_rosters_with_heuristic(
        flights, crews, bus_info, ground_duties, crew_leg_match_dict, layover_stations
    )
    
    if not initial_rosters:
        print("错误：启发式算法未能生成任何初始解。程序退出。")
        return
        
    print("将初始解添加至主问题...")
    for roster in initial_rosters:
        master_problem.add_roster(roster, is_initial_roster=True)  # 标记为初始解，设置保护
    
    # --- 4. 列生成循环 ---
    print("\n开始列生成过程...")
    previous_obj_val = float('inf')  # 最小化问题：初始值为正无穷
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
        pi_duals, sigma_duals, ground_duty_duals, current_obj = master_problem.solve_lp(verbose=False)
        
        if pi_duals is None:
            print("主问题求解失败，退出列生成。")
            break
        
        # 简要显示列生成轮次信息
        print(f"第{i+1}轮列生成: 目标函数值={current_obj:.2f}")
        
        # 详细分析对偶价格（记录到日志文件）
        log_debug(f"第{i+1}轮列生成详细信息:")
        log_debug(f"  当前目标函数值: {current_obj:.2f}")
        log_debug(f"  航班对偶价格数量: {len(pi_duals)}")
        log_debug(f"  机组对偶价格数量: {len(sigma_duals)}")
        
        # 分析对偶价格分布
        flight_dual_values = list(pi_duals.values())
        crew_dual_values = list(sigma_duals.values())
        
        if flight_dual_values:
            log_debug(f"  航班对偶价格: min={min(flight_dual_values):.6f}, max={max(flight_dual_values):.6f}, avg={sum(flight_dual_values)/len(flight_dual_values):.6f}")
            positive_flight_duals = [d for d in flight_dual_values if d > 1e-6]
            log_debug(f"  正航班对偶价格数量: {len(positive_flight_duals)}/{len(flight_dual_values)}")
        
        if crew_dual_values:
            log_debug(f"  机组对偶价格: min={min(crew_dual_values):.6f}, max={max(crew_dual_values):.6f}, avg={sum(crew_dual_values)/len(crew_dual_values):.6f}")
        
        print("为所有机组人员求解子问题...")
        new_rosters_found_count = 0
        
        # 初始化进度条
        import sys
        def print_progress_bar(current, total, bar_length=50):
            progress = current / total
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            percent = progress * 100
            sys.stdout.write(f'\r  进度: [{bar}] {percent:.1f}% ({current}/{total})')
            sys.stdout.flush()
        
        # 显示初始进度条
        print_progress_bar(0, len(crews))
        
        # 创建评分系统用于详细成本分析
        scoring_system = ScoringSystem(flights, crews, layover_stations)
        
        # 添加总体调试信息
        log_debug(f"\n=== 第 {i+1} 轮列生成调试信息 ===")
        log_debug(f"使用最大化线性目标函数")
        
        # 分析对偶价格分布
        dual_values = list(pi_duals.values())
        if dual_values:
            log_debug(f"航班对偶价格统计: 最小={min(dual_values):.3f}, 最大={max(dual_values):.3f}, 平均={sum(dual_values)/len(dual_values):.3f}")
            positive_duals = [d for d in dual_values if d > 0]
            log_debug(f"正对偶价格数量: {len(positive_duals)}/{len(dual_values)}")
        
        # 分析机组对偶价格
        sigma_values = list(sigma_duals.values())
        if sigma_values:
            log_debug(f"机组对偶价格统计: 最小={min(sigma_values):.3f}, 最大={max(sigma_values):.3f}, 平均={sum(sigma_values)/len(sigma_values):.3f}")
        
        crew_processed = 0
        
        for crew in crews:
            crew_processed += 1
            crew_specific_gds = [gd for gd in ground_duties if gd.crewId == crew.crewId]
            crew_sigma_dual = sigma_duals.get(crew.crewId, 0.0)
            
            # 检查该机组是否有可执行的航班
            eligible_flights = crew_leg_match_dict.get(crew.crewId, [])
            # 将详细信息记录到日志文件
            log_debug(f"  机组{crew.crewId}: 可执行航班{len(eligible_flights)}个, 对偶价格{crew_sigma_dual:.6f}")
            
            if not eligible_flights:
                log_debug(f"    跳过：无可执行航班")
                continue
            
            # 计算该机组可执行航班的对偶价格统计
            crew_flight_duals = [pi_duals.get(fid, 0.0) for fid in eligible_flights]
            if crew_flight_duals:
                log_debug(f"    可执行航班对偶价格: min={min(crew_flight_duals):.6f}, max={max(crew_flight_duals):.6f}")
                positive_crew_flight_duals = [d for d in crew_flight_duals if d > 1e-6]
                log_debug(f"    正对偶价格航班: {len(positive_crew_flight_duals)}/{len(crew_flight_duals)}")
            
            # 调用子问题求解 - 确保参数顺序正确
            try:
                new_rosters = solve_subproblem_for_crew_with_attention(
                    crew, flights, bus_info, crew_specific_gds,
                    pi_duals, layover_stations, crew_leg_match_dict,
                    crew_sigma_dual, ground_duty_duals=ground_duty_duals, iteration_round=i, external_log_func=log_debug
                )
                
                if new_rosters:
                    valuable_count = 0
                    for idx, r in enumerate(new_rosters):
                        # 获取详细的成本分解
                        cost_details = scoring_system.calculate_roster_cost_with_dual_prices(
                            r, crew, pi_duals, crew_sigma_dual
                        )
                        
                        reduced_cost = cost_details['reduced_cost']
                        
                        # 详细记录每个roster的信息到日志文件
                        log_debug(f"    机组{crew.crewId} Roster#{idx+1}: reduced_cost={reduced_cost:.6f}")
                        log_debug(f"      总成本: {cost_details['total_cost']:.6f}")
                        log_debug(f"      飞行奖励: {cost_details['flight_reward']:.6f}")
                        log_debug(f"      对偶贡献: {cost_details['dual_contribution']:.6f}")
                        log_debug(f"      任务数: {len(r.duties)}")
                        
                        if reduced_cost < -0.01:  # 放宽阈值，允许更多潜在有价值的roster
                            valuable_count += 1
                            master_problem.add_roster(r)
                            new_rosters_found_count += 1
                            log_debug(f"      ✓ 有价值，已添加到主问题")
                        else:
                            log_debug(f"      ✗ 无价值，不添加")
                    
                    # 记录到日志文件
                    log_debug(f"    机组{crew.crewId}: 生成{len(new_rosters)}个，有价值{valuable_count}个")
                else:
                    log_debug(f"    机组{crew.crewId}: 未生成任何roster")
                        
            except Exception as e:
                # 将详细错误信息记录到日志文件
                log_debug(f"    机组{crew.crewId}: 子问题求解出错 - {e}")
                import traceback
                log_debug(f"    错误堆栈: {traceback.format_exc()}")
                # 控制台只显示简要错误信息
                print(f"    机组{crew.crewId}: 子问题求解出错")
            
            # 更新进度条
            if crew_processed % 10 == 0 or crew_processed == len(crews):
                print_progress_bar(crew_processed, len(crews))
                    
        # 确保进度条完成后换行
        print()  # 换行
        
        # 显示每轮列生成后的最优解变化
        print(f"\n=== 第 {i+1} 轮列生成结果 ===")
        print(f"本轮新增有价值roster数量: {new_rosters_found_count}")
        
        if new_rosters_found_count == 0:
            print("❌ 本轮未找到任何有价值的roster！")
            print("可能原因：")
            print("1. 对偶价格不合理")
            print("2. 约束过于严格")
            print("3. 搜索空间不足")
            print("4. reduced cost计算错误")
        
        # 求解当前主问题获取最优解
        pi_duals, sigma_duals, ground_duty_duals, current_obj_val = master_problem.solve_lp(verbose=True)
        if current_obj_val is not None:  # 求解成功
            print(f"当前主问题最优目标函数值: {current_obj_val:.6f}")
            
            # 跟踪目标函数变化（最小化问题：目标函数值应该递减）
            if i > 0 and previous_obj_val != float('inf'):
                obj_change = current_obj_val - previous_obj_val
                if obj_change > 1e-6:
                    print(f"警告：目标函数增加了 {obj_change:.6f}，不满足列生成的单调性！")
                else:
                    print(f"目标函数变化：{obj_change:.6f}")
            
            # 如果不是第一轮，显示目标函数值的变化
            if i > 0:
                improvement = previous_obj_val - current_obj_val  # 最小化问题：改善 = 上轮值 - 当前值
                print(f"相比上轮的改善: {improvement:.6f}")
                
                # 基于目标函数改善判断收敛（最小化问题：改善应该为正）
                if improvement < 1e-6:  # 改善微小
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
    
    # 计算初始解的目标函数值
    # 使用与主问题一致的线性目标函数值计算方法
    total_flight_hours = 0.0
    total_duty_days = 0.0
    covered_flights = set()
    
    # 计算初始解的线性目标函数值（与主问题一致）
    initial_roster_cost_sum = 0
    for roster in initial_rosters:
        # 使用与主问题一致的成本计算方法
        roster_cost = master_problem._calculate_roster_cost(roster)
        initial_roster_cost_sum += roster_cost
        # 同时更新roster.cost以保持一致性
        roster.cost = roster_cost
        
        # 统计覆盖的航班（用于显示）
        for duty in roster.duties:
            if isinstance(duty, Flight):
                covered_flights.add(duty.id)
                # 计算飞行时间（分钟转小时）
                if hasattr(duty, 'flyTime') and duty.flyTime:
                    total_flight_hours += duty.flyTime / 60.0
                # 记录值勤日期
                if hasattr(duty, 'std') and duty.std:
                    total_duty_days += 1
            elif hasattr(duty, 'startTime') and duty.startTime:
                # 对于非飞行任务，也记录值勤日期
                total_duty_days += 1
    
    uncovered_flights_count = len(flights) - len(covered_flights)
    
    # 计算未覆盖地面任务数量
    covered_ground_duties = set()
    for roster in initial_rosters:
        for duty in roster.duties:
            if hasattr(duty, 'crewId') and hasattr(duty, 'airport'):  # 地面任务特征
                covered_ground_duties.add(duty.id)
    uncovered_ground_duties_count = len(ground_duties) - len(covered_ground_duties)
    
    # 计算初始解的线性目标函数值（与优化目标一致，包含所有惩罚项）
    initial_linear_objective = (initial_roster_cost_sum + 
                               uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY +
                               uncovered_ground_duties_count * master_problem.UNCOVERED_GROUND_DUTY_PENALTY)
    
    # 为了显示，计算竞赛评分公式的值（仅用于参考）
    total_penalties = uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY
    if total_duty_days > 0:
        initial_objective_value = (total_flight_hours * 1000) / total_duty_days - total_penalties / total_duty_days
    else:
        initial_objective_value = 0.0
    
    # 输出初始解的目标函数值信息
    print(f"=== 初始解目标函数分析 ===")
    print(f"总航班数: {len(flights)}")
    print(f"覆盖航班数: {len(covered_flights)}")
    print(f"未覆盖航班数: {uncovered_flights_count}")
    print(f"航班覆盖率: {len(covered_flights)/len(flights)*100:.1f}%")
    print(f"总地面任务数: {len(ground_duties)}")
    print(f"覆盖地面任务数: {len(covered_ground_duties)}")
    print(f"未覆盖地面任务数: {uncovered_ground_duties_count}")
    print(f"地面任务覆盖率: {len(covered_ground_duties)/len(ground_duties)*100:.1f}%" if ground_duties else "地面任务覆盖率: N/A")
    print(f"排班方案数: {len(initial_rosters)}")
    print(f"总飞行时间: {total_flight_hours:.2f} 小时")
    print(f"总值勤天数: {total_duty_days:.0f} 天")
    print(f"日均飞行时间: {total_flight_hours/total_duty_days if total_duty_days > 0 else 0:.2f} 小时")
    print(f"目标函数组成:")
    print(f"  - Roster成本总和: {initial_roster_cost_sum:.2f}")
    print(f"  - 未覆盖航班惩罚: {uncovered_flights_count * master_problem.UNCOVERED_FLIGHT_PENALTY:.2f}")
    print(f"  - 未覆盖地面任务惩罚: {uncovered_ground_duties_count * master_problem.UNCOVERED_GROUND_DUTY_PENALTY:.2f}")
    print(f"初始解线性目标函数值: {initial_linear_objective:.2f}")
    print(f"初始解竞赛评分（日均飞时得分）: {initial_objective_value:.2f}")
    
    # 验证初始解航班覆盖率
    print(f"\n=== 初始解航班覆盖率验证 ===")
    validator = CoverageValidator(min_coverage_rate=0.8)
    initial_coverage_result = validator.validate_coverage(flights, initial_rosters)
    print(validator.get_coverage_report(initial_coverage_result))
    
    # 验证初始解占位任务规则
    print(f"\n=== 初始解占位任务规则验证 ===")
    print("注意：占位任务现在使用软约束结构，允许部分未覆盖")
    ground_duty_validator = GroundDutyValidator(ground_duties)
    initial_ground_duty_result = ground_duty_validator.validate_solution(initial_rosters, master_problem)
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
    final_model = master_problem.solve_bip(verbose=True)
    
    # 调试：显示目标函数值的详细组成
    print(f"\n=== 目标函数调试信息 ===")
    if final_model.SolCount > 0 and final_model.Status == 2:  # 2表示OPTIMAL状态
        obj_val = final_model.ObjVal
        print(f"最终目标函数值: {obj_val:.2f}")
        
        # 分解目标函数

        roster_cost_sum = 0
        uncovered_flights_penalty = 0
        uncovered_ground_duties_penalty = 0
        
        try:
            for roster, var in master_problem.roster_vars.items():
                if var.X > 0.001:  # 被选中的roster（使用小阈值处理LP松弛）
                    roster_cost_sum += roster.cost * var.X
                    
            for flight_id, var in master_problem.uncovered_vars.items():
                if var.X > 0.001:  # 未覆盖的航班（使用小阈值处理LP松弛）
                    uncovered_flights_penalty += master_problem.UNCOVERED_FLIGHT_PENALTY * var.X
            
            # 计算未覆盖占位任务惩罚
            for ground_duty_id, var in master_problem.uncovered_ground_duty_vars.items():
                if var.X > 0.001:  # 未覆盖的占位任务（使用小阈值处理LP松弛）
                    uncovered_ground_duties_penalty += master_problem.UNCOVERED_GROUND_DUTY_PENALTY * var.X
            
            total_calculated = roster_cost_sum + uncovered_flights_penalty + uncovered_ground_duties_penalty
            
            print(f"目标函数组成:")
            print(f"  - 选中Roster成本总和: {roster_cost_sum:.2f}")
            print(f"  - 未覆盖航班惩罚: {uncovered_flights_penalty:.2f}")
            print(f"  - 未覆盖占位任务惩罚: {uncovered_ground_duties_penalty:.2f}")
            print(f"  - 总计: {total_calculated:.2f}")
            print(f"  - 验证: 与目标函数值差异 = {abs(obj_val - total_calculated):.6f}")
        except Exception as e:
            print(f"访问变量值时出错: {e}")
            print("可能原因：模型求解状态异常或变量索引超出范围")
    else:
        print(f"未找到可行解，模型状态: {final_model.Status}")
        if hasattr(final_model, 'SolCount'):
            print(f"解的数量: {final_model.SolCount}")

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
            print("注意：占位任务现在使用软约束结构，允许部分未覆盖")
            ground_duty_validator = GroundDutyValidator(ground_duties, crews)
            ground_duty_result = ground_duty_validator.validate_solution(selected_rosters, master_problem)
            print(ground_duty_validator.get_validation_report(ground_duty_result))
            
            if not coverage_result['is_valid']:
                print("\n⚠️  警告：最终解不满足80%航班覆盖率要求！")
                print("根据竞赛规则，此解决方案可能被判定为无效。")
                suggestions = validator.suggest_improvements(coverage_result)
                for suggestion in suggestions:
                    print(suggestion)
            
            # 使用线性目标函数值进行比较（与优化目标一致）
            final_linear_objective = total_calculated  # 使用手动计算的完整目标函数值
            
            # 初始解的线性目标函数值已在前面计算过，直接使用
            # initial_linear_objective 变量已经包含了正确的值
            
            print(f"\n=== 最终解目标函数分析 ===")
            print(f"最终解线性目标函数值: {final_linear_objective:.2f}")
            print(f"初始解线性目标函数值: {initial_linear_objective:.2f}")
            
            # 比较解的质量（注意：这是最小化问题，目标函数值越小越好）
            # 修改逻辑：即使占位任务没有全部覆盖也可以输出最终解，只要航班覆盖率满足要求
            final_solution_valid = coverage_result['is_valid']  # 只要求航班覆盖率满足要求
            
            if final_linear_objective < initial_linear_objective and final_solution_valid:
                print(f"\n✅ 最终解优于初始解且满足航班覆盖率要求 (改善: {initial_linear_objective - final_linear_objective:.2f})")
                if not ground_duty_result['is_valid']:
                    print(f"⚠️  注意：占位任务覆盖率为 {ground_duty_result['coverage_rate']:.1%}，低于80%建议值，但在软约束结构下仍可输出")
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"output/rosterResult_{timestamp}.csv"
                write_results_to_csv(selected_rosters, output_file, master_problem)
                print(f"最终结果已写入文件: {output_file}")
                final_solution_found = True
            elif final_solution_valid:
                print(f"\n⚠️  最终解满足航班覆盖率要求但劣于初始解 (恶化: {final_linear_objective - initial_linear_objective:.2f})")
                if not ground_duty_result['is_valid']:
                    print(f"⚠️  注意：占位任务覆盖率为 {ground_duty_result['coverage_rate']:.1%}，低于80%建议值，但在软约束结构下仍可输出")
                print("将检查初始解的约束满足情况后决定使用哪个解")
            else:
                print(f"\n❌ 最终解不满足航班覆盖率要求，将使用初始解")
                if not ground_duty_result['is_valid']:
                    print(f"⚠️  注意：占位任务覆盖率为 {ground_duty_result['coverage_rate']:.1%}，低于80%建议值")
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
                print(f"⚠️  注意：占位任务覆盖率为 {initial_ground_duty_result['coverage_rate']:.1%}，低于80%建议值，但在软约束结构下仍可输出")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"output/rosterResult_initial_{timestamp}.csv"
        write_results_to_csv(initial_rosters, output_file, master_problem)
        print(f"初始解已写入文件: {output_file}")
        print(f"初始解统计: 目标函数值（日均飞时得分） {initial_objective_value:.2f}, 未覆盖航班 {uncovered_flights_count} 个")
        print(f"覆盖率: {initial_coverage_result['coverage_rate']:.1%}")
    
    # --- 8. 程序执行总结 ---
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n=== 程序执行总结 ===")
    print(f"总执行时间: {total_time:.2f} 秒")
    print(f"列生成轮数: {i+1}")
    print(f"最终解状态: {'找到满足要求的解' if final_solution_found else '使用初始解'}")
    algorithm_name = "简化线性目标函数算法"
    print(f"使用算法: {algorithm_name}")
    
    # 关闭日志文件
    log_debug(f"\n=== 程序结束 ===\n结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_file.close()
    print(f"\n调试日志已保存到: {log_file_path}")


if __name__ == '__main__':
    if not ATTENTION_AVAILABLE:
        print("Error: Attention module not available!")
        exit(1)
    main()