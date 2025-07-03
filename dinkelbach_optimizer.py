#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dinkelbach算法实现
用于求解混合整数线性分数规划问题（MILFP）

目标函数：最大化 (1000 * 总飞行时间 / 总值勤日数) - 各种惩罚项
"""

import gurobipy as gp
from gurobipy import GRB
from typing import List, Dict, Tuple, Optional
from data_models import Flight, Roster, Crew, BusInfo, GroundDuty
from datetime import timedelta

class DinkelbachOptimizer:
    def __init__(self, flights: List[Flight], crews: List[Crew]):
        self.flights = flights
        self.crews = crews
        self.rosters = []
        self.roster_vars = {}
        self.uncovered_vars = {}
        self.flight_constraints = {}
        self.crew_constraints = {}
        self.ground_duty_constraints = {}  # 添加地面值勤约束字典
        
        # 惩罚系数（根据竞赛规则设定）
        self.UNCOVERED_FLIGHT_PENALTY = 500  # 未覆盖航班惩罚（大幅增加以提高覆盖率）
        self.NEW_LAYOVER_PENALTY = 10      # 新增过夜站点惩罚
        self.AWAY_OVERNIGHT_PENALTY = 0.5  # 外站过夜惩罚
        self.POSITIONING_PENALTY = 0.5     # 置位惩罚
        self.VIOLATION_PENALTY = 10        # 违规惩罚
        
        # Dinkelbach算法参数
        # 改进：使用更合理的lambda初始值估计
        # 基于平均飞行时间/平均值勤天数的粗略估计
        avg_flight_hours = sum(f.flyTime for f in flights) / (60.0 * len(flights)) if flights else 8.0
        self.lambda_k = avg_flight_hours  # 使用平均飞行时间作为初始估计
        self.tolerance = 1e-6  # 收敛容差
        self.max_iterations = 50  # 最大迭代次数
        
        # 改进的收敛控制参数
        self.min_iterations = 5  # 最小迭代次数，防止过早收敛
        self.stagnation_threshold = 3  # 连续停滞次数阈值
        self.adaptive_tolerance = True  # 自适应容差调整
        
        self._setup_base_model()
    
    def _setup_base_model(self):
        """设置基础模型结构"""
        self.model = gp.Model("CrewSchedulingDinkelbach")
        
        # 设置Gurobi参数以改善对偶价格访问
        self.model.setParam('OutputFlag', 0)  # 关闭详细输出
        self.model.setParam('Presolve', 0)  # 禁用预处理
        self.model.setParam('NumericFocus', 2)  # 提高数值稳定性
        self.model.setParam('InfUnbdInfo', 1)  # 启用无界信息
        self.model.setParam('DualReductions', 0)  # 禁用对偶约简
        
        # 1. 创建未覆盖航班变量（LP松弛：连续变量）
        for flight in self.flights:
            uncovered_var = self.model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"uncovered_{flight.id}"
            )
            self.uncovered_vars[flight.id] = uncovered_var
            
            # 初始约束：所有航班开始时都是未覆盖状态
            self.flight_constraints[flight.id] = self.model.addConstr(
                uncovered_var == 1, name=f"cover_{flight.id}"
            )
        
        # 2. 机组唯一性约束（初始为合理的约束）
        for crew in self.crews:
            self.crew_constraints[crew.crewId] = self.model.addConstr(
                0 <= 1, name=f"crew_{crew.crewId}"
            )
    
    def add_roster_column(self, roster: Roster):
        """添加新的roster列到模型中"""
        flight_ids_in_roster = {task.id for task in roster.duties if isinstance(task, Flight)}
        
        # 计算roster的各项指标
        roster_metrics = self._calculate_roster_metrics(roster)
        
        # 创建roster变量（LP松弛：连续变量）
        var = self.model.addVar(
            vtype=GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=f"roster_{len(self.rosters)}"
        )
        
        self.rosters.append(roster)
        self.roster_vars[roster] = var
        
        # 存储roster的指标信息
        roster.metrics = roster_metrics
        
        # 更新航班覆盖约束
        for flight_id in flight_ids_in_roster:
            # 移除旧约束
            old_constr = self.flight_constraints[flight_id]
            self.model.remove(old_constr)
            
            # 收集所有覆盖此航班的roster变量
            covering_vars = []
            for r, v in self.roster_vars.items():
                if any(isinstance(task, Flight) and task.id == flight_id for task in r.duties):
                    covering_vars.append(v)
            
            # 新约束：uncovered_var + sum(所有覆盖此航班的rosters) == 1
            self.flight_constraints[flight_id] = self.model.addConstr(
                self.uncovered_vars[flight_id] + gp.quicksum(covering_vars) == 1,
                name=f"cover_{flight_id}"
            )
        
        # 更新机组唯一性约束
        old_crew_constr = self.crew_constraints[roster.crew_id]
        self.model.remove(old_crew_constr)
        
        # 收集此机组的所有roster变量
        crew_vars = [v for r, v in self.roster_vars.items() if r.crew_id == roster.crew_id]
        
        # 新约束：sum(此机组的所有rosters) <= 1
        self.crew_constraints[roster.crew_id] = self.model.addConstr(
            gp.quicksum(crew_vars) <= 1,
            name=f"crew_{roster.crew_id}"
        )
        # 强制Gurobi模型立即处理所有挂起的修改
        self.model.update()
    
    def _calculate_roster_metrics(self, roster: Roster) -> Dict:
        """计算roster的各项指标"""
        total_flight_hours = 0.0
        duty_days = 0
        positioning_count = 0
        away_overnight_days = 0
        new_layover_stations = set()
        
        # 计算值勤日数（简化版本：直接累加每个roster的值勤天数）
        duty_dates = set()
        for duty in roster.duties:
            if isinstance(duty, Flight):
                total_flight_hours += duty.flyTime / 60.0
                
                # 计算值勤日历日
                start_date = duty.std.date()
                end_date = duty.sta.date()
                current_date = start_date
                while current_date <= end_date:
                    duty_dates.add(current_date)
                    current_date += timedelta(days=1)
            
            # 统一的置位判断逻辑
            if hasattr(duty, 'type') and 'positioning' in duty.type.lower():
                positioning_count += 1
            elif isinstance(duty, BusInfo):  # BusInfo类型直接视为置位
                positioning_count += 1
        
        duty_days = len(duty_dates)
        
        return {
            'total_flight_hours': total_flight_hours,
            'duty_days': duty_days,
            'positioning_count': positioning_count,
            'away_overnight_days': away_overnight_days,
            'new_layover_stations': len(new_layover_stations)
        }
    
    def _update_objective_function(self, lambda_k: float):
        """根据当前lambda_k更新目标函数"""
        # 构建Dinkelbach参数化后的目标函数
        # 最大化: 50 * C_p * x_p - lambda_k * d_p * x_p - 其他惩罚项
        
        objective_expr = gp.LinExpr()
        roster_terms_count = 0
        uncovered_terms_count = 0
        
        # 1. Roster相关项：50 * 总飞行时间 - lambda_k * 值勤天数 - 其他惩罚
        for roster, var in self.roster_vars.items():
            metrics = roster.metrics
            
            # 飞行时间收益项
            flight_hours_coeff = 50 * metrics['total_flight_hours']
            
            # 值勤天数惩罚项（Dinkelbach参数化）
            duty_days_coeff = -lambda_k * metrics['duty_days']
            
            # 其他惩罚项
            positioning_coeff = -self.POSITIONING_PENALTY * metrics['positioning_count']
            away_overnight_coeff = -self.AWAY_OVERNIGHT_PENALTY * metrics['away_overnight_days']
            new_layover_coeff = -self.NEW_LAYOVER_PENALTY * metrics['new_layover_stations']
            
            # 总系数
            total_coeff = (flight_hours_coeff + duty_days_coeff + 
                          positioning_coeff + away_overnight_coeff + new_layover_coeff)
            
            objective_expr.addTerms(total_coeff, var)
            roster_terms_count += 1
        
        # 2. 未覆盖航班惩罚项
        for flight_id, uncovered_var in self.uncovered_vars.items():
            objective_expr.addTerms(-self.UNCOVERED_FLIGHT_PENALTY, uncovered_var)
            uncovered_terms_count += 1
        
        # 第一步：打印并验证目标函数表达式
        print(f"目标函数调试: roster项数={roster_terms_count}, 未覆盖项数={uncovered_terms_count}, λ={lambda_k:.6f}")
        if roster_terms_count == 0 and uncovered_terms_count == 0:
            print("⚠️ 警告：目标函数表达式为空！")
        
        # 详细打印目标函数表达式的内容
        print(f"目标函数表达式详情:")
        print(f"  - 表达式大小: {objective_expr.size()}")
        print(f"  - 常数项: {objective_expr.getConstant()}")
        
        # 打印前5个变量的系数示例
        if objective_expr.size() > 0:
            print(f"  - 前5个变量系数示例:")
            count = 0
            for i in range(min(5, objective_expr.size())):
                try:
                    var = objective_expr.getVar(i)
                    coeff = objective_expr.getCoeff(i)
                    var_name = var.VarName if hasattr(var, 'VarName') else f"Var_{i}"
                    print(f"    变量 {var_name}: 系数 {coeff:.6f}")
                    count += 1
                    if count >= 5:
                        break
                except Exception as e:
                    print(f"    变量 {i}: 无法获取详细信息 ({e})")
                    count += 1
                    if count >= 5:
                        break
        
        # 设置目标函数（最大化）
        self.model.setObjective(objective_expr, sense=GRB.MAXIMIZE)
        
        # 第二步：将Gurobi模型写入文件进行详细分析
        try:
            model_filename = f"debug_model_lambda_{lambda_k:.2f}.lp"
            self.model.write(model_filename)
            print(f"模型已写入文件: {model_filename}")
        except Exception as e:
            print(f"写入模型文件失败: {e}")
    
    def solve_dinkelbach_with_multiple_starts(self, num_starts: int = 3, verbose: bool = True) -> Tuple[Optional[Dict], Optional[Dict], float]:
        """使用多起始点策略的Dinkelbach算法求解分数规划问题"""
        
        if verbose:
            print(f"开始多起始点Dinkelbach算法求解 (起始点数量: {num_starts})...")
        
        best_solution = None
        best_objective = float('-inf')
        best_pi_duals = None
        best_sigma_duals = None
        
        # 计算不同的初始λ值
        avg_flight_hours = sum(f.flyTime for f in self.flights) / (60.0 * len(self.flights)) if self.flights else 8.0
        initial_lambdas = [
            avg_flight_hours * 0.5,  # 保守估计
            avg_flight_hours,        # 平均估计
            avg_flight_hours * 1.5,  # 激进估计
        ]
        
        # 如果需要更多起始点，添加随机扰动
        if num_starts > 3:
            import random
            for i in range(num_starts - 3):
                perturbation = random.uniform(0.3, 2.0)
                initial_lambdas.append(avg_flight_hours * perturbation)
        
        for start_idx, initial_lambda in enumerate(initial_lambdas[:num_starts]):
            # 简化起始点信息输出
            
            # 重置lambda值
            self.lambda_k = initial_lambda
            
            # 求解当前起始点
            pi_duals, sigma_duals, objective = self._solve_single_dinkelbach(verbose=verbose)
            
            # 检查是否是更好的解
            if objective > best_objective:
                best_objective = objective
                best_pi_duals = pi_duals
                best_sigma_duals = sigma_duals
                # 找到更好的解
        
        if verbose:
            print(f"多起始点求解完成，最佳目标函数值: {best_objective:.6f}")
        
        return best_pi_duals, best_sigma_duals, best_objective
    
    def solve_dinkelbach(self, verbose: bool = True) -> Tuple[Optional[Dict], Optional[Dict], float]:
        """使用Dinkelbach算法求解分数规划问题（单起始点版本）"""
        return self._solve_single_dinkelbach(verbose)
    
    def _solve_single_dinkelbach(self, verbose: bool = True) -> Tuple[Optional[Dict], Optional[Dict], float]:
        """单起始点Dinkelbach算法实现"""
        
        if verbose:
            print("开始Dinkelbach算法求解...")
        
        for iteration in range(self.max_iterations):
            # 简化迭代信息输出
            
            # 更新目标函数
            self._update_objective_function(self.lambda_k)
            
            # 求解当前参数化问题
            self.model.optimize()
            
            if self.model.Status != GRB.OPTIMAL:
                if verbose:
                    print(f"模型求解失败，状态: {self.model.Status}")
                # 返回空的对偶价格字典
                pi_duals = {flight_id: 0.0 for flight_id in self.flight_constraints.keys()}
                sigma_duals = {crew_id: 0.0 for crew_id in self.crew_constraints.keys()}
                return pi_duals, sigma_duals, float('inf')
            
            # 计算当前解的分子和分母
            numerator = 0.0  # 1000 * 总飞行时间 - 其他惩罚项
            denominator = 0.0  # 总值勤天数
            
            for roster, var in self.roster_vars.items():
                if var.X > 0.5:  # 被选中的roster
                    metrics = roster.metrics
                    numerator += 50 * metrics['total_flight_hours']  # 修复：与目标函数中的50系数保持一致
                    denominator += metrics['duty_days']
                    
                    # 减去其他惩罚项
                    numerator -= (self.POSITIONING_PENALTY * metrics['positioning_count'] +
                                 self.AWAY_OVERNIGHT_PENALTY * metrics['away_overnight_days'] +
                                 self.NEW_LAYOVER_PENALTY * metrics['new_layover_stations'])
            
            # 减去未覆盖航班惩罚
            for uncovered_var in self.uncovered_vars.values():
                if uncovered_var.X > 0.5:
                    numerator -= self.UNCOVERED_FLIGHT_PENALTY
            
            # 计算新的lambda值
            if denominator > 1e-9:
                new_lambda = numerator / denominator
            else:
                new_lambda = 0.0
            
            # 简化迭代详细信息输出
            
            # 改进的收敛性检查
            lambda_change = abs(new_lambda - self.lambda_k)
            
            # 自适应容差调整
            current_tolerance = self.tolerance
            if self.adaptive_tolerance and iteration > 10:
                # 后期迭代使用更严格的容差
                current_tolerance = self.tolerance * 0.1
            
            # 检查收敛性（需要满足最小迭代次数）
            if iteration >= self.min_iterations and lambda_change < current_tolerance:
                if verbose:
                    print(f"Dinkelbach算法收敛！最终lambda = {new_lambda:.6f}")
                break
            
            # 检查停滞情况
            if hasattr(self, '_lambda_history'):
                self._lambda_history.append(new_lambda)
                # 保留最近的历史记录
                if len(self._lambda_history) > self.stagnation_threshold + 2:
                    self._lambda_history = self._lambda_history[-(self.stagnation_threshold + 2):]
                
                # 检查是否连续停滞
                if len(self._lambda_history) >= self.stagnation_threshold:
                    recent_changes = [abs(self._lambda_history[i] - self._lambda_history[i-1]) 
                                    for i in range(1, len(self._lambda_history))]
                    if all(change < current_tolerance * 10 for change in recent_changes[-self.stagnation_threshold:]):
                        if verbose:
                            print(f"检测到λ值停滞，提前终止迭代")
                        break
            else:
                self._lambda_history = [self.lambda_k, new_lambda]
            
            self.lambda_k = new_lambda
        
        else:
            if verbose:
                print(f"Dinkelbach算法达到最大迭代次数 {self.max_iterations}")
        
        # 提取对偶价格
        pi_duals = {}
        sigma_duals = {}
        
        # 获取对偶价格（LP松弛模型可以直接获取）
        if self.model.Status == GRB.OPTIMAL:
            # 简化模型状态输出
            
            # 直接获取对偶价格（连续变量模型）
            for flight_id, constr in self.flight_constraints.items():
                pi_duals[flight_id] = -constr.Pi
            
            for crew_id, constr in self.crew_constraints.items():
                sigma_duals[crew_id] = -constr.Pi
                
            # 简化对偶价格输出
        else:
            if verbose:
                print(f"模型状态非最优: Status={self.model.Status}")
            # 如果模型未达到最优解，返回零对偶价格
            for flight_id in self.flight_constraints.keys():
                pi_duals[flight_id] = 0.0
            
            for crew_id in self.crew_constraints.keys():
                sigma_duals[crew_id] = 0.0
        
        # 计算原始问题的目标函数值（不是Dinkelbach参数化后的值）
        # 原始问题目标函数：(100 * 总飞行时间 - 各种惩罚项) / 总值勤天数
        final_numerator = 0.0
        final_denominator = 0.0
        
        for roster, var in self.roster_vars.items():
            if var.X > 0.5:  # 被选中的roster
                metrics = roster.metrics
                final_numerator += 100 * metrics['total_flight_hours']
                final_denominator += metrics['duty_days']
                
                # 减去其他惩罚项
                final_numerator -= (self.POSITIONING_PENALTY * metrics['positioning_count'] +
                                   self.AWAY_OVERNIGHT_PENALTY * metrics['away_overnight_days'] +
                                   self.NEW_LAYOVER_PENALTY * metrics['new_layover_stations'])
        
        # 减去未覆盖航班惩罚
        for uncovered_var in self.uncovered_vars.values():
            if uncovered_var.X > 0.5:
                final_numerator -= self.UNCOVERED_FLIGHT_PENALTY
        
        # 计算原始问题的目标函数值（日均飞时得分）
        if final_denominator > 1e-9:
            original_objective = final_numerator / final_denominator
        else:
            original_objective = 0.0
            
        return pi_duals, sigma_duals, original_objective
    
    def get_solution_summary(self) -> Dict:
        """获取当前解的详细信息"""
        if self.model.Status != GRB.OPTIMAL:
            return {}
        
        selected_rosters = []
        total_flight_hours = 0.0
        total_duty_days = 0.0
        uncovered_flights = 0
        
        # 统计选中的rosters
        for roster, var in self.roster_vars.items():
            if var.X > 0.5:
                selected_rosters.append(roster)
                metrics = roster.metrics
                total_flight_hours += metrics['total_flight_hours']
                total_duty_days += metrics['duty_days']
        
        # 统计未覆盖航班
        for uncovered_var in self.uncovered_vars.values():
            if uncovered_var.X > 0.5:
                uncovered_flights += 1
        
        # 计算最终得分（原始问题的目标函数值：日均飞时得分）
        # 原始问题目标函数：(100 * 总飞行时间 - 各种惩罚项) / 总值勤天数
        total_penalties = 0.0
        
        # 计算所有惩罚项
        for roster in selected_rosters:
            metrics = roster.metrics
            total_penalties += (self.POSITIONING_PENALTY * metrics['positioning_count'] +
                               self.AWAY_OVERNIGHT_PENALTY * metrics['away_overnight_days'] +
                               self.NEW_LAYOVER_PENALTY * metrics['new_layover_stations'])
        
        # 加上未覆盖航班惩罚
        total_penalties += uncovered_flights * self.UNCOVERED_FLIGHT_PENALTY
        
        # 计算原始问题的目标函数值（日均飞时得分）
        if total_duty_days > 0:
            final_score = (100 * total_flight_hours - total_penalties) / total_duty_days
            avg_daily_flight_hours = total_flight_hours / total_duty_days
        else:
            final_score = 0.0
            avg_daily_flight_hours = 0.0
        
        return {
            'selected_rosters': selected_rosters,
            'total_flight_hours': total_flight_hours,
            'total_duty_days': total_duty_days,
            'avg_daily_flight_hours': avg_daily_flight_hours,
            'uncovered_flights': uncovered_flights,
            'final_score': final_score,
            'lambda_final': self.lambda_k
        }