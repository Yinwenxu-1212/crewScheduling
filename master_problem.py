# file: master_problem.py

import gurobipy as gp
from gurobipy import GRB
from typing import List
from datetime import timedelta
from data_models import Flight, Roster, Crew
from unified_config import UnifiedConfig

class MasterProblem:
    def __init__(self, flights: List[Flight], crews: List[Crew], ground_duties: List = None, layover_stations = None):
        self.flights = flights
        self.crews = crews
        self.ground_duties = ground_duties or []
        self.layover_stations = layover_stations or []
        
        # 初始化统一评分系统
        from scoring_system import ScoringSystem
        self.scoring_system = ScoringSystem(flights, crews, layover_stations)
        
        # 使用统一配置的参数
        optimization_params = UnifiedConfig.get_optimization_params()
        self.FLIGHT_TIME_REWARD = optimization_params['flight_time_reward']
        self.POSITIONING_PENALTY = optimization_params['positioning_penalty']
        self.AWAY_OVERNIGHT_PENALTY = optimization_params['away_overnight_penalty']
        self.NEW_LAYOVER_PENALTY = optimization_params['new_layover_penalty']
        self.UNCOVERED_FLIGHT_PENALTY = optimization_params['uncovered_flight_penalty']
        self.UNCOVERED_GROUND_DUTY_PENALTY = optimization_params['uncovered_ground_duty_penalty']
        self.VIOLATION_PENALTY = optimization_params['violation_penalty']
        
        # 设置线性目标函数
        self.use_simple_objective = True
        
        # 新增：分支约束管理（用于分支定价）
        self.branching_constraints = {}  # (crew_id, roster_id) -> value
        self.fixed_variables = {}  # var -> (lb, ub)

    def add_roster(self, roster, is_initial_roster=False):
        """向主问题添加新的排班方案"""
        self._add_roster(roster, is_initial_roster)
    
    def solve_lp(self, verbose=False) -> tuple[dict, dict, dict, float]:
        """求解LP松弛问题"""
        return self._solve_lp(verbose=verbose)
        
    def solve_bip(self, verbose=False):
        """求解二进制整数规划问题"""
        return self._solve_bip(verbose=verbose)

    def get_selected_rosters(self):
        """获取被选中的排班方案"""
        return self._get_selected_rosters()
    
    def get_solution_summary(self):
        """获取解决方案摘要"""
        return self._get_solution_summary()
    
    def _solve_lp(self, verbose=False):
        """求解LP松弛问题"""
        if not hasattr(self, 'model'):
            self._setup_model()
        
        # 设置为连续变量
        for var in self.roster_vars.values():
            var.vtype = GRB.CONTINUOUS
        for var in self.uncovered_vars.values():
            var.vtype = GRB.CONTINUOUS
        for var in self.uncovered_ground_duty_vars.values():
            var.vtype = GRB.CONTINUOUS
        
        self.model.optimize()
        
        if self.model.status == GRB.OPTIMAL:
            # 获取对偶价格
            pi_duals = {}
            sigma_duals = {}
            ground_duty_duals = {}
            
            # 机组约束的对偶价格
            for crew_id, constr in self.crew_constraints.items():
                sigma_duals[crew_id] = constr.Pi
            
            # 航班覆盖约束的对偶价格
            for flight_id, constr in self.flight_constraints.items():
                pi_duals[flight_id] = constr.Pi
            
            # 占位任务约束的对偶价格
            for ground_duty_id, constr in self.ground_duty_constraints.items():
                ground_duty_duals[ground_duty_id] = constr.Pi
            
            obj_val = self.model.ObjVal
            
            if verbose:
                # 计算实际的未覆盖数量
                uncovered_flights_count = 0
                uncovered_ground_duties_count = 0
                
                try:
                    # 统计未覆盖航班数量
                    for flight_id, var in self.uncovered_vars.items():
                        if var.X > 0.5:
                            uncovered_flights_count += 1
                    
                    # 统计未覆盖占位任务数量
                    for ground_duty_id, var in self.uncovered_ground_duty_vars.items():
                        if var.X > 0.5:
                            uncovered_ground_duties_count += 1
                except Exception as e:
                    print(f"计算未覆盖数量时出错: {e}")
                    # 如果计算失败，回退到显示变量数量
                    uncovered_flights_count = len(self.uncovered_vars)
                    uncovered_ground_duties_count = len(self.uncovered_ground_duty_vars)
                
                print(f"\n=== 线性目标函数求解结果 ===")
                print(f"目标函数值: {obj_val:.2f}")
                print(f"求解状态: 最优")
                print(f"roster变量数量: {len(self.roster_vars)}")
                print(f"未覆盖航班数量: {uncovered_flights_count}")
                print(f"未覆盖占位任务数量: {uncovered_ground_duties_count}")
            
            return pi_duals, sigma_duals, ground_duty_duals, obj_val
        else:
            if verbose:
                print(f"求解失败，状态: {self.model.status}")
            return None, None, None, None
    
    def _solve_bip(self, verbose=False):
        """求解BIP问题"""
        if not hasattr(self, 'model'):
            self._setup_model()
        
        # 设置为二进制变量
        for var in self.roster_vars.values():
            var.vtype = GRB.BINARY
        for var in self.uncovered_vars.values():
            var.vtype = GRB.BINARY
        for var in self.uncovered_ground_duty_vars.values():
            var.vtype = GRB.BINARY
        
        # 设置BIP求解参数以提高可行性
        self.model.setParam('TimeLimit', 1200)  # 20分钟时间限制
        self.model.setParam('MIPGap', 0.05)     # 5% MIP gap
        self.model.setParam('MIPFocus', 1)      # 优先找可行解
        
        if verbose:
            print("正在求解最终的BIP模型...")
            print(f"模型包含 {len(self.roster_vars)} 个roster变量")
            print(f"模型包含 {len(self.uncovered_vars)} 个未覆盖航班变量") 
            print(f"模型包含 {len(self.uncovered_ground_duty_vars)} 个未覆盖占位任务变量")
            print(f"BIP求解参数: TimeLimit=1200s, MIPGap=0.05, MIPFocus=1")
        
        # 在求解前验证目标函数设置
        self._validate_objective_function()
        
        self.model.optimize()
        
        # 求解后进行详细的目标函数验证
        if self.model.status == GRB.OPTIMAL and verbose:
            self._detailed_objective_validation()
        
        return self.model
    
    def _get_selected_rosters(self):
        """获取被选中的排班方案"""
        import csv
        from datetime import datetime
        import os
        
        selected = []
        print("=== 调试：排班方案变量值 ===\n")
        print(f"总共有 {len(self.roster_vars)} 个排班方案变量")
        
        # 检查模型状态
        if not hasattr(self, 'model') or self.model.status != GRB.OPTIMAL:
            print(f"模型状态异常: {getattr(self.model, 'status', 'Unknown') if hasattr(self, 'model') else 'Model not found'}")
            return selected
        
        print(f"目标函数值: {self.model.ObjVal:.2f}")
        
        # 确保debug目录存在
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # 创建CSV文件记录所有方案的详细信息
        csv_filename = f"debug/debug_rosters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        try:
            with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['方案编号', '变量值', '成本', '机组ID', '是否选中', '任务详情']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for i, (roster, var) in enumerate(self.roster_vars.items()):
                    try:
                        var_value = var.X
                        is_selected = var_value > 0.5
                        
                        if is_selected:
                            selected.append(roster)
                        
                        # 构建任务详情字符串
                        task_details = []
                        for duty in roster.duties:
                            if hasattr(duty, 'flightNo'):
                                # 区分执行航班和置位航班
                                if getattr(duty, 'is_positioning', False):
                                    task_details.append(f"PositioningFlight:{duty.flightNo}")
                                else:
                                    task_details.append(f"Flight:{duty.flightNo}")
                            elif hasattr(duty, 'task'):
                                task_details.append(f"Ground:{duty.task}")
                            elif type(duty).__name__ == 'BusInfo':
                                task_details.append(f"Bus:{duty.id}")
                            elif type(duty).__name__ == 'GroundDuty':
                                task_details.append(f"Ground:{duty.id}")
                            else:
                                task_details.append(f"Other:{type(duty).__name__}")
                        
                        task_details_str = "; ".join(task_details)
                        
                        writer.writerow({
                            '方案编号': i + 1,
                            '变量值': f"{var_value:.6f}",
                            '成本': f"{roster.cost:.2f}",
                            '机组ID': roster.crew_id,
                            '是否选中': '是' if is_selected else '否',
                            '任务详情': task_details_str
                        })
                    except Exception as e:
                        print(f"处理第{i+1}个排班方案时出错: {e}")
                        writer.writerow({
                            '方案编号': i + 1,
                            '变量值': 'ERROR',
                            '成本': f"{roster.cost:.2f}",
                            '机组ID': roster.crew_id,
                            '是否选中': '错误',
                            '任务详情': 'Variable access failed'
                        })
            
            print(f"总共选中了 {len(selected)} 个排班方案")
            print(f"详细信息已保存到: {csv_filename}")
        except Exception as e:
            print(f"创建调试文件时出错: {e}")
        
        return selected
    
    def _get_solution_summary(self):
        """获取解决方案摘要"""
        if not hasattr(self, 'model') or self.model.status != GRB.OPTIMAL:
            return {}
        
        # 计算基本统计信息
        total_covered_flights = 0
        total_duty_days = 0
        uncovered_flights = 0
        uncovered_ground_duties = 0
        
        try:
            # 统计未覆盖航班
            for flight_id, var in self.uncovered_vars.items():
                if var.X > 0.5:
                    uncovered_flights += 1
            
            # 统计未覆盖占位任务
            for ground_duty_id, var in self.uncovered_ground_duty_vars.items():
                if var.X > 0.5:
                    uncovered_ground_duties += 1
            
            # 统计选中的排班方案
            selected_rosters = []
            for roster, var in self.roster_vars.items():
                if var.X > 0.5:
                    selected_rosters.append(roster)
                    total_covered_flights += sum(1 for duty in roster.duties if hasattr(duty, 'flightNo'))
                    total_duty_days += len([duty for duty in roster.duties if hasattr(duty, 'flightNo') or hasattr(duty, 'task')])
            
            avg_daily_coverage = total_covered_flights / max(total_duty_days, 1)
            
            # 计算覆盖率
            total_flights = len(self.flights)
            covered_flights = total_flights - uncovered_flights
            flight_coverage_rate = covered_flights / total_flights if total_flights > 0 else 0
            
            total_ground_duties = len(self.ground_duties)
            covered_ground_duties = total_ground_duties - uncovered_ground_duties
            ground_duty_coverage_rate = covered_ground_duties / total_ground_duties if total_ground_duties > 0 else 0
            
            return {
                'final_score': self.model.ObjVal,
                'total_covered_flights': total_covered_flights,
                'total_duty_days': total_duty_days,
                'avg_daily_coverage': avg_daily_coverage,
                'uncovered_flights': uncovered_flights,
                'uncovered_ground_duties': uncovered_ground_duties,
                'covered_flights': covered_flights,
                'total_flights': total_flights,
                'flight_coverage_rate': flight_coverage_rate,
                'covered_ground_duties': covered_ground_duties,
                'total_ground_duties': total_ground_duties,
                'ground_duty_coverage_rate': ground_duty_coverage_rate,
                'selected_rosters_count': len(selected_rosters)
             }
        except Exception as e:
            print(f"获取解决方案摘要时出错: {e}")
            return {
                'final_score': self.model.ObjVal if hasattr(self.model, 'ObjVal') else 0,
                'total_covered_flights': 0,
                'total_duty_days': 0,
                'avg_daily_coverage': 0,
                'uncovered_flights': 0,
                'uncovered_ground_duties': 0,
                'covered_flights': 0,
                'total_flights': 0,
                'flight_coverage_rate': 0,
                'covered_ground_duties': 0,
                'total_ground_duties': 0,
                'ground_duty_coverage_rate': 0,
                'selected_rosters_count': 0
             }

    def _validate_objective_function(self):
        """验证目标函数设置是否正确"""
        print("\n=== 目标函数验证 ===")
        
        # 检查roster变量的目标函数系数
        total_roster_coeff = 0
        for roster, var in self.roster_vars.items():
            if hasattr(var, 'Obj'):
                total_roster_coeff += var.Obj
        
        # 检查未覆盖变量的目标函数系数
        total_uncovered_flight_coeff = sum(var.Obj for var in self.uncovered_vars.values() if hasattr(var, 'Obj'))
        total_uncovered_gd_coeff = sum(var.Obj for var in self.uncovered_ground_duty_vars.values() if hasattr(var, 'Obj'))
        
        print(f"Roster变量总目标函数系数: {total_roster_coeff:.2f}")
        print(f"未覆盖航班变量总目标函数系数: {total_uncovered_flight_coeff:.2f}")
        print(f"未覆盖占位任务变量总目标函数系数: {total_uncovered_gd_coeff:.2f}")
        
        # 预期系数验证
        expected_flight_coeff = len(self.uncovered_vars) * self.UNCOVERED_FLIGHT_PENALTY
        expected_gd_coeff = len(self.uncovered_ground_duty_vars) * self.UNCOVERED_GROUND_DUTY_PENALTY
        
        print(f"预期未覆盖航班系数: {expected_flight_coeff:.2f}")
        print(f"预期未覆盖占位任务系数: {expected_gd_coeff:.2f}")

    def _detailed_objective_validation(self):
        """详细的目标函数验证"""
        print("\n=== 详细目标函数验证 ===")
        
        try:
            obj_val = self.model.ObjVal
            print(f"模型目标函数值: {obj_val:.2f}")
            
            # 计算各部分贡献
            roster_contribution = 0
            uncovered_flight_contribution = 0
            uncovered_gd_contribution = 0
            
            # Roster贡献
            for roster, var in self.roster_vars.items():
                if hasattr(var, 'X') and var.X > 0.001:
                    roster_contribution += roster.cost * var.X
                    
            # 未覆盖航班贡献
            uncovered_flights_count = 0
            for flight_id, var in self.uncovered_vars.items():
                if hasattr(var, 'X') and var.X > 0.5:
                    uncovered_flights_count += 1
                    uncovered_flight_contribution += self.UNCOVERED_FLIGHT_PENALTY * var.X
            
            # 未覆盖占位任务贡献
            uncovered_gd_count = 0
            for gd_id, var in self.uncovered_ground_duty_vars.items():
                if hasattr(var, 'X') and var.X > 0.5:
                    uncovered_gd_count += 1
                    uncovered_gd_contribution += self.UNCOVERED_GROUND_DUTY_PENALTY * var.X
            
            total_calculated = roster_contribution + uncovered_flight_contribution + uncovered_gd_contribution
            difference = abs(obj_val - total_calculated)
            
            print(f"目标函数组成分析:")
            print(f"  - 选中Roster成本总和: {roster_contribution:.2f}")
            print(f"  - 未覆盖航班惩罚 ({uncovered_flights_count}个): {uncovered_flight_contribution:.2f}")
            print(f"  - 未覆盖占位任务惩罚 ({uncovered_gd_count}个): {uncovered_gd_contribution:.2f}")
            print(f"  - 计算总和: {total_calculated:.2f}")
            print(f"  - 与模型值差异: {difference:.6f}")
            
            if difference > 1e-3:
                print(f"⚠️  警告：目标函数计算差异过大！")
            else:
                print(f"✅ 目标函数计算一致")
                
        except Exception as e:
            print(f"目标函数验证出错: {e}")

    def _setup_model(self):
        """设置线性目标函数的模型"""
        self.model = gp.Model("MasterProblem")
        self.model.setParam('OutputFlag', 0)
        
        self.roster_vars = {}
        self.uncovered_vars = {}
        self.crew_constraints = {}
        self.flight_constraints = {}
        self.ground_duty_constraints = {}
        
        # 为每个航班创建未覆盖变量
        for flight in self.flights:
            self.uncovered_vars[flight.id] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, 
                obj=self.UNCOVERED_FLIGHT_PENALTY,  # 直接设置目标函数系数
                name=f"uncovered_{flight.id}"
            )
        
        # 为每个占位任务创建未覆盖变量
        self.uncovered_ground_duty_vars = {}
        for ground_duty in self.ground_duties:
            self.uncovered_ground_duty_vars[ground_duty.id] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, 
                obj=self.UNCOVERED_GROUND_DUTY_PENALTY,  # 直接设置目标函数系数
                name=f"uncovered_gd_{ground_duty.id}"
            )
        
        # 为每个机组创建约束：每个机组最多选择一个roster
        for crew in self.crews:
            self.crew_constraints[crew.crewId] = self.model.addConstr(
                0 <= 0, name=f"crew_{crew.crewId}"
            )
        
        # 为每个航班创建覆盖约束：初始为未覆盖 = 1
        for flight in self.flights:
            self.flight_constraints[flight.id] = self.model.addConstr(
                self.uncovered_vars[flight.id] == 1,
                name=f"flight_cover_{flight.id}"
            )
        
        # 为每个占位任务创建覆盖约束：初始为未覆盖 = 1
        for ground_duty in self.ground_duties:
            self.ground_duty_constraints[ground_duty.id] = self.model.addConstr(
                self.uncovered_ground_duty_vars[ground_duty.id] == 1, 
                name=f"ground_duty_{ground_duty.id}"
            )
        
        # 设置目标函数为最小化 - 不需要显式设置，Gurobi会自动使用变量的obj系数
        self.model.ModelSense = GRB.MINIMIZE

    def _add_roster(self, roster, is_initial_roster=False):
        """向模型添加新的排班方案"""
        if not hasattr(self, 'model'):
            self._setup_model()
        
        # 计算roster成本
        roster_cost = self._calculate_roster_cost(roster)
        roster.cost = roster_cost
        
        # 为初始解设置保护下界（降低到0.0以提高求解灵活性）
        lower_bound = 0.0 if is_initial_roster else 0.0
        
        # 创建roster变量，直接设置目标函数系数
        var_name = f"initial_roster_{roster.crew_id}_{len(self.roster_vars)}" if is_initial_roster else f"roster_{roster.crew_id}_{len(self.roster_vars)}"
        var = self.model.addVar(
            vtype=GRB.CONTINUOUS, 
            lb=lower_bound,  # 初始解设置下界保护
            ub=1, 
            obj=roster_cost,  # 直接设置目标函数系数，会自动加入目标函数
            name=var_name
        )
        self.roster_vars[roster] = var
        
        # 更新机组约束
        if roster.crew_id in self.crew_constraints:
            old_constr = self.crew_constraints[roster.crew_id]
            self.model.remove(old_constr)
            
            crew_vars = [v for r, v in self.roster_vars.items() if r.crew_id == roster.crew_id]
            
            self.crew_constraints[roster.crew_id] = self.model.addConstr(
                gp.quicksum(crew_vars) <= 1,
                name=f"crew_{roster.crew_id}"
            )
        
        # 更新航班覆盖约束
        for duty in roster.duties:
            if hasattr(duty, 'flightNo'):
                is_execution = not getattr(duty, 'is_positioning', False)
                
                if is_execution and duty.id in self.flight_constraints:
                    old_constr = self.flight_constraints[duty.id]
                    self.model.remove(old_constr)
                    
                    covering_vars = []
                    for r, v in self.roster_vars.items():
                        for d in r.duties:
                            if (hasattr(d, 'flightNo') and d.id == duty.id and 
                                not getattr(d, 'is_positioning', False)):
                                covering_vars.append(v)
                                break
                    
                    self.flight_constraints[duty.id] = self.model.addConstr(
                        gp.quicksum(covering_vars) + self.uncovered_vars[duty.id] == 1,
                        name=f"flight_cover_{duty.id}"
                    )
        
        # 更新占位任务约束
        for duty in roster.duties:
            is_ground_duty = False
            ground_duty_id = None
            
            if hasattr(duty, 'id') and str(duty.id).startswith('Grd_'):
                is_ground_duty = True
                ground_duty_id = duty.id
            elif type(duty).__name__ == 'GroundDuty':
                is_ground_duty = True
                ground_duty_id = duty.id
            elif hasattr(duty, 'task') and str(duty.task).startswith('Grd_'):
                is_ground_duty = True
                ground_duty_id = duty.task
            
            if is_ground_duty and ground_duty_id in self.ground_duty_constraints:
                old_constr = self.ground_duty_constraints[ground_duty_id]
                self.model.remove(old_constr)
                
                covering_vars = []
                for r, v in self.roster_vars.items():
                    for d in r.duties:
                        duty_id = None
                        if hasattr(d, 'id') and str(d.id).startswith('Grd_'):
                            duty_id = d.id
                        elif type(d).__name__ == 'GroundDuty':
                            duty_id = d.id
                        elif hasattr(d, 'task') and str(d.task).startswith('Grd_'):
                            duty_id = d.task
                        
                        if duty_id == ground_duty_id:
                            covering_vars.append(v)
                            break
                
                self.ground_duty_constraints[ground_duty_id] = self.model.addConstr(
                    gp.quicksum(covering_vars) + self.uncovered_ground_duty_vars[ground_duty_id] == 1,
                    name=f"ground_duty_{ground_duty_id}"
                )
    
    def _calculate_roster_cost(self, roster, include_violations=False):
        """
        计算roster成本c_r
        
        使用统一的评分系统，确保与其他模块的计算逻辑一致
        
        Args:
            roster: 排班方案
            include_violations: 是否包含违规检查（主问题应该包含）
        """
        # 找到对应的机组
        crew = None
        for c in self.crews:
            if c.crewId == roster.crew_id:
                crew = c
                break
        
        if not crew:
            return 0.0
        
        if include_violations:
            # 使用包含违规检查的完整成本计算
            cost_details = self.scoring_system.calculate_roster_cost_with_violations(roster, crew)
            return cost_details['total_cost']
        else:
            # 使用基础成本计算（不包含违规检查）
            return self.scoring_system.calculate_unified_roster_cost(roster, crew)
    
    # === 分支定价相关方法 ===
    
    def fix_variable(self, roster, value):
        """固定一个roster变量的值（用于分支定价）"""
        if roster in self.roster_vars:
            var = self.roster_vars[roster]
            self.fixed_variables[var] = (value, value)
            var.lb = value
            var.ub = value
            
    def add_branching_constraint(self, crew_id: str, roster, value: int):
        """添加分支约束"""
        key = (crew_id, id(roster))
        self.branching_constraints[key] = value
        
        # 如果变量已存在，立即应用约束
        if roster in self.roster_vars:
            self.fix_variable(roster, value)
    
    def get_lp_solution_details(self):
        """获取LP解的详细信息（供分支定价使用）"""
        solution = {}
        fractional_vars = []
        
        for roster, var in self.roster_vars.items():
            if var.X > 1e-6:
                solution[roster] = var.X
                
                # 检查是否为分数解
                if 1e-6 < var.X < 1 - 1e-6:
                    fractional_vars.append({
                        'roster': roster,
                        'var': var,
                        'value': var.X,
                        'crew_id': roster.crew_id,
                        'distance_to_half': abs(var.X - 0.5)
                    })
        
        # 按距离0.5的远近排序（用于分支变量选择）
        fractional_vars.sort(key=lambda x: x['distance_to_half'])
        
        return {
            'solution': solution,
            'fractional_vars': fractional_vars,
            'is_integer': len(fractional_vars) == 0,
            'objective_value': self.model.ObjVal if hasattr(self.model, 'ObjVal') else None
        }
    
    def clone_for_branching(self):
        """创建用于分支的副本"""
        new_mp = MasterProblem(self.flights, self.crews, self.ground_duties, self.layover_stations)
        
        # 复制所有roster
        for roster in self.roster_vars.keys():
            new_mp.add_roster(roster)
        
        # 复制分支约束
        new_mp.branching_constraints = self.branching_constraints.copy()
        
        return new_mp