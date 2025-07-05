# file: master_problem.py

import gurobipy as gp
from gurobipy import GRB
from typing import List
from datetime import timedelta
from data_models import Flight, Roster, Crew
from unified_config import config

class MasterProblem:
    def __init__(self, flights: List[Flight], crews: List[Crew], ground_duties: List = None, layover_stations = None):
        self.flights = flights
        self.crews = crews
        self.ground_duties = ground_duties or []
        self.layover_stations = layover_stations or []
        
        # 使用统一配置的参数
        optimization_params = config.get_optimization_params()
        self.FLIGHT_TIME_REWARD = optimization_params['flight_time_reward']
        self.POSITIONING_PENALTY = optimization_params['positioning_penalty']
        self.AWAY_OVERNIGHT_PENALTY = optimization_params['away_overnight_penalty']
        self.NEW_LAYOVER_PENALTY = optimization_params['new_layover_penalty']
        self.UNCOVERED_FLIGHT_PENALTY = optimization_params['uncovered_flight_penalty']
        # 🔧 修复：占位任务未覆盖惩罚（恢复合理值以确保优化器有动力覆盖占位任务）
        # 注意：GroundDuty是占位任务，不是置位任务。置位任务包括飞行置位和大巴置位(BusInfo)
        self.UNCOVERED_GROUND_DUTY_PENALTY = optimization_params['uncovered_ground_duty_penalty']
        
        # 设置线性目标函数
        self.use_simple_objective = True

    def add_roster(self, roster):
        """向主问题添加新的排班方案"""
        self._add_roster(roster)
    
    def _add_roster(self, roster):
        """向模型添加新的排班方案"""
        if not hasattr(self, 'model'):
            self._setup_model()
        
        # 计算roster的成本
        roster_cost = self._calculate_roster_cost(roster)
        
        # 创建roster变量
        var = self.model.addVar(
            vtype=GRB.CONTINUOUS, lb=0, ub=1, 
            obj=roster_cost,
            name=f"roster_{roster.crew_id}_{len(self.roster_vars)}"
        )
        self.roster_vars[roster] = var
        
        # 更新目标函数以包含新的roster成本
        # 重新构建完整的目标函数表达式
        self._update_objective_function()
        
        # 更新机组约束
        if roster.crew_id in self.crew_constraints:
            # 移除旧约束
            old_constr = self.crew_constraints[roster.crew_id]
            self.model.remove(old_constr)
            
            # 收集该机组的所有roster变量
            crew_vars = [v for r, v in self.roster_vars.items() if r.crew_id == roster.crew_id]
            
            # 添加新约束
            self.crew_constraints[roster.crew_id] = self.model.addConstr(
                gp.quicksum(crew_vars) <= 1,
                name=f"crew_{roster.crew_id}"
            )
        
        # 更新航班覆盖约束（精细化模型：roster-航班执行变量）
        for duty in roster.duties:
            if hasattr(duty, 'flightNo') and duty.id in self.flight_constraints:
                # 为当前roster-航班对创建执行变量
                roster_flight_key = (roster.crew_id, duty.id)
                if roster_flight_key not in self.roster_flight_execution_vars:
                    self.roster_flight_execution_vars[roster_flight_key] = self.model.addVar(
                        vtype=GRB.CONTINUOUS, lb=0, ub=1, 
                        name=f"exec_{roster.crew_id}_{duty.id}"
                    )
                
                # 约束1：只有选中的roster才能执行其航班
                # exec_roster_flight <= roster_var
                self.model.addConstr(
                    self.roster_flight_execution_vars[roster_flight_key] <= self.roster_vars[roster],
                    name=f"exec_limit_{roster.crew_id}_{duty.id}"
                )
                
                # 移除旧的航班覆盖约束
                old_constr = self.flight_constraints[duty.id]
                self.model.remove(old_constr)
                
                # 收集该航班的所有roster-航班执行变量
                flight_exec_vars = []
                for key, var in self.roster_flight_execution_vars.items():
                    if key[1] == duty.id:  # key[1]是flight_id
                        flight_exec_vars.append(var)
                
                # 约束2：每个航班最多被一个roster执行
                # sum(exec_roster_flight) + uncovered = 1
                self.flight_constraints[duty.id] = self.model.addConstr(
                    gp.quicksum(flight_exec_vars) + self.uncovered_vars[duty.id] == 1,
                    name=f"flight_unique_exec_{duty.id}"
                )
                
                # 更新航班总执行状态变量
                # flight_execution_var = sum(exec_roster_flight for this flight)
                self.model.addConstr(
                    self.flight_execution_vars[duty.id] == gp.quicksum(flight_exec_vars),
                    name=f"flight_total_exec_{duty.id}"
                )
        
        # 更新占位任务约束（修正：根据ID识别占位任务，ID以Grd_开头）
        # 注意：GroundDuty是占位任务，不是置位任务。置位任务包括飞行置位和大巴置位(BusInfo)
        for duty in roster.duties:
            # 识别占位任务：检查ID是否以Grd_开头或类型为GroundDuty
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
                # 移除旧约束
                old_constr = self.ground_duty_constraints[ground_duty_id]
                self.model.remove(old_constr)
                
                # 收集包含该占位任务的所有roster变量
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
                
                # 添加新约束：覆盖变量 + 未覆盖变量 = 1
                self.ground_duty_constraints[ground_duty_id] = self.model.addConstr(
                    gp.quicksum(covering_vars) + self.uncovered_ground_duty_vars[ground_duty_id] == 1,
                    name=f"ground_duty_{ground_duty_id}"
                )

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
        # 航班执行变量也设置为连续变量
        for var in self.flight_execution_vars.values():
            var.vtype = GRB.CONTINUOUS
        # roster-航班执行变量也设置为连续变量
        for var in self.roster_flight_execution_vars.values():
            var.vtype = GRB.CONTINUOUS
        # 未覆盖占位任务变量也设置为连续变量
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
            
            # 航班覆盖约束的对偶价格（现在是航班执行约束）
            for flight_id, constr in self.flight_constraints.items():
                pi_duals[flight_id] = constr.Pi
            
            # 占位任务约束的对偶价格
            for ground_duty_id, constr in self.ground_duty_constraints.items():
                ground_duty_duals[ground_duty_id] = constr.Pi
            
            obj_val = self.model.ObjVal
            
            if verbose:
                print(f"\n=== 线性目标函数求解结果 ===")
                print(f"目标函数值: {obj_val:.2f}")
                print(f"求解状态: 最优")
                print(f"航班执行变量数量: {len(self.flight_execution_vars)}")
            
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
        # 航班执行变量也设置为二进制
        for var in self.flight_execution_vars.values():
            var.vtype = GRB.BINARY
        # roster-航班执行变量也设置为二进制
        for var in self.roster_flight_execution_vars.values():
            var.vtype = GRB.BINARY
        # 未覆盖占位任务变量也设置为二进制
        for var in self.uncovered_ground_duty_vars.values():
            var.vtype = GRB.BINARY
        
        if verbose:
            print("正在求解最终的BIP模型...")
            print(f"模型包含 {len(self.roster_vars)} 个roster变量")
            print(f"模型包含 {len(self.flight_execution_vars)} 个航班执行变量")
            print(f"模型包含 {len(self.roster_flight_execution_vars)} 个roster-航班执行变量")
            print(f"模型包含 {len(self.uncovered_vars)} 个未覆盖变量")
        
        self.model.optimize()
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
                                task_details.append(f"Flight:{duty.flightNo}")
                            elif hasattr(duty, 'task'):
                                task_details.append(f"Ground:{duty.task}")
                            elif type(duty).__name__ == 'BusInfo':
                                task_details.append(f"Bus:{duty.id}")  # 大巴置位任务
                            elif type(duty).__name__ == 'GroundDuty':
                                task_details.append(f"Ground:{duty.id}")  # 占位任务，非置位任务
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
                    # 计算覆盖的航班数量（替代飞行时间）
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
    
    def _setup_model(self):
        """设置线性目标函数的模型"""
        self.model = gp.Model("MasterProblem")
        self.model.setParam('OutputFlag', 0)
        
        self.roster_vars = {}
        self.uncovered_vars = {}
        self.crew_constraints = {}
        self.flight_constraints = {}
        self.ground_duty_constraints = {}
        # 新增：航班执行变量（用于区分执行和置位）
        self.flight_execution_vars = {}
        # 新增：roster-航班执行变量（精细化模型：每个roster-航班对一个执行变量）
        self.roster_flight_execution_vars = {}
        
        # 为每个航班创建未覆盖变量
        for flight in self.flights:
            self.uncovered_vars[flight.id] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"uncovered_{flight.id}"
            )
            # 为每个航班创建执行变量（表示该航班被实际执行，而非置位）
            self.flight_execution_vars[flight.id] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"execute_{flight.id}"
            )
        
        # 为每个机组创建约束：每个机组最多选择一个roster（初始为空约束，添加roster时更新）
        for crew in self.crews:
            self.crew_constraints[crew.crewId] = self.model.addConstr(
                0 <= 1, name=f"crew_{crew.crewId}"
            )
        
        # 为每个航班创建覆盖约束：航班执行 + 未覆盖 = 1（修改：允许多个机组选择但只有一个执行）
        for flight in self.flights:
            self.flight_constraints[flight.id] = self.model.addConstr(
                self.flight_execution_vars[flight.id] + self.uncovered_vars[flight.id] == 1,
                name=f"flight_cover_{flight.id}"
            )
        
        # 为每个占位任务创建未覆盖变量和软约束
        # 注意：GroundDuty是占位任务，不是置位任务。置位任务包括飞行置位和大巴置位(BusInfo)
        self.uncovered_ground_duty_vars = {}
        for ground_duty in self.ground_duties:
            # 为每个占位任务创建未覆盖变量
            self.uncovered_ground_duty_vars[ground_duty.id] = self.model.addVar(
                vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"uncovered_gd_{ground_duty.id}"
            )
            # 初始约束：未覆盖 = 1（将在添加roster时更新）
            self.ground_duty_constraints[ground_duty.id] = self.model.addConstr(
                self.uncovered_ground_duty_vars[ground_duty.id] == 1, 
                name=f"ground_duty_{ground_duty.id}"
            )
        
        # 设置初始目标函数（最小化成本：未覆盖航班惩罚 + 未覆盖占位任务惩罚 + roster成本 + 航班执行奖励）
        # 根据新要求：飞行时间奖励只通过执行变量获得，roster基础成本不包含飞行奖励
        from unified_config import config
        optimization_params = config.get_optimization_params()
        flight_reward = optimization_params['flight_time_reward']  # 负值表示奖励
        
        obj_expr = gp.quicksum(
            self.UNCOVERED_FLIGHT_PENALTY * var for var in self.uncovered_vars.values()
        ) + gp.quicksum(
            self.UNCOVERED_GROUND_DUTY_PENALTY * var for var in self.uncovered_ground_duty_vars.values()
        ) - gp.quicksum(
            flight_reward * self._get_flight_hours(flight_id) * var for flight_id, var in self.flight_execution_vars.items()
        )
        self.model.setObjective(obj_expr, GRB.MINIMIZE)
    
    def _calculate_roster_cost(self, roster):
        """
        计算roster成本
        
        根据新要求：roster基础成本不包含飞行奖励，飞行奖励只通过执行变量获得
        """
        from scoring_system import ScoringSystem
        
        # 创建评分系统实例，传入必需的参数
        scoring_system = ScoringSystem(self.flights, self.crews, self.layover_stations)
        
        # 获取roster对应的机组
        crew = None
        for c in self.crews:
            if c.crewId == roster.crew_id:
                crew = c
                break
        
        if crew is None:
            # 如果找不到机组，使用默认计算
            return 0
        
        # 使用scoring_system计算详细成本，传入空的对偶价格（与初始解保持一致）
        # 注意：现在total_cost不包含飞行奖励，飞行奖励通过执行变量单独计算
        cost_details = scoring_system.calculate_roster_cost_with_dual_prices(roster, crew, {}, 0.0)
        
        # 直接返回total_cost，现在不包含飞行奖励
        return cost_details['total_cost']
    
    def _get_flight_hours(self, flight_id):
        """获取指定航班的飞行时间（小时）"""
        for flight in self.flights:
            if flight.id == flight_id:
                return flight.flyTime / 60.0  # 分钟转小时
        return 0.0
    
    def _update_objective_function(self):
        """更新目标函数以包含所有roster变量的成本"""
        # 构建完整的目标函数表达式
        # 根据新要求：飞行时间奖励只通过执行变量获得，roster基础成本不包含飞行奖励
        from unified_config import config
        optimization_params = config.get_optimization_params()
        flight_reward = optimization_params['flight_time_reward']  # 负值表示奖励
        
        obj_expr = gp.quicksum(
            self.UNCOVERED_FLIGHT_PENALTY * var for var in self.uncovered_vars.values()
        ) + gp.quicksum(
            self.UNCOVERED_GROUND_DUTY_PENALTY * var for var in self.uncovered_ground_duty_vars.values()
        ) + gp.quicksum(
            roster.cost * var for roster, var in self.roster_vars.items()
        ) - gp.quicksum(
            flight_reward * self._get_flight_hours(flight_id) * var for flight_id, var in self.flight_execution_vars.items()
        )
        
        # 重新设置目标函数
        self.model.setObjective(obj_expr, GRB.MINIMIZE)
