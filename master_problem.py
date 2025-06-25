# file: master_problem.py

import gurobipy as gp
from gurobipy import GRB
from typing import List
from datetime import timedelta
from data_models import Flight, Roster, Crew

class MasterProblem:
    def __init__(self, flights: List[Flight], crews: List[Crew]):
        self.model = gp.Model("CrewSchedulingMaster")
        self.flights = flights
        self.crews = crews
        self.rosters = []
        self.roster_vars = {}  # 主要决策变量
        self.uncovered_vars = {}  # 辅助变量
        self.flight_constraints = {}
        self.crew_constraints = {}
        self.UNCOVERED_FLIGHT_PENALTY = 5  # 题目规定的未覆盖航班惩罚
        self._setup_model()
    
    def _setup_model(self):
        # 1. 航班覆盖约束（初始时强制所有航班为未覆盖状态）
        for flight in self.flights:
            uncovered_var = self.model.addVar(
                vtype=GRB.BINARY, 
                obj=self.UNCOVERED_FLIGHT_PENALTY,  # 未覆盖的惩罚
                name=f"uncovered_{flight.id}"
            )
            self.uncovered_vars[flight.id] = uncovered_var
            # 修正：初始约束强制所有航班为未覆盖状态
            # 这个约束会在add_roster_column中被正确的覆盖约束替换
            self.flight_constraints[flight.id] = self.model.addConstr(
                uncovered_var == 1, name=f"cover_{flight.id}"
            )
        
        # 2. 机组唯一性约束（初始为合理的约束）
        for crew in self.crews:
            self.crew_constraints[crew.crewId] = self.model.addConstr(
                0 <= 1, name=f"crew_{crew.crewId}"
            )
        
        # 3. 显式设置最小化目标函数
        self.model.setObjective(
            gp.quicksum(self.uncovered_vars.values()) * self.UNCOVERED_FLIGHT_PENALTY,
            sense=GRB.MINIMIZE
        )

    def add_roster_column(self, roster: Roster):
        flight_ids_in_roster = {task.id for task in roster.duties if isinstance(task, Flight)}
        
        var = self.model.addVar(
            vtype=GRB.BINARY, 
            obj=roster.cost,  # 使用原始成本（可能为负）
            name=f"roster_{len(self.rosters)}"
        )
        
        self.rosters.append(roster)
        self.roster_vars[roster] = var
        
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
        
        # 设置目标函数
        self.model.setObjective(
            gp.quicksum(var * roster.cost for roster, var in self.roster_vars.items()) +
            gp.quicksum(self.uncovered_vars.values()) * self.UNCOVERED_FLIGHT_PENALTY,
            sense=GRB.MINIMIZE
        )

    def solve_lp(self, verbose=False) -> tuple[dict, dict, float]:
        """
        求解主问题的线性松弛版本，并返回两种对偶价格和目标函数值。
        参数:
            verbose: 是否输出详细的调试信息
        返回: (pi_duals, sigma_duals, obj_val)
        """
        self.model.update()
        lp_model = self.model.relax()
        lp_model.setParam('Presolve', 0)
        lp_model.setParam('LogToConsole', 0)
        lp_model.optimize()
    
        if lp_model.status == GRB.OPTIMAL:
            obj_val = lp_model.ObjVal
            if verbose:
                print(f"LP松弛最优解目标函数值: {obj_val}")
            
            if verbose:
                # 计算并输出目标函数各个部分的值
                roster_cost_total = 0
                for roster, var_name in self.roster_vars.items():
                    # 在LP松弛模型中获取变量
                    lp_var = lp_model.getVarByName(var_name.VarName)
                    if lp_var is not None:
                        roster_cost_total += lp_var.X * roster.cost
                
                flight_time_total = 0
                for roster, var_name in self.roster_vars.items():
                    roster_flight_time = sum(duty.flyTime / 60.0 for duty in roster.duties if hasattr(duty, 'flyTime'))
                    lp_var = lp_model.getVarByName(var_name.VarName)
                    if lp_var is not None:
                        flight_time_total += lp_var.X * roster_flight_time
                
                duty_days_total = 0
                for roster, var_name in self.roster_vars.items():
                    duty_dates = set()
                    for duty in roster.duties:
                        if hasattr(duty, 'std') and hasattr(duty, 'sta'):
                            start_date = duty.std.date()
                            end_date = duty.sta.date()
                            current_date = start_date
                            while current_date <= end_date:
                                duty_dates.add(current_date)
                                current_date += timedelta(days=1)
                        elif hasattr(duty, 'startTime') and hasattr(duty, 'endTime'):
                            start_date = duty.startTime.date()
                            end_date = duty.endTime.date()
                            current_date = start_date
                            while current_date <= end_date:
                                duty_dates.add(current_date)
                                current_date += timedelta(days=1)
                    duty_days = len(duty_dates)
                    lp_var = lp_model.getVarByName(var_name.VarName)
                    if lp_var is not None:
                        duty_days_total += lp_var.X * duty_days
                
                uncovered_penalty_total = 0
                for flight_id, var_name in self.uncovered_vars.items():
                    lp_var = lp_model.getVarByName(var_name.VarName)
                    if lp_var is not None:
                        uncovered_penalty_total += lp_var.X * self.UNCOVERED_FLIGHT_PENALTY
                
                print("\n目标函数各部分的值:")
                print(f"1. roster.cost总和: {roster_cost_total}")
                print(f"   - 总飞行时间: {flight_time_total} 小时")
                print(f"   - 总值勤日数: {duty_days_total} 天")
                if duty_days_total > 0:
                    print(f"   - 平均每日飞行小时数: {flight_time_total/duty_days_total:.2f} 小时/天")
                print(f"3. 未覆盖航班惩罚: {uncovered_penalty_total}")
                
                # 详细输出未覆盖航班信息
                print(f"   - 未覆盖航班变量总数: {len(self.uncovered_vars)}")
                uncovered_count = 0
                for flight_id, var_name in self.uncovered_vars.items():
                    lp_var = lp_model.getVarByName(var_name.VarName)
                    if lp_var is not None and lp_var.X > 0.001:  # 大于0.001认为是未覆盖
                        uncovered_count += 1

                print(f"   - 实际未覆盖航班数: {uncovered_count}")
                
                print(f"4. 目标函数总值: {obj_val}")
            
            pi_duals = {}
            sigma_duals = {}
            
            # 遍历所有约束，根据名称区分并提取对偶价格
            if verbose:
                print("\n=== 对偶价格调试信息 ===")
            constraint_count = 0
            for constr in lp_model.getConstrs():
                name = constr.ConstrName
                dual_value = constr.Pi
                constraint_count += 1
                
                if verbose and constraint_count <= 10:  # 只打印前10个约束的调试信息
                    print(f"约束 {name}: 对偶价格 = {dual_value:.6f}")
                
                if name.startswith("cover_"):
                    flight_id = name.split('_')[1]
                    pi_duals[flight_id] = dual_value
                # --- 提取机长约束的对偶价格 ---
                elif name.startswith("crew_"):
                    crew_id = name.split('_')[1]
                    sigma_duals[crew_id] = dual_value
            
            if verbose:
                print(f"总约束数: {constraint_count}")
                print(f"航班对偶价格数量: {len(pi_duals)}")
                print(f"机组对偶价格数量: {len(sigma_duals)}")
                
                # 打印一些样本对偶价格
                if pi_duals:
                    sample_flight_ids = list(pi_duals.keys())[:3]
                    for fid in sample_flight_ids:
                        print(f"航班 {fid} 对偶价格: {pi_duals[fid]:.6f}")
                
                if sigma_duals:
                    sample_crew_ids = list(sigma_duals.keys())[:3]
                    for cid in sample_crew_ids:
                        print(f"机组 {cid} 对偶价格: {sigma_duals[cid]:.6f}")
            
            # 确保每个机长都有一个对偶价格，即使它没有约束（通常为0）
            for crew in self.crews:
                if crew.crewId not in sigma_duals:
                    sigma_duals[crew.crewId] = 0.0
    
            return pi_duals, sigma_duals, obj_val
        

    def solve_bip(self):
        print("正在求解最终的BIP模型...")
        self.model.optimize()
        return self.model

    def get_selected_rosters(self):
        """获取被选中的排班方案"""
        import csv
        from datetime import datetime
        
        selected = []
        print("=== 调试：排班方案变量值 ===")
        print(f"总共有 {len(self.roster_vars)} 个排班方案变量")
        print(f"目标函数值: {self.model.ObjVal:.2f}")
        
        # 创建CSV文件记录所有方案的详细信息
        csv_filename = f"debug/debug_rosters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['方案编号', '变量值', '成本', '机组ID', '是否选中', '任务详情']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, (roster, var) in enumerate(self.roster_vars.items()):
                var_value = var.X
                is_selected = var_value > 0.5
                
                # 如果被选中，添加到selected列表
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
                        task_details.append(f"Bus:{duty.id}")
                    elif type(duty).__name__ == 'GroundDuty':
                        task_details.append(f"Ground:{duty.id}")
                    else:
                        task_details.append(f"Other:{type(duty).__name__}")
                
                task_details_str = "; ".join(task_details)
                
                # 写入CSV
                writer.writerow({
                    '方案编号': i + 1,
                    '变量值': f"{var_value:.6f}",
                    '成本': f"{roster.cost:.2f}",
                    '机组ID': roster.crew_id,
                    '是否选中': '是' if is_selected else '否',
                    '任务详情': task_details_str
                })
        
        print(f"总共选中了 {len(selected)} 个排班方案")
        print(f"详细信息已保存到: {csv_filename}")
        return selected
