# file: master_problem.py

import gurobipy as gp
from gurobipy import GRB
from typing import List
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
        self.UNCOVERED_FLIGHT_PENALTY = 5  # 高惩罚确保尽量覆盖
        self._setup_model()
    
    def _setup_model(self):
        # 1. 航班覆盖约束（初始时允许未覆盖，但不强制）
        for flight in self.flights:
            uncovered_var = self.model.addVar(
                vtype=GRB.BINARY, 
                obj=self.UNCOVERED_FLIGHT_PENALTY,  # 未覆盖的惩罚
                name=f"uncovered_{flight.id}"
            )
            self.uncovered_vars[flight.id] = uncovered_var
            # 修改：初始约束允许未覆盖，但不强制未覆盖
            self.flight_constraints[flight.id] = self.model.addConstr(
                uncovered_var <= 1, name=f"cover_{flight.id}"
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
        
        constrs = []
        coeffs = []
        
        # 创建roster决策变量（使用正确的成本）
        # 确保roster.cost是正数
        roster_cost = abs(roster.cost) if roster.cost < 0 else roster.cost
        
        var = self.model.addVar(
            vtype=GRB.BINARY, 
            obj=roster_cost,  # roster的正数成本
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
        
        # 重新设置完整的目标函数（可选）
        self.model.setObjective(
            gp.quicksum(var * roster.cost for roster, var in self.roster_vars.items()) +
            gp.quicksum(self.uncovered_vars.values()) * self.UNCOVERED_FLIGHT_PENALTY,
            sense=GRB.MINIMIZE
        )

    def solve_lp(self) -> tuple[dict, dict]:
            """
            求解主问题的线性松弛版本，并返回两种对偶价格。
            返回: (pi_duals, sigma_duals)
            """
            self.model.update()
            lp_model = self.model.relax()
            lp_model.setParam('Presolve', 0)
            # 在调试时可以设为1，平时设为0让输出更整洁
            lp_model.setParam('LogToConsole', 0) 
            lp_model.optimize()

            if lp_model.status == GRB.OPTIMAL:
                print(f"LP松弛最优解目标函数值: {lp_model.ObjVal}")
                pi_duals = {}
                sigma_duals = {}
                
                # 遍历所有约束，根据名称区分并提取对偶价格
                for constr in lp_model.getConstrs():
                    name = constr.ConstrName
                    if name.startswith("cover_"):
                        flight_id = name.split('_')[1]
                        pi_duals[flight_id] = constr.Pi
                    # --- 【新增】提取机长约束的对偶价格 ---
                    elif name.startswith("crew_"):
                        crew_id = name.split('_')[1]
                        sigma_duals[crew_id] = constr.Pi
                
                # 确保每个机长都有一个对偶价格，即使它没有约束（通常为0）
                for crew in self.crews:
                    if crew.crewId not in sigma_duals:
                        sigma_duals[crew.crewId] = 0.0

                return pi_duals, sigma_duals
            
            print("主问题LP求解失败，无法获取对偶价格。")
            return None, None

    def solve_bip(self):
        print("正在求解最终的BIP模型...")
        self.model.optimize()
        return self.model

    def get_selected_rosters(self) -> List[Roster]:
        selected = []
        if self.model.SolCount > 0:
            for roster, var in self.roster_vars.items():
                if var.X > 0.5:
                    selected.append(roster)
        return selected
