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
        self.roster_vars = {}
        self.uncovered_vars = {}
        self.flight_constraints = {}
        self.crew_constraints = {}
        self.UNCOVERED_FLIGHT_PENALTY = 5.0
        self._setup_model()

    def _setup_model(self):
        # 1. 航班覆盖约束
        for flight in self.flights:
            uncovered_var = self.model.addVar(vtype=GRB.BINARY, name=f"uncovered_{flight.id}")
            self.uncovered_vars[flight.id] = uncovered_var
            self.flight_constraints[flight.id] = self.model.addConstr(uncovered_var == 1, name=f"cover_{flight.id}")

        # 2. 机长唯一性约束
        for crew in self.crews:
            self.crew_constraints[crew.crewId] = self.model.addConstr(0 <= 1, name=f"crew_{crew.crewId}")

        # 3. 目标函数
        self.model.setObjective(
            gp.quicksum(self.uncovered_vars[f.id] * self.UNCOVERED_FLIGHT_PENALTY for f in self.flights),
            GRB.MINIMIZE
        )
        self.model.update()

    def add_roster_column(self, roster: Roster):
        flight_ids_in_roster = {task.id for task in roster.duties if isinstance(task, Flight)}
        
        constrs = [self.flight_constraints[flight_id] for flight_id in flight_ids_in_roster]
        coeffs = [1.0] * len(flight_ids_in_roster)
        
        # 将新列也加入到对应的机长约束中
        crew_constr = self.crew_constraints[roster.crew_id]
        constrs.append(crew_constr)
        coeffs.append(1.0)

        col = gp.Column(coeffs, constrs)
        var = self.model.addVar(vtype=GRB.BINARY, obj=roster.cost, name=f"roster_{len(self.rosters)}", column=col)
        self.rosters.append(roster)
        self.roster_vars[roster] = var

    def solve_lp(self) -> dict:
        self.model.update()
        lp_model = self.model.relax()
        lp_model.setParam('Presolve', 0)
        lp_model.setParam('LogToConsole', 1)
        print("正在求解主问题LP松弛 (Gurobi Log)...")
        lp_model.optimize()
        if lp_model.status == GRB.OPTIMAL:
            duals = {}
            for constr in lp_model.getConstrs():
                name = constr.ConstrName
                if name.startswith("cover_"):
                    flight_id = name.split('_')[1]
                    duals[flight_id] = constr.Pi
            return duals
        return None

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