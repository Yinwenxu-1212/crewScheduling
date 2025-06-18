# file: master_problem.py
# Logic restored: Added artificial variables to ensure initial feasibility.

import gurobipy as gp
from gurobipy import GRB
from data_models import Roster, Flight
from typing import List, Dict

class MasterProblem:
    def __init__(self, flights: List[Flight]):
        self.model = gp.Model("CrewScheduling_Master")
        self.model.setParam('OutputFlag', 0)
        self.flights = flights
        self.flight_map = {f.id: f for f in flights}
        
        # --- LOGIC RESTORATION START ---
        # The original code likely used artificial variables to ensure the initial
        # master problem is feasible. We add them back here.
        BIG_M = 100000  # A very high cost for artificial variables

        self.flight_constraints = {}
        for flight in self.flights:
            # For each flight constraint, we add a high-cost artificial variable.
            # This makes the constraint initially: (roster_sum) + art_var = 1
            # Since roster_sum is initially empty, this starts as: art_var = 1.
            # The model is now feasible, and the optimizer will try to drive
            # these expensive artificial variables to zero by finding real rosters.
            art_var = self.model.addVar(obj=BIG_M, name=f"Art_{flight.id}")
            
            # The initial constraint is art_var == 1. When we add a column (roster),
            # Gurobi will modify this to become: roster_var + art_var == 1.
            self.flight_constraints[flight.id] = self.model.addConstr(
                art_var == 1, name=f"cover_{flight.id}"
            )
        # --- LOGIC RESTORATION END ---
        
        self.model.update()
        
        self.rosters: List[Roster] = []
        self.roster_vars: List[gp.Var] = []

    def add_roster_column(self, roster: Roster):
        col = gp.Column()
        for duty in roster.duties:
            if isinstance(duty, Flight) and duty.id in self.flight_constraints:
                col.addTerms(1, self.flight_constraints[duty.id])
        
        new_var = self.model.addVar(
            obj=roster.cost, vtype=GRB.CONTINUOUS,
            name=f"Roster_{len(self.roster_vars)}", column=col
        )
        self.rosters.append(roster)
        self.roster_vars.append(new_var)
        self.model.update()

    def solve_lp(self) -> Dict[str, float]:
        self.model.optimize()
        if self.model.status == GRB.OPTIMAL:
            dual_prices = {f.id: self.flight_constraints[f.id].Pi for f in self.flights}
            return dual_prices
        else:
            # Added more info on failure
            print(f"Master LP solve failed with status code: {self.model.status}")
            return None

    def solve_bip(self):
        if not self.roster_vars:
            print("No rosters generated. Cannot solve final BIP.")
            return None
            
        print("Solving final Binary Integer Program...")
        for var in self.roster_vars:
            var.vtype = GRB.BINARY
        
        self.model.setParam('OutputFlag', 1)
        self.model.optimize()
        return self.model

    def get_selected_rosters(self) -> List[Roster]:
        selected_rosters = []
        if self.model.status == GRB.OPTIMAL:
            for i, var in enumerate(self.roster_vars):
                if var.X > 0.5:
                    selected_rosters.append(self.rosters[i])
        return selected_rosters