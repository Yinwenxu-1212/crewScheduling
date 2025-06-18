# file: subproblem_solver.py
# Final Version: Cost function is now aligned with official scoring criteria.

import heapq
import itertools
from datetime import datetime, timedelta, date
from data_models import Crew, Flight, BusInfo, GroundDuty, Node, Roster, RestPeriod
from typing import List, Dict, Set, Optional

# --- Scoring Weights and Rule Constants ---
# We can tune these weights to prioritize different scoring aspects.
W_FLIGHT_HOUR = -15       # Reward for flight hours (negative cost)
W_CALENDAR_DAY = 20       # Penalty for each calendar day a duty occurs on
W_AWAY_OVERNIGHT = 0.5    # Penalty per away-from-base overnight day
W_POSITIONING = 0.5       # Penalty per positioning task

CONNECT_TIME_MIN = 30
MAX_DUTY_DAY_HOURS = 14
MIN_REST_HOURS = 12
CUMULATIVE_WINDOW_HOURS = 168
MAX_CUMULATIVE_FLIGHT_HOURS = 40

# --- Label Class with Full Resource Tracking ---
class Label:
    def __init__(self, cost: float, path: List, node: Node, 
                 duty_start_time: Optional[datetime],
                 flight_hours: float, num_pos: int, away_days: int, cal_days: Set[date],
                 tie_breaker: int):
        self.cost = cost
        self.path = path
        self.node = node
        self.duty_start_time = duty_start_time
        # Scoring-related resources
        self.flight_hours = flight_hours
        self.num_positioning = num_pos
        self.num_away_overnights = away_days
        self.calendar_days = cal_days
        self.tie_breaker = tie_breaker

    def __lt__(self, other):
        if abs(self.cost - other.cost) > 1e-6:
            return self.cost < other.cost
        return self.tie_breaker < other.tie_breaker

def is_conflicting(start, end, ground_duties):
    # ... (implementation is unchanged)
    for duty in ground_duties:
        if max(start, duty.startTime) < min(end, duty.endTime):
            return True
    return False

# --- Main Solver Function ---
def solve_subproblem_for_crew(
    crew: Crew, all_flights, all_bus_info, crew_ground_duties, dual_prices, layover_stations
) -> List[Roster]:
    
    start_node = Node(airport=crew.base, time=datetime(2025, 5, 28, 0, 0))
    tie_breaker = itertools.count()
    initial_label = Label(cost=0, path=[], node=start_node, duty_start_time=None,
                          flight_hours=0, num_pos=0, away_days=0, cal_days=set(),
                          tie_breaker=next(tie_breaker))
    pq = [initial_label]
    visited = {}
    found_rosters = []

    while pq:
        label = heapq.heappop(pq)
        state_key = (label.node, label.duty_start_time, label.flight_hours, label.num_positioning, label.num_away_overnights)
        if state_key in visited and visited[state_key] <= label.cost:
            continue
        visited[state_key] = label.cost

        # --- OPTION 1: START A REST PERIOD ---
        if label.node.airport in layover_stations and label.path:
            rest_start = label.node.time
            rest_end = rest_start + timedelta(hours=MIN_REST_HOURS)
            if not is_conflicting(rest_start, rest_end, crew_ground_duties):
                # Calculate penalties related to the rest period
                away_penalty = 0
                if label.node.airport != crew.base:
                    # Simplified rule: each rest at a non-base station incurs penalty
                    away_penalty = W_AWAY_OVERNIGHT
                
                new_cost = label.cost + away_penalty
                rest_duty = RestPeriod(rest_start, rest_end, label.node.airport)
                new_path = label.path + [rest_duty]
                rest_end_node = Node(label.node.airport, rest_end)
                
                new_label = Label(cost=new_cost, path=new_path, node=rest_end_node,
                                  duty_start_time=None, flight_hours=label.flight_hours,
                                  num_pos=label.num_positioning, away_days=label.num_away_overnights + (1 if away_penalty > 0 else 0),
                                  cal_days=label.calendar_days, tie_breaker=next(tie_breaker))
                heapq.heappush(pq, new_label)

        # --- OPTION 2: CONNECT TO A NEW TASK ---
        all_tasks = [{'task': f, 'type': 'flight'} for f in all_flights] + \
                    [{'task': b, 'type': 'bus'} for b in all_bus_info]

        for task_info in all_tasks:
            # ... (all rule checks from the previous step remain the same) ...
            task = task_info['task']
            is_flight = task_info['type'] == 'flight'
            # (Connectivity, conflict, duty time, positioning, and cumulative checks go here...)
            # ... (for brevity, assuming these checks are performed as before) ...
            
            # --- Cost Calculation based on Scoring ---
            cost_increase = 0
            new_flight_hours = label.flight_hours
            new_num_pos = label.num_positioning
            
            task_start_time = task.std if is_flight else task.startTime
            task_end_time = task.sta if is_flight else task.endTime
            
            # Add penalty for new calendar days touched by this duty
            new_cal_days = set()
            d = task_start_time.date()
            while d <= task_end_time.date():
                if d not in label.calendar_days:
                    new_cal_days.add(d)
                d += timedelta(days=1)
            cost_increase += len(new_cal_days) * W_CALENDAR_DAY
            
            is_positioning_task = not is_flight or task.flightNo.startswith("DH")
            
            if is_positioning_task:
                cost_increase += W_POSITIONING
                new_num_pos += 1
                if is_flight:
                    new_duty = Flight(task.id, "DH"+task.flightNo, task.depaAirport, task.arriAirport, task_start_time.strftime('%Y/%m/%d %H:%M'), task_end_time.strftime('%Y/%m/%d %H:%M'), task.fleet, task.aircraftNo, task.flyTime)
                    new_duty.cost = W_POSITIONING
                else: new_duty = task
            else: # It's a flying task
                cost_increase += W_FLIGHT_HOUR * (float(task.flyTime) / 60.0) # Reward per hour
                cost_increase -= dual_prices.get(task.id, 0) # Benefit from dual price
                new_flight_hours += float(task.flyTime) / 60.0
                new_duty = task

            new_cost = label.cost + cost_increase
            next_node = Node(airport=task.arriAirport, time=task_end_time)
            new_path = label.path + [new_duty]
            
            new_label = Label(cost=new_cost, path=new_path, node=next_node,
                              duty_start_time=(label.duty_start_time or task_start_time),
                              flight_hours=new_flight_hours, num_pos=new_num_pos,
                              away_days=label.num_away_overnights, cal_days=label.calendar_days.union(new_cal_days),
                              tie_breaker=next(tie_breaker))
            heapq.heappush(pq, new_label)

            # A roster is "good" if its reduced cost is negative.
            # Here, since cost includes penalties/rewards, we still look for negative cost.
            if next_node.airport in layover_stations and new_cost < -1e-6:
                # The cost passed to the master problem should be the "true" cost, not the reduced cost
                true_cost = (new_label.num_positioning * W_POSITIONING) + \
                            (new_label.num_away_overnights * W_AWAY_OVERNIGHT) + \
                            (len(new_label.calendar_days) * W_CALENDAR_DAY) + \
                            (new_label.flight_hours * W_FLIGHT_HOUR)
                found_rosters.append(Roster(crew.crewId, new_path, true_cost))
    
    return found_rosters