# file: heuristic_subproblem_solver.py
from datetime import datetime, timedelta
from typing import List, Dict, Set
from data_models import Crew, Flight, BusInfo, GroundDuty, Roster

MAX_DUTY_HOURS = 12
MAX_FLIGHTS_PER_DUTY = 4
MAX_TASKS_PER_DUTY = 6
MAX_FLIGHT_HOURS_PER_DUTY = 8
MIN_CONNECT_FLIGHT = timedelta(hours=3)
MIN_CONNECT_BUS = timedelta(hours=2)


def is_qualified_and_available(task, crew, qualified_ids: Set[str], ground_duties: List[GroundDuty], last_end_time):
    if isinstance(task, Flight):
        if task.id not in qualified_ids:
            return False
        start, end = task.std, task.sta
    else:
        start, end = task.startTime, task.endTime

    for gd in ground_duties:
        if max(start, gd.startTime) < min(end, gd.endTime):
            return False

    if last_end_time:
        gap = start - last_end_time
        if isinstance(task, Flight):
            if gap < MIN_CONNECT_FLIGHT:
                return False
        else:
            if gap < MIN_CONNECT_BUS:
                return False

    return True


def generate_duties_for_crew(
    crew: Crew, all_flights: List[Flight], all_bus: List[BusInfo],
    ground_duties: List[GroundDuty], dual_prices: Dict[str, float],
    crew_leg_match_dict: Dict[str, List[str]]
) -> List[Roster]:
    qualified_ids = set(crew_leg_match_dict.get(crew.crewId, []))
    tasks = [f for f in all_flights if f.id in qualified_ids] + all_bus
    tasks.sort(key=lambda x: x.std if isinstance(x, Flight) else x.startTime)

    rosters = []
    used_ids = set()

    for task in tasks:
        if isinstance(task, Flight):
            start, end = task.std, task.sta
            task_id = task.id
        else:
            start, end = task.startTime, task.endTime
            task_id = task.id
        
        if task_id in used_ids:
            continue

        duty = []
        duty_flights = 0
        duty_tasks = 0
        flight_hours = 0
        current_time = start
        current_loc = task.depaAirport if isinstance(task, Flight) else task.depaAirport

        if not is_qualified_and_available(task, crew, qualified_ids, ground_duties, None):
            continue

        duty.append(task)
        used_ids.add(task_id)

        if isinstance(task, Flight):
            duty_flights += 1
            flight_hours += task.flyTime / 60.0
        duty_tasks += 1

        for next_task in tasks:
            next_id = next_task.id if isinstance(next_task, Flight) else next_task.id
            if next_id in used_ids:
                continue

            next_start = next_task.std if isinstance(next_task, Flight) else next_task.startTime
            next_end = next_task.sta if isinstance(next_task, Flight) else next_task.endTime
            gap = next_start - end
            
            if isinstance(next_task, Flight):
                if gap < MIN_CONNECT_FLIGHT:
                    continue
            else:
                if gap < MIN_CONNECT_BUS:
                    continue

            if not is_qualified_and_available(next_task, crew, qualified_ids, ground_duties, end):
                continue

            projected_flight = flight_hours + (next_task.flyTime / 60.0 if isinstance(next_task, Flight) else 0)
            projected_flights = duty_flights + (1 if isinstance(next_task, Flight) else 0)
            projected_tasks = duty_tasks + 1
            projected_duration = (next_end - start).total_seconds() / 3600.0

            if projected_duration > MAX_DUTY_HOURS or projected_flight > MAX_FLIGHT_HOURS_PER_DUTY or projected_flights > MAX_FLIGHTS_PER_DUTY or projected_tasks > MAX_TASKS_PER_DUTY:
                break

            duty.append(next_task)
            used_ids.add(next_id)
            if isinstance(next_task, Flight):
                duty_flights += 1
                flight_hours += next_task.flyTime / 60.0
            duty_tasks += 1
            end = next_end

        cost = -1000 * flight_hours - 0.5 * sum(1 for t in duty if isinstance(t, BusInfo))
        rosters.append(Roster(crew.crewId, duty, cost))

    return rosters
