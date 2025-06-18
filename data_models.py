# file: data_models.py
# Final version based on official column names from a-T-集.xlsx

from datetime import datetime
from typing import List, Any
import pandas as pd

class Flight:
    """Represents a flight segment. Columns from flight.csv."""
    def __init__(self, id, flightNo, depaAirport, arriAirport, std, sta, fleet, aircraftNo, flyTime):
        self.id = str(id).strip() if pd.notna(id) else None
        self.flightNo = flightNo
        self.depaAirport = depaAirport
        self.arriAirport = arriAirport
        self.std = datetime.strptime(std, '%Y/%m/%d %H:%M')
        self.sta = datetime.strptime(sta, '%Y/%m/%d %H:%M')
        self.fleet = fleet
        self.aircraftNo = aircraftNo
        self.flyTime = int(flyTime)
        self.cost = self.flyTime 

    def __repr__(self):
        return (f"Flight(ID: {self.id}, {self.depaAirport} -> {self.arriAirport}, "
                f"STD: {self.std.strftime('%y/%m/%d %H:%M')}, STA: {self.sta.strftime('%y/%m/%d %H:%M')})")

class Crew:
    """Represents a crew member. Columns from crew.csv."""
    def __init__(self, crewId, base, stayStation):
        self.crewId = str(crewId).strip() if pd.notna(crewId) else None
        self.base = base
        self.stayStation = stayStation

    def __repr__(self):
        return f"Crew(ID: {self.crewId}, Base: {self.base})"

class GroundDuty:
    """Represents a ground duty. Columns from groundDuty.csv."""
    def __init__(self, id, crewId, startTime, endTime, airport, isDuty):
        self.id = str(id).strip() if pd.notna(id) else None
        self.crewId = str(crewId).strip() if pd.notna(crewId) else None
        self.isDuty = isDuty
        self.startTime = datetime.strptime(startTime, '%Y/%m/%d %H:%M')
        self.endTime = datetime.strptime(endTime, '%Y/%m/%d %H:%M')
        self.airport = airport

    def __repr__(self):
        duty_status = "Duty" if self.isDuty else "Rest"
        return (f"GroundDuty(ID: {self.id}, Crew: {self.crewId}, "
                f"Status: {duty_status}, Start: {self.startTime}, End: {self.endTime})")

class BusInfo:
    """Represents ground transportation. Columns from bus.csv."""
    def __init__(self, id, depaAirport, arriAirport, td, ta):
        self.id = id,
        self.depaAirport = depaAirport
        self.arriAirport = arriAirport
        self.startTime = datetime.strptime(td, '%Y/%m/%d %H:%M')
        self.endTime = datetime.strptime(ta, '%Y/%m/%d %H:%M')
        self.cost = 0

    def __repr__(self):
        return f"Bus(Dep: {self.depaAirport}, Arr: {self.arriAirport}, Time: {self.startTime} -> {self.endTime})"

class LayoverStation:
    """Represents a layover station. Columns from layoverStation.csv."""
    def __init__(self, airport):
        self.airport = airport

    def __repr__(self):
        return f"LayoverStation(Airport: {self.airport})"

class CrewLegMatch:
    """Represents crew-flight compatibility. Columns from crewLegMatch.csv."""
    def __init__(self, crewId, legId):
        self.crewId = str(crewId).strip() if pd.notna(crewId) else None
        self.flightId = str(legId).strip() if pd.notna(legId) else None

    def __repr__(self):
        return f"CrewLegMatch(Crew: {self.crewId}, Flight: {self.flightId})"

class RestPeriod:
    """Represents a rest period in a roster."""
    def __init__(self, start_time, end_time, location):
        self.start_time = start_time
        self.end_time = end_time
        self.location = location

    def __repr__(self):
        # Calculating duration for display
        duration = self.end_time - self.start_time
        return f"Rest(at:{self.location}, {duration.total_seconds()/3600:.1f}h)"

    # Add a dummy .cost and .id attribute so it can be added to a path without breaking other code
    @property
    def cost(self):
        return 0
    @property
    def id(self):
        return f"Rest_{self.location}_{self.start_time.isoformat()}"
    
class Roster:
    """Represents a full schedule for one crew member (a column in the master problem)."""
    def __init__(self, crew_id: str, duties: List[Any], cost: float):
        self.crew_id = crew_id
        self.duties = duties
        self.cost = cost
        self.is_ddh = 'DDH' in str(duties)

    def __repr__(self):
        duty_repr = ", ".join([d.flightNo if isinstance(d, Flight) else d.task if isinstance(d, GroundDuty) else type(d).__name__ for d in self.duties])
        return f"Roster(Crew: {self.crew_id}, Cost: {self.cost:.2f}, Duties: [{duty_repr}])"

# --- Helper classes for the subproblem solver ---

class Node:
    """Node for the shortest path algorithm in the subproblem."""
    def __init__(self, airport, time):
        self.airport = airport
        self.time = time

    def __eq__(self, other):
        return self.airport == other.airport and self.time == other.time

    def __hash__(self):
        return hash((self.airport, self.time))
        
    def __repr__(self):
        return f"Node(At: {self.airport}, Time: {self.time.strftime('%H:%M')})"

class Label:
    """Label for resource-constrained shortest path algorithm."""
    def __init__(self, cost, path, current_node):
        self.cost = cost
        self.path = path
        self.current_node = current_node

    def __lt__(self, other):
        return self.cost < other.cost