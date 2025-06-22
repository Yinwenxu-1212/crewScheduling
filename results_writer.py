# file: results_writer.py

import csv
from typing import List
from data_models import Roster, Flight, BusInfo, GroundDuty, RestPeriod

def write_results_to_csv(selected_rosters: List[Roster], output_path: str):
    """
    Writes the final selected rosters to a CSV file with the required
    3-column format: ['crewId', 'taskId', 'isDDH'].
    
    Args:
        selected_rosters (List[Roster]): The list of rosters in the final solution.
        output_path (str): The path for the output CSV file.
    """
    if not selected_rosters:
        print("No rosters were selected. Nothing to write.")
        return

    header = ['crewId', 'taskId', 'isDDH']
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            
            for roster in selected_rosters:
                crew_id = roster.crew_id
                for task in roster.duties:
                    # Skip non-task items like RestPeriod
                    if isinstance(task, RestPeriod):
                        continue

                    # All task objects (Flight, GroundDuty, BusInfo) must have an '.id' attribute
                    task_id = task.id
                    
                    # Determine the isDDH flag based on the task type
                    # isDDH = 1 means it's a positioning task (deadhead).
                    # isDDH = 0 means it's an operational task (flight duty, ground duty).
                    is_ddh = 0
                    if isinstance(task, BusInfo):
                        # Ground positioning is a deadhead task.
                        is_ddh = 1
                    elif isinstance(task, Flight) and task.flightNo.startswith("DH"):
                        # Flights marked with 'DH' are deadhead tasks.
                        is_ddh = 1
                    
                    writer.writerow([crew_id, task_id, is_ddh])
        
        print(f"\nFinal rosters successfully written to {output_path}")

    except AttributeError as e:
        print(f"\n[ERROR] An error occurred while writing results: {e}.")
        print("Please ensure all task objects (Flight, GroundDuty, BusInfo) have an '.id' attribute.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred while writing results file: {e}")