# file: results_writer.py

import csv
from typing import List
from data_models import Roster, Flight, BusInfo, GroundDuty, RestPeriod

def write_results_to_csv(selected_rosters: List[Roster], output_path: str, master_problem=None):
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
                        # 检查大巴任务是否为置位任务
                        if hasattr(task, 'type') and 'positioning' in str(task.type).lower():
                            is_ddh = 1
                        else:
                            is_ddh = 1  # 默认大巴任务都是置位任务
                    elif isinstance(task, Flight):
                        # For flights, check if this flight is executed (not positioning)
                        # First check if task has type attribute indicating positioning
                        if hasattr(task, 'type') and 'positioning' in str(task.type).lower():
                            is_ddh = 1
                        elif master_problem and hasattr(master_problem, 'flight_execution_vars'):
                            # If the flight execution variable is 0, this flight is used for positioning
                            exec_var = master_problem.flight_execution_vars.get(task.id)
                            if exec_var and hasattr(exec_var, 'X'):
                                # If execution variable is close to 0, this is positioning
                                is_ddh = 1 if exec_var.X < 0.5 else 0
                            else:
                                # Fallback: assume it's execution if no execution variable found
                                is_ddh = 0
                        else:
                            # Fallback: check if flight number starts with "DH" (old logic)
                            is_ddh = 1 if task.flightNo.startswith("DH") else 0
                    
                    writer.writerow([crew_id, task_id, is_ddh])
        
        print(f"\nFinal rosters successfully written to {output_path}")

    except AttributeError as e:
        print(f"\n[ERROR] An error occurred while writing results: {e}.")
        print("Please ensure all task objects (Flight, GroundDuty, BusInfo) have an '.id' attribute.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred while writing results file: {e}")