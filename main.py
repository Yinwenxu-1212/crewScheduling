# file: main.py
# Change: Added a hard time limit to ensure termination within the competition's 2-hour window.

import time # 1. 导入时间模块
from data_loader import load_all_data
from master_problem import MasterProblem
from subproblem_solver import solve_subproblem_for_crew
from results_writer import write_results_to_csv
from gurobipy import GRB

def main():
    # --- TIME LIMIT SETUP START ---
    # 2. 在程序开始时记录当前时间
    start_time = time.time()
    # 3. 设置时间限制（单位：秒）。2小时 = 7200秒。我们设为1小时55分钟（6900秒）以留出安全余量。
    TIME_LIMIT_SECONDS = 1 * 3600 + 55 * 60 
    # --- TIME LIMIT SETUP END ---

    data_path = 'data/'
    all_data = load_all_data(data_path)
    if not all_data:
        print("Data loading failed. Exiting.")
        return

    flights = all_data["flights"]
    crews = all_data["crews"]
    layover_stations = all_data["layover_stations"]
    bus_info = all_data["bus_info"]
    ground_duties = all_data["ground_duties"]

    master_problem = MasterProblem(flights=flights)
    
    # 我们可以将最大迭代次数设置得很高，因为时间限制会成为主要的终止条件
    MAX_ITERATIONS = 500 
    for i in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {i+1} ---")
        
        # --- TIME LIMIT CHECK START ---
        # 4. 在每一轮循环开始时，检查已经过去了多长时间
        elapsed_time = time.time() - start_time
        print(f"Time elapsed: {elapsed_time / 60:.2f} minutes.")
        if elapsed_time > TIME_LIMIT_SECONDS:
            print("TIME LIMIT REACHED. Terminating column generation to proceed to final solve.")
            break
        # --- TIME LIMIT CHECK END ---

        print("Solving master LP...")
        dual_prices = master_problem.solve_lp()
        if dual_prices is None:
            print("Failed to solve master LP. Stopping column generation.")
            break
        
        print("Solving subproblems for all crews...")
        new_rosters_found_count = 0
        for crew in crews:
            crew_specific_ground_duties = [gd for gd in ground_duties if gd.crewId == crew.crewId]
            new_rosters = solve_subproblem_for_crew(
                crew, flights, bus_info, crew_specific_ground_duties, dual_prices, layover_stations
            )
            if new_rosters:
                for roster in new_rosters[:5]: 
                    master_problem.add_roster_column(roster)
                    new_rosters_found_count += 1

        print(f"Found and added {new_rosters_found_count} new rosters (columns).")
        
        # 只有在经过几轮初始迭代后，才因为找不到新列而终止
        if new_rosters_found_count == 0 and i > 5:
            print("\nNo more profitable rosters found. Column generation finished.")
            break
    
    # The 'else' clause for the for-loop is removed as breaking due to time is a normal exit.

    final_model = master_problem.solve_bip()

    if final_model and final_model.status == GRB.OPTIMAL:
        selected_rosters = master_problem.get_selected_rosters()
        print(f"\nFound optimal solution with {len(selected_rosters)} rosters.")
        output_file = "rosterResult.csv" # Ensure correct output filename
        write_results_to_csv(selected_rosters, output_file)
    else:
        print("\nCould not find an optimal integer solution within the time limit.")


if __name__ == "__main__":
    main()