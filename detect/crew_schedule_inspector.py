#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter

def load_data(data_dir, result_dir):
    """Load all required data files"""
    try:
        flight = pd.read_csv(data_dir / "flight.csv")
        bus = pd.read_csv(data_dir / "busInfo.csv")
        ground = pd.read_csv(data_dir / "groundDuty.csv")
        crew = pd.read_csv(data_dir / "crew.csv")
        crew_leg_match = pd.read_csv(data_dir / "crewLegMatch.csv")
        roster = pd.read_csv(result_dir / "rosterResult.csv")
        return flight, bus, ground, crew, crew_leg_match, roster
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None, None, None, None

def build_task_dict(flight, bus, ground):
    """Build task dictionary from flight, bus and ground data"""
    task_dict = {}
    
    # Flight tasks - use the existing ID directly
    for _, row in flight.iterrows():
        task_id = row['id']  # Already includes 'Flt_' prefix
        task_dict[task_id] = {
            "type": "FLIGHT",
            "depAirport": row["depaAirport"],
            "arrAirport": row["arriAirport"],
            "depTime": row["std"],
            "arrTime": row["sta"],
            "flyTime": row.get("flyTime", 0),
            "aircraftNo": row.get("aircraftNo", ""),
            "fleet": row.get("fleet", "")
        }
    
    # Bus tasks - use the existing ID directly
    for _, row in bus.iterrows():
        task_id = row['id']  # Should already include 'Bus_' prefix
        task_dict[task_id] = {
            "type": "BUS",
            "depAirport": row["depaAirport"],
            "arrAirport": row["arriAirport"],
            "depTime": row["td"],
            "arrTime": row["ta"],
            "flyTime": 0,
            "aircraftNo": "",
            "fleet": ""
        }
    
    # Ground tasks - use the existing ID directly
    for _, row in ground.iterrows():
        task_id = row['id']  # Should already include 'Grd_' prefix
        task_dict[task_id] = {
            "type": "GROUND",
            "depAirport": row["airport"],
            "arrAirport": row["airport"],
            "depTime": row["startTime"],
            "arrTime": row["endTime"],
            "flyTime": 0,
            "aircraftNo": "",
            "fleet": ""
        }
    
    return task_dict

def extract_schedule(roster, crew_id, task_dict, crew_info, crew_leg_match, all_roster):
    """Extract and analyze crew schedule with 12 rules detection"""
    
    # Filter roster for specific crew
    crew_roster = roster[roster["crewId"] == crew_id].copy()
    
    if len(crew_roster) == 0:
        raise ValueError(f"No tasks found for crew {crew_id}")
    
    # Build schedule records
    records = []
    for _, row in crew_roster.iterrows():
        task_id = row["taskId"]
        
        # Get task info from task_dict (including ddh_ tasks from busInfo.csv)
        task_info = task_dict.get(task_id, {})
        if not task_info:
            print(f"Warning: Task {task_id} not found in task dictionary")
            continue
            
        rec = {
            "taskId": task_id,
            "type": task_info.get("type", "UNKNOWN"),
            "dep": task_info.get("depAirport", ""),
            "arr": task_info.get("arrAirport", ""),
            "start": task_info.get("depTime", ""),
            "end": task_info.get("arrTime", ""),
            "flyTime": task_info.get("flyTime", 0),
            "aircraftNo": task_info.get("aircraftNo", ""),
            "warn": "",
        }
        
        records.append(rec)
    
    if not records:
        raise ValueError(f"No valid tasks found for crew {crew_id}")
    
    df = pd.DataFrame(records)
    
    # Convert time format
    df["start_dt"] = pd.to_datetime(df["start"], errors="coerce")
    df["end_dt"] = pd.to_datetime(df["end"], errors="coerce")
    df["date"] = df["start_dt"].dt.date
    
    # Sort by start time
    df = df.sort_values("start_dt").reset_index(drop=True)
    
    print(f"Processing {len(df)} tasks for {crew_id}...")
    
    # ---- 12 Rules Detection ---- #
    
    # Rule 2: Location connection
    crew_stay_station = crew_info.get("stayStation", "")
    if len(df) > 0 and crew_stay_station and df.iloc[0]["dep"] != crew_stay_station:
        df.loc[0, "warn"] += "⚠R2(First_task_location) "
    
    for i in range(len(df) - 1):
        if df.loc[i, "arr"] != df.loc[i + 1, "dep"]:
            df.loc[i + 1, "warn"] += "⚠R2(Location_connection) "
    
    # Rule 3: Minimum connection time
    for i in range(len(df) - 1):
        curr_task = df.iloc[i]
        next_task = df.iloc[i + 1]
        
        # Skip rest tasks
        if "REST" in curr_task["type"] or "REST" in next_task["type"]:
            continue
            
        try:
            interval_hours = (next_task["start_dt"] - curr_task["end_dt"]).total_seconds() / 3600.0
            
            # Flight to flight connection time
            if "FLIGHT" in curr_task["type"] and "FLIGHT" in next_task["type"]:
                # Different aircraft requires 3 hours interval
                if curr_task["aircraftNo"] != next_task["aircraftNo"] and interval_hours < 3:
                    df.loc[i + 1, "warn"] += "⚠R3(Flight_interval) "
            
            # Bus positioning and flight task connection time
            elif "BUS" in curr_task["type"] or "BUS" in next_task["type"]:
                if interval_hours < 2:
                    df.loc[i + 1, "warn"] += "⚠R3(Bus_interval) "
        except:
            continue
    
    # Rule 7: Minimum rest time (12 hours)
    for i in range(len(df) - 1):
        curr_task = df.iloc[i]
        next_task = df.iloc[i + 1]
        
        # If current task end to next task start is cross-duty-day rest
        if curr_task["date"] != next_task["date"]:
            try:
                rest_hours = (next_task["start_dt"] - curr_task["end_dt"]).total_seconds() / 3600.0
                if rest_hours < 12:
                    df.loc[i + 1, "warn"] += "⚠R7(Insufficient_rest) "
            except:
                continue
    
    # Group by duty day for rules 4, 5, 6
    duty_days = df.groupby("date")
    
    for date, day_tasks in duty_days:
        # Filter out rest tasks
        work_tasks = day_tasks[~day_tasks["type"].str.contains("REST", na=False)]
        flight_tasks = work_tasks[work_tasks["type"].str.contains("FLIGHT", na=False)]
        
        if len(work_tasks) == 0:
            continue
            
        # Rule 4: Daily task quantity limit
        if len(flight_tasks) > 4:
            for idx in work_tasks.index:
                df.loc[idx, "warn"] += "⚠R4(Flight_task_count) "
        
        if len(work_tasks) > 6:
            for idx in work_tasks.index:
                df.loc[idx, "warn"] += "⚠R4(Total_task_count) "
        
        # Rule 5: Daily flight time limit (8 hours)
        try:
            total_fly_time = flight_tasks["flyTime"].sum()
            if total_fly_time > 8:
                for idx in flight_tasks.index:
                    df.loc[idx, "warn"] += "⚠R5(Flight_time) "
        except:
            pass
        
        # Rule 6: Daily duty time limit (12 hours)
        if len(flight_tasks) > 0:
            try:
                duty_start = work_tasks["start_dt"].min()
                duty_end = flight_tasks["end_dt"].max()  # Last flight task arrival time
                duty_hours = (duty_end - duty_start).total_seconds() / 3600.0
                
                if duty_hours > 12:
                    for idx in work_tasks.index:
                        df.loc[idx, "warn"] += "⚠R6(Duty_time) "
            except:
                pass
    
    # Rule 9: Total flight duty time limit (60 hours)
    total_duty_time = 0
    for date, day_tasks in duty_days:
        work_tasks = day_tasks[~day_tasks["type"].str.contains("REST", na=False)]
        flight_tasks = work_tasks[work_tasks["type"].str.contains("FLIGHT", na=False)]
        
        if len(flight_tasks) > 0:
            try:
                duty_start = work_tasks["start_dt"].min()
                duty_end = flight_tasks["end_dt"].max()
                duty_hours = (duty_end - duty_start).total_seconds() / 3600.0
                total_duty_time += duty_hours
            except:
                pass
    
    if total_duty_time > 60:
        for idx in df.index:
            if "FLIGHT" in df.loc[idx, "type"]:
                df.loc[idx, "warn"] += "⚠R9(Total_duty_time) "
    
    # Rule 10: Qualification requirements
    try:
        crew_qualified_tasks = set(crew_leg_match[crew_leg_match["crewId"] == crew_id]["legId"])
        for idx, row in df.iterrows():
            if "FLIGHT" in row["type"] and "DDH" not in row["type"]:
                if row["taskId"] not in crew_qualified_tasks:
                    df.loc[idx, "warn"] += "⚠R10(Qualification_mismatch) "
    except:
        pass
    
    # Rule 11: Task overlap detection
    for i in range(len(df) - 1):
        curr_task = df.iloc[i]
        next_task = df.iloc[i + 1]
        
        # Ground tasks can overlap, others cannot
        if "GROUND" not in curr_task["type"] or "GROUND" not in next_task["type"]:
            try:
                if curr_task["end_dt"] > next_task["start_dt"]:
                    df.loc[i + 1, "warn"] += "⚠R11(Task_overlap) "
            except:
                pass
    
    # Rule 12: Captain uniqueness detection
    for idx, row in df.iterrows():
        if "FLIGHT" in row["type"] and "DDH" not in row["type"]:
            try:
                # Check if other captains are also assigned to this task
                other_crews = all_roster[(all_roster["taskId"] == row["taskId"]) & 
                                       (all_roster["crewId"] != crew_id)]
                if len(other_crews) > 0:
                    # Check if other captains are also executing tasks (non-positioning)
                    for _, other_row in other_crews.iterrows():
                        if "isDDH" not in other_row or other_row["isDDH"] == 0:
                            df.loc[idx, "warn"] += "⚠R12(Captain_duplicate) "
                            break
            except:
                pass
    
    return df.drop(columns=["start_dt", "end_dt", "date", "flyTime", "aircraftNo"], errors='ignore')

def main():
    parser = argparse.ArgumentParser(description="Analyze crew schedule and detect violations based on 12 rules")
    parser.add_argument("crew_ids", nargs="+", help="One or more crew IDs, e.g. Crew_10430")
    parser.add_argument("-d", "--data-path", default="submit/0701-1-（-5221）", type=Path, help="Data file directory")
    parser.add_argument("-o", "--output", help="Export results to CSV file if specified")
    args = parser.parse_args()

    # Load data from both data/ and specified result directory
    data_dir = Path("data")
    result_dir = Path(args.data_path)
    
    flight, bus, ground, crew, crew_leg_match, roster = load_data(data_dir, result_dir)
    if flight is None:
        return
    
    print(f"Data loaded: {len(flight)} flights, {len(bus)} bus tasks, {len(ground)} ground tasks")
    print(f"Crew data: {len(crew)} crews, Roster: {len(roster)} assignments")
    
    task_dict = build_task_dict(flight, bus, ground)
    print(f"Task dictionary built with {len(task_dict)} tasks")
    
    # Build crew info dictionary
    crew_dict = {}
    for _, row in crew.iterrows():
        crew_dict[row["crewId"]] = {
            "stayStation": row.get("stayStation", ""),
            "base": row.get("base", "")
        }

    all_df = []
    for cid in args.crew_ids:
        print(f"\n{'='*50}")
        print(f"Analyzing crew: {cid}")
        print(f"{'='*50}")
        
        try:
            crew_info = crew_dict.get(cid, {})
            df = extract_schedule(roster, cid, task_dict, crew_info, crew_leg_match, roster)
            df.insert(0, "crewId", cid)
            all_df.append(df)
            
            print(f"\nSchedule for {cid}:")
            print(df.to_string(index=False))
            
            # Check violations
            violations_df = df[df["warn"] != ""]
            if len(violations_df) > 0:
                print(f"\n🚨 Found {len(violations_df)} tasks with violations:")
                for idx, row in violations_df.iterrows():
                    print(f"  Task {row['taskId']} ({row['type']}): {row['warn']}")
                
                # Summary of violation types
                all_warnings = []
                for warn_str in violations_df["warn"]:
                    # Extract individual warnings (split by space and filter out empty)
                    warnings = [w.strip() for w in warn_str.split() if w.strip()]
                    all_warnings.extend(warnings)
                
                if all_warnings:
                    violation_counts = Counter(all_warnings)
                    print("\n📊 Violation Summary:")
                    for violation, count in violation_counts.items():
                        print(f"  {violation}: {count} times")
            else:
                print(f"\n✅ No violations found for {cid}.")
                
        except ValueError as e:
            print(f"❌ Error analyzing {cid}: {e}")
        except Exception as e:
            print(f"❌ Unexpected error analyzing {cid}: {e}")

    if args.output and all_df:
        output_df = pd.concat(all_df, ignore_index=True)
        output_df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to {args.output}")

if __name__ == "__main__":
    main()
