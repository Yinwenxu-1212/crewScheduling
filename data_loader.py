# file: data_loader.py
import pandas as pd
from data_models import Flight, Crew, GroundDuty, BusInfo, LayoverStation, CrewLegMatch
from typing import Dict, List
import time
import pickle
import os

def load_all_data(data_path: str = './data/') -> Dict:
    print("Loading data...")
    # try:
    start_time = time.time()        
    flights_df = pd.read_csv(data_path + 'flight.csv')
    print(f"flights_df time: {time.time()-start_time}s")
    crews_df = pd.read_csv(data_path + 'crew.csv')
    print(f"crews_df time: {time.time()-start_time}s")
    ground_duty_df = pd.read_csv(data_path + 'groundDuty.csv')
    print(f"ground_duty_df time: {time.time()-start_time}s")
    bus_df = pd.read_csv(data_path + 'busInfo.csv')
    print(f"bus_df time: {time.time()-start_time}s")
    layover_stations_df = pd.read_csv(data_path + 'layoverStation.csv')
    print(f"layover_stations_df time: {time.time()-start_time}s")
    crew_leg_match_df = pd.read_csv(data_path + 'crewLegMatch.csv')
    print(f"crew_leg_match_df time: {time.time()-start_time}s")

    flights = [Flight(**row) for _, row in flights_df.iterrows()]
    print(f"flights time: {time.time()-start_time}s")
    crews = [Crew(**row) for _, row in crews_df.iterrows()]
    print(f"crews time: {time.time()-start_time}s")
    ground_duties = [GroundDuty(**row) for _, row in ground_duty_df.iterrows()]
    print(f"ground_duties time: {time.time()-start_time}s")
    bus_info = [BusInfo(**row) for _, row in bus_df.iterrows()]
    print(f"bus_info time: {time.time()-start_time}s")
    layover_stations = [LayoverStation(**row) for _, row in layover_stations_df.iterrows()]
    print(f"layover_stations time: {time.time()-start_time}s")
    
    # 尝试从缓存文件pkl读取crew_leg_matches，失败则重新创建并保存
    cache_file = data_path + 'crew_leg_matches_cache.pkl'
    try:
        # 检查缓存文件是否存在且比CSV文件新
        csv_file = data_path + 'crewLegMatch.csv'
        if os.path.exists(cache_file) and os.path.exists(csv_file):
            cache_time = os.path.getmtime(cache_file)
            csv_time = os.path.getmtime(csv_file)
            if cache_time > csv_time:
                print("Loading crew_leg_matches from cache...")
                with open(cache_file, 'rb') as f:
                    crew_leg_matches = pickle.load(f)
                print(f"crew_leg_matches loaded from cache time: {time.time()-start_time}s")
            else:
                raise FileNotFoundError("Cache is outdated")
        else:
            raise FileNotFoundError("Cache file not found")
    except (FileNotFoundError, pickle.PickleError, Exception) as e:
        print("Creating crew_leg_matches from CSV (this may take a while)...")
        crew_leg_matches = [CrewLegMatch(**row) for _, row in crew_leg_match_df.iterrows()]
        print(f"crew_leg_matches created time: {time.time()-start_time}s")
        
        # 保存到缓存文件
        try:
            print("Saving crew_leg_matches to cache...")
            with open(cache_file, 'wb') as f:
                pickle.dump(crew_leg_matches, f)
            print("Cache saved successfully.")
        except Exception as save_error:
            print(f"Warning: Failed to save cache: {save_error}")
    
    layover_station_set = {ls.airport for ls in layover_stations}
    print(f"layover_station_set time: {time.time()-start_time}s")

    print("Data loaded successfully.")
    print(f"time: {time.time()-start_time}s")
    
    return {
        "flights": flights, "crews": crews, "ground_duties": ground_duties,
        "bus_info": bus_info, "layover_stations": layover_station_set,
        "crew_leg_matches": crew_leg_matches
    }
    # except FileNotFoundError as e:
    #     print(f"Error: {e}. Make sure data files are in '{data_path}'.")
    #     return None
    # except Exception as e:
    #     print(f"An error occurred during data loading: {e}")
    #     return None