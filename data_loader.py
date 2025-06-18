# file: data_loader.py
import pandas as pd
from data_models import Flight, Crew, GroundDuty, BusInfo, LayoverStation, CrewLegMatch
from typing import Dict, List

def load_all_data(data_path: str = './data/') -> Dict:
    print("Loading data...")
    try:
        flights_df = pd.read_csv(data_path + 'flight.csv')
        crews_df = pd.read_csv(data_path + 'crew.csv')
        ground_duty_df = pd.read_csv(data_path + 'groundDuty.csv')
        bus_df = pd.read_csv(data_path + 'busInfo.csv')
        layover_stations_df = pd.read_csv(data_path + 'layoverStation.csv')
        crew_leg_match_df = pd.read_csv(data_path + 'crewLegMatch.csv')

        flights = [Flight(**row) for _, row in flights_df.iterrows()]
        crews = [Crew(**row) for _, row in crews_df.iterrows()]
        ground_duties = [GroundDuty(**row) for _, row in ground_duty_df.iterrows()]
        bus_info = [BusInfo(**row) for _, row in bus_df.iterrows()]
        layover_stations = [LayoverStation(**row) for _, row in layover_stations_df.iterrows()]
        crew_leg_matches = [CrewLegMatch(**row) for _, row in crew_leg_match_df.iterrows()]
        
        layover_station_set = {ls.airport for ls in layover_stations}

        print("Data loaded successfully.")
        
        return {
            "flights": flights, "crews": crews, "ground_duties": ground_duties,
            "bus_info": bus_info, "layover_stations": layover_station_set,
            "crew_leg_matches": crew_leg_matches
        }
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure data files are in '{data_path}'.")
        return None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None