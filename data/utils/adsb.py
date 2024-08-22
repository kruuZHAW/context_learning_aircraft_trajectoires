import os
from pathlib import Path
from typing import Union, Tuple
import glob


from datetime import datetime, timezone
import pandas as pd
from traffic.data import airports, opensky
from traffic.core import Traffic, Flight

import multiprocessing as mp
from tqdm.auto import tqdm

def combine_adsb(path_raw: str, path_combined: str):
    """
    Combines all parquet files in the path provided in "path_raw" into one
    parquet file and saves it in the "path_combined" folder.

    Parameters
    ----------
    path_raw : str
        Folder path where the daily parquet files are stored
    path_combined : str
        Folder path where the combined parquet file will be stored
    """
    # Iterate over all monthly folders
    folder_paths = glob.glob(f"{path_raw}/*/*/")
    for path in tqdm(folder_paths):
        # extract year and month from the path
        path_parts = path.split(os.sep)
        year, month = path_parts[-3], path_parts[-2]
        # check whether file already exists
        check_file = Path(f"{path_combined}/{year}_{month}.parquet")
        if check_file.is_file() is False:
            # list all parquet files in the raw folder
            files = glob.glob(f"{path}/*.parquet")
            if files:
                # concatenate all files into one Traffic object
                alldata = Traffic(
                    pd.concat(
                        [Traffic.from_file(file).data for file in files],
                        ignore_index=True,
                    )
                )
                # create "combined" folder if it does not exist
                if os.path.isdir(path_combined) is False:
                    os.mkdir(path_combined)
                # save the Traffic object as a parquet file
                alldata.to_parquet(f"{path_combined}/{year}_{month}.parquet")
                    
def download_adsb_para(
    start: str,
    stop: str,
    folder: str,
    departure_airport: str,
    arrival_airport:str,
    max_process: int = 8,
):
    """
    Parllelisation of the download_adsb function. Queries ADS-B data from
    Opensky Network for the given time interval, geographical footprint,
    altitude constraints and saves the data as Traffic in a parquet file format
    one day at a time.

    Parameters
    ----------
    start : str
        Start datetime for the query in the format "YYYY-MM-DD"
    stop : str
        Stop datetime for the query in the format "YYYY-MM-DD"
    folder : str
        Path to the folder where the data will be saved as one parquet file per
        day
    departure_airport : str
        Departure airport
    arrival_airport : str
        Arrival airport
    max_process : int, optional
        Number of processes to use for parallelization, by default 8
    """
    # create list of dates between start and stop
    dates = pd.date_range(start, stop, freq="D", tz="UTC")
    # create folder if it does not exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    # create list of arguments for the download_adsb function
    t0 = dates[:-1]
    tf = dates[1:]
    fol = [folder for i in range(len(t0))]
    deps = [departure_airport for i in range(len(t0))]
    arrs = [arrival_airport for i in range(len(t0))]
    t = [
        (t0, t1, fol, airs, other_paramsl)
        for t0, t1, fol, airs, other_paramsl in zip(
            t0, tf, fol, deps, arrs
        )
    ]
    # run the download_adsb function in parallel
    with mp.Pool(max_process) as pool:
        pool.starmap(download_adsb, t)
        
def download_adsb(
    t0: str,
    tf: str,
    folder: str,
    departure_airport: str,
    arrival_airport:str,
):
    """
    Queries ADS-B data from Opensky Network for the given time interval,
    geographical footprint, altitude constraints and saves the data as Traffic
    in a parquet file format.

    Parameters
    ----------
    t0 : str
        Start datetime for the query in the format "YYYY-MM-DD hh:mm:ss"
    tf : str
        Stop datetime for the query in the format "YYYY-MM-DD hh:mm:ss"
    folder : str
        Path to the folder where the data will be saved
    departure_airport : str
        Departure airport
    arrival_airport : str
        Arrival airport
    """
    
    year = t0.year
    month = t0.month
    path = f"{folder}/{year}/{month}"
    if not os.path.exists(path):
        os.makedirs(path)
    # check whether file already exists
    check_file = Path(f"{path}/{t0.date()}_{tf.date()}.parquet")
    # if not, print which day is being downloaded and download
    if check_file.is_file() is False:
        print(f"Downloading {t0.date()}...")
        traffic_data = opensky.history(
            t0,
            tf,
            departure_airport=departure_airport,
            arrival_airport=arrival_airport,
            progressbar=tqdm,
            cached=True,
        )
        # if not empty, save. Otherwise, print empty day
        if traffic_data is None:
            print("empty day")
        else:
            traffic_data.to_parquet(f"{path}/{t0.date()}_{tf.date()}.parquet")