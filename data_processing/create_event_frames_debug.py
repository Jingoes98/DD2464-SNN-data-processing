import pandas as pd
import numpy as np
import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from utility import *

cuda_device = 0
start_vedio_nr = 1
end_video_nr = 1
decay_rate = 1
frame_width = 640
frame_height = 480
binning_interval = 10 # in milliseconds
n_frames_per_clip = 1 * 60 * 1000 // binning_interval  # 1 minute of events at 10ms intervals

torch.cuda.set_device(cuda_device)

dir2event_clips = "raw_data/2024_09_10_16_00_41_recordings/processed/event_clips"
path2timewindows = "raw_data/2024_09_10_16_00_41_recordings/timestamps.csv"

for video in range(start_vedio_nr, end_video_nr + 1):
    print(f"Processing event clip number {video}...")
    video_nr = video
    file_path = os.path.join(dir2event_clips, f'clip_{video_nr}.csv')
    df_events = pd.read_csv(file_path, on_bad_lines='warn') # debug
    df_timestamps = pd.read_csv(path2timewindows, skiprows=range(0, video_nr * n_frames_per_clip), nrows=n_frames_per_clip)

    # Extracting data from the events DataFrame
    x_coords = df_events.iloc[:, 0].to_numpy().tolist()  # x-coordinates
    y_coords = df_events.iloc[:, 1].to_numpy().tolist()  # y-coordinates
    polarities = df_events.iloc[:, 2].to_numpy().tolist()  # polarities 
    
    # Normalize so each series starts at 0 to match the clip and the timestamps
    timestamps1 = df_events.iloc[:, 3].to_numpy() # timestamps for events
    timestamps = (timestamps1 - timestamps1[0]).tolist() # normalized event timestamps
    time_windows1 = df_timestamps.iloc[:, 1].to_numpy() # timestamps of last event in a frame
    time_windows = time_windows1 - time_windows1[0] # normalized frame timestamps

    print("First 10 normalized event timestamps:", timestamps[:10])
    print("First 10 normalized time windows:", time_windows[:10])

    for idx, t in enumerate(timestamps[:10]):
        found = False
        for i in range(len(time_windows) - 1):
            if time_windows[i] <= t < time_windows[i + 1]:
                print(f"Event {idx} (t={t}) falls into window [{time_windows[i]}, {time_windows[i+1]}) (index {i})")
                found = True
                break
        if not found:
            print(f"Event {idx} (t={t}) does NOT fall into any window!")