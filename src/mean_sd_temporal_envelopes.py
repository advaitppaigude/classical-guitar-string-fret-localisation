# mean_sd_temporal_envelopes.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io
from scipy.io import wavfile
from pathlib import Path

note_name = "E4"
positions = [{"S": 1, "F": 0}, {"S": 2, "F": 5}, {"S": 3, "F": 9}]

# RMS envelope functions
def get_normalised_waveform(filename, start_time, end_time):
    rate, data = wavfile.read(filename)
    start_sample = int(start_time * rate)
    end_sample = int(end_time * rate)
    aligned_data = data[start_sample:]
    max_amplitude = np.max(np.abs(aligned_data))
    normalised_data = np.divide(aligned_data, max_amplitude)
    return rate, normalised_data

def get_RMS_envelope(rate, data):
    RMS_data = []
    RMS_times = []
    window_samples = int(0.035 * rate)
    hop_samples = int(0.010 * rate)
    start_sample_index = 0
    while start_sample_index + window_samples < len(data):
        temp_RMS = np.sqrt(np.mean(np.square(data[start_sample_index:start_sample_index + window_samples])))
        RMS_data.append(temp_RMS)
        RMS_times.append((start_sample_index + window_samples/2)/rate)
        start_sample_index += hop_samples

    return RMS_times, RMS_data

def get_normalised_RMS_envelope(filename, start_time, end_time):
    rate, data = get_normalised_waveform(filename, start_time, end_time)
    times, RMS_data = get_RMS_envelope(rate, data)
    max_amplitude = np.max(np.abs(RMS_data))
    normalised_data = np.divide(RMS_data, max_amplitude)
    return times, normalised_data


fig, ax = plt.subplots(figsize=(14, 5))



current_position = 0

# plotting functions
def draw_current_file():

    metadata = pd.read_csv("metadata.csv")
    metadata["filename"] = metadata["filename"].astype(str).str.strip()

    ax.clear()

    all_envelopes = []

    for take in range(1, 11):
        # name formatting
        if take != 10:
            filename = f"{note_name}_S{positions[current_position]['S']}_F{positions[current_position]['F']}_0{take}.wav"
        else:
            filename = f"{note_name}_S{positions[current_position]['S']}_F{positions[current_position]['F']}_{take}.wav"
        
        row = metadata[metadata["filename"] == filename].iloc[0]
        start_time = row["onset_time"]
        end_time = row["analysis_end"]
        
        times, data = get_normalised_RMS_envelope(f"Recordings/{filename}", start_time, end_time)
        all_envelopes.append(data)

    ax.set_title(f"{note_name}_S{positions[current_position]['S']}_F{positions[current_position]['F']} RMS envelope")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normalised amplitude")
    ax.legend()

    min_length = min(len(envelope) for envelope in all_envelopes)

    all_envelopes = [
        envelope[:min_length]
        for envelope in all_envelopes
    ]

    envelopes = np.array(all_envelopes)
    mean_envelope = np.mean(envelopes, axis=0)
    std_envelope = np.std(envelopes, axis=0)
    times = np.array(times[:min_length])

    ax.plot(times, mean_envelope, label="Mean RMS envelope")

    # shading the +- 1 standard deviation region
    ax.fill_between(
        times,
        mean_envelope - std_envelope,
        mean_envelope + std_envelope,
        alpha=0.2,
        label="±1 SD"
    )
    ax.legend()
    fig.canvas.draw_idle()



# cycling through different positions
def on_key(event):
    global current_position

    if event.key == "right":
        current_position = (current_position + 1) % 3
        draw_current_file()

    elif event.key == "left":
        current_position = (current_position - 1) % 3
        draw_current_file()

    elif event.key == "q":
        plt.close(fig)


# driver code
fig.canvas.mpl_connect("key_press_event", on_key)

draw_current_file()
plt.show()
