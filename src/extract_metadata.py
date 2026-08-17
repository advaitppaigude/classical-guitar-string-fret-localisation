# extract_metadata.py

"""
Header structure:
filename, note, string, fret, take, onset_time, rms_peak_time, analysis_start_time, analysis_end_time

"""

from pathlib import Path
import re
import csv

import numpy as np
from scipy.io import wavfile

from signal_processing import *


# configuration

WAV_FOLDER = Path("Recordings")
OUTPUT_CSV = Path("metadata.csv")

RMS_WINDOW_S = 0.03 # 30ms
RMS_HOP_S = 0.005 # 5ms

ANALYSIS_OFFSET_FROM_RMS_PEAK_S = 0.25
ANALYSIS_DURATION_S = 0.5



def parse_filename(filename):
    """
    Expected format:
        E4_S2_F5_01.wav
        D3_S4_F0_02.wav
        F4_S1_F1_10.wav
    """

    pattern = r"^([A-G][#b]?\d)_S(\d)_F(\d+)_(\d+)\.wav$"
    match = re.match(pattern, filename)

    if match is None:
        raise ValueError(f"Filename does not match expected format: {filename}")

    note = match.group(1)
    string = int(match.group(2))
    fret = int(match.group(3))
    take = int(match.group(4))

    return note, string, fret, take


# main batch processing
rows = []

wav_files = sorted(WAV_FOLDER.glob("*.wav"))

def main():
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No WAV files found in {WAV_FOLDER.resolve()}")

    for wav_path in wav_files:
        try:
            note, string, fret, take = parse_filename(wav_path.name)

            rate, data = get_normalised_waveform(wav_path)
            duration = len(data) / rate

            rms_times, rms_values = get_rms_envelope(
                data,
                rate,
                window_s=RMS_WINDOW_S,
                hop_s=RMS_HOP_S
            )

            onset_time = find_onset_time(rms_times, rms_values)
            rms_peak_time, rms_peak_value = find_rms_peak_and_time(rms_times, rms_values)

            analysis_start = rms_peak_time + ANALYSIS_OFFSET_FROM_RMS_PEAK_S
            analysis_end = analysis_start + ANALYSIS_DURATION_S

            rows.append({
                "filename": wav_path.name,
                "note": note,
                "string": string,
                "fret": fret,
                "take": take,
                "onset_time": onset_time,
                "rms_peak_time": rms_peak_time,
                "analysis_start_time": analysis_start,
                "analysis_end_time": analysis_end
            })

            print(f"Processed {wav_path.name}")

        except Exception as e:
            rows.append({
                "filename": wav_path.name,
                "note": "",
                "string": "",
                "fret": "",
                "take": "",
                "onset_time": "",
                "rms_peak_time": "",
                "analysis_start_time": "",
                "analysis_end_time": ""
            })

            print(f"ERROR processing {wav_path.name}: {e}")


    # saving CSV
    fieldnames = [
        "filename",
        "note",
        "string",
        "fret",
        "take",
        "onset_time",
        "rms_peak_time",
        "analysis_start_time",
        "analysis_end_time",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved metadata for {len(rows)} files to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
