# extract_features.py

"""
metadata.csv:
filename,note,string,fret,take,onset_time,rms_peak_time,analysis_start_time,analysis_end_time

features.csv:
filename	note	string	fret	take	f0_ideal	h1_freq	h2_freq	h3_freq	h4_freq	h5_freq	h1_amp	h2_amp	h3_amp	h4_amp	h5_amp	h2_h1	h3_h1	h4_h1	h5_h1	h3_h2	harmonic_centroid	spectral_centroid	rolloff_85	mean_relative_harmonic_deviation	pitch_error_hz	pitch_error_cents	tuning_ok

"""

from pathlib import Path
import re
import csv

import numpy as np
import math
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann

from signal_processing import *

NOTE_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


def note_to_ideal_frequency(note):
    # converts note name like E4, F#3, Bb2 to equal-tempered frequency.
    # A4 = 440 Hz

    if len(note) == 2:
        note_name = note[0]
        octave = int(note[1])
    else:
        note_name = note[:2]
        octave = int(note[2])

    semitone = NOTE_TO_SEMITONE[note_name]

    midi_number = 12 * (octave + 1) + semitone

    frequency = 440 * (2 ** ((midi_number - 69) / 12))

    return frequency



# file handling

WAV_FOLDER = Path("Recordings")
METADATA_CSV = Path("metadata.csv")
OUTPUT_CSV = Path("features.csv")

BIN_THRESHOLD_HZ = 10



fieldnames = [
    "filename",
    "note",
    "string",
    "fret",
    "take",

    "f0_ideal",

    "h1_freq",
    "h2_freq",
    "h3_freq",
    "h4_freq",
    "h5_freq",

    "h1_amp",
    "h2_amp",
    "h3_amp",
    "h4_amp",
    "h5_amp",

    "h2_h1",
    "h3_h1",
    "h4_h1",
    "h5_h1",
    "log_h3_h2",

    "harmonic_centroid",
    "spectral_centroid",
    "mean_relative_harmonic_deviation",

    "pitch_error_hz",
    "pitch_error_cents",
    "tuning_ok"
]


# filename parsing
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



# driver code

metadata = pd.read_csv(METADATA_CSV)
metadata["filename"] = metadata["filename"].astype(str).str.strip()

# lookup by filename
metadata_by_filename = metadata.set_index("filename")

rows = []

wav_files = sorted(WAV_FOLDER.glob("*.wav"))



def main():
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No WAV files found in {WAV_FOLDER.resolve()}")

    # iterating through all recordings
    for wav_path in wav_files:
        try:
            filename = wav_path.name

            # Labels from filename
            note, string, fret, take = parse_filename(filename)

            # Analysis window from metadata
            if filename not in metadata_by_filename.index:
                raise ValueError(f"No metadata row found for {filename}")

            metadata_row = metadata_by_filename.loc[filename]

            analysis_start_time = float(metadata_row["analysis_start_time"])
            analysis_end_time = float(metadata_row["analysis_end_time"])

            rate, normalised_data = get_normalised_waveform(wav_path)

            frequency_axis, fft_magnitude = get_fft(
                rate,
                normalised_data,
                analysis_start_time,
                analysis_end_time
            )

            f0_ideal = note_to_ideal_frequency(note)

            fundamental_frequency, fundamental_index = find_h1_near_ideal(frequency_axis, fft_magnitude, f0_ideal)

            h_freq, h_amp = find_harmonic_peaks(
                frequency_axis,
                fft_magnitude,
                fundamental_frequency,
                max_harmonic=5
            )


            harmonic_centroid = calculate_harmonic_centroid(h_freq, h_amp)
            spectral_centroid = calculate_spectral_centroid(frequency_axis, fft_magnitude)
            rolloff_85 = calculate_rolloff(frequency_axis, fft_magnitude, 0.85)
            mean_relative_harmonic_deviation = calculate_mean_relative_harmonic_deviation(h_freq, h_freq[1])

            pitch_error_hz = h_freq[1] - f0_ideal
            pitch_error_cents = 1200 * np.log2(h_freq[1] / f0_ideal)
            tuning_ok = abs(pitch_error_cents) <= 25

            rows.append({
                "filename": filename,
                "note": note,
                "string": string,
                "fret": fret,
                "take": take,

                "f0_ideal": f0_ideal,

                "h1_freq": h_freq[1],
                "h2_freq": h_freq[2],
                "h3_freq": h_freq[3],
                "h4_freq": h_freq[4],
                "h5_freq": h_freq[5],

                "h1_amp": h_amp[1],
                "h2_amp": h_amp[2],
                "h3_amp": h_amp[3],
                "h4_amp": h_amp[4],
                "h5_amp": h_amp[5],

                "h2_h1": safe_divide(h_amp[2], h_amp[1]),
                "h3_h1": safe_divide(h_amp[3], h_amp[1]),
                "h4_h1": safe_divide(h_amp[4], h_amp[1]),
                "h5_h1": safe_divide(h_amp[5], h_amp[1]),
                "log_h3_h2": log_h3_h2(h_amp[3], h_amp[2]),

                "harmonic_centroid": harmonic_centroid,
                "spectral_centroid": spectral_centroid,
                "rolloff_85": rolloff_85,
                "mean_relative_harmonic_deviation": mean_relative_harmonic_deviation,

                "pitch_error_hz": pitch_error_hz,
                "pitch_error_cents": pitch_error_cents,
                "tuning_ok": tuning_ok
            })

            print(f"Processed {filename}")

        except Exception as e:
            print(f"ERROR processing {wav_path.name}: {e}")

            rows.append({
                "filename": wav_path.name,
                "note": "",
                "string": "",
                "fret": "",
                "take": "",
                "f0_ideal": "",
                "h1_freq": "",
                "h2_freq": "",
                "h3_freq": "",
                "h4_freq": "",
                "h5_freq": "",
                "h1_amp": "",
                "h2_amp": "",
                "h3_amp": "",
                "h4_amp": "",
                "h5_amp": "",
                "h2_h1": "",
                "h3_h1": "",
                "h4_h1": "",
                "h5_h1": "",
                "log_h3_h2": "",
                "harmonic_centroid": "",
                "spectral_centroid": "",
                "rolloff_85": "",
                "mean_relative_harmonic_deviation": "",
                "pitch_error_hz": "",
                "pitch_error_cents": "",
                "tuning_ok": ""
            })


    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved features to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
