# guitar_string_fret_localiser.py

import math
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from signal_processing import (
    get_normalised_waveform,
    get_rms_envelope,
    find_onset_time,
    find_rms_peak_and_time,
    get_fft,
    find_fundamental,
    find_harmonic_peaks,
    calculate_harmonic_centroid,
    calculate_spectral_centroid,
    calculate_mean_relative_harmonic_deviation,
    safe_divide,
    log_h3_h2,
)



notes = ["E4", "D3", "E3", "F4", "G3"]

midi_to_note = {
    40: "E2", 41: "F2", 42: "F#2", 43: "G2", 44: "G#2",
    45: "A2", 46: "A#2", 47: "B2", 48: "C3", 49: "C#3",
    50: "D3", 51: "D#3", 52: "E3", 53: "F3", 54: "F#3",
    55: "G3", 56: "G#3", 57: "A3", 58: "A#3", 59: "B3",
    60: "C4", 61: "C#4", 62: "D4", 63: "D#4", 64: "E4",
    65: "F4", 66: "F#4", 67: "G4", 68: "G#4", 69: "A4",
    70: "A#4", 71: "B4", 72: "C5", 73: "C#5", 74: "D5",
    75: "D#5", 76: "E5"
}

positions = {
    "E4": [{"S": 1, "F": 0}, {"S": 2, "F": 5}, {"S": 3, "F":  9}],
    "D3": [{"S": 4, "F": 0}, {"S": 5, "F": 5}, {"S": 6, "F": 10}],
    "E3": [{"S": 4, "F": 2}, {"S": 5, "F": 7}, {"S": 6, "F": 12}],
    "F4": [{"S": 1, "F": 1}, {"S": 2, "F": 6}, {"S": 3, "F": 10}],
    "G3": [{"S": 3, "F": 0}, {"S": 4, "F": 5}, {"S": 5, "F": 10}]
}

feature_columns = [
    "h2_h1",
    "h3_h1",
    "h4_h1",
    "h5_h1",
    "log_h3_h2",
    "spectral_centroid",
    "harmonic_centroid",
    "mean_relative_harmonic_deviation"
]


RMS_WINDOW_S = 0.03 # 30ms
RMS_HOP_S = 0.005 # 5ms

ANALYSIS_OFFSET_FROM_RMS_PEAK_S = 0.25
ANALYSIS_DURATION_S = 0.5


# helper functions
def frequency_to_midi(frequency):
    return round(12 * math.log(frequency/440, 2) + 69)

def generate_filename_from_numbers(note_name, recording_num):
    string = positions[note_name][recording_num // 10]["S"]
    fret = positions[note_name][recording_num // 10]["F"]

    take_number = (recording_num % 10) + 1
    take = str(take_number)
    if take_number < 10:
        take = f"0{take}"

    filename = f"{note_name}_S{string}_F{fret}_{take}.wav"
    return filename




def main():

    # allowing wav_path to be passed in terminal
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python guitar_string_fret_localiser.py <recording.wav>"
        )

    wav_path = sys.argv[1]

    dataset_features = pd.read_csv("features.csv")
    dataset_features["filename"] = dataset_features["filename"].astype(str).str.strip()

    rate, normalised_data = get_normalised_waveform(wav_path)

    rms_times, rms_values = get_rms_envelope(
        normalised_data,
        rate,
        window_s=RMS_WINDOW_S,
        hop_s=RMS_HOP_S
    )

    rms_peak_time, rms_peak_value = find_rms_peak_and_time(rms_times, rms_values)

    analysis_start_time = rms_peak_time + ANALYSIS_OFFSET_FROM_RMS_PEAK_S
    analysis_end_time = analysis_start_time + ANALYSIS_DURATION_S

    frequency_axis, fft_magnitude = get_fft(
        rate,
        normalised_data,
        analysis_start_time,
        analysis_end_time
    )


    # calculating the necessary feature values
    fundamental_frequency = find_fundamental(frequency_axis, fft_magnitude)
    note_name = midi_to_note[frequency_to_midi(fundamental_frequency)]
    if note_name not in notes:
        raise ValueError(f"Detected note {note_name}, but this proof-of-concept supports only: {', '.join(notes)}")

    note_num = notes.index(note_name)
    h_freq, h_amp = find_harmonic_peaks(frequency_axis, fft_magnitude, fundamental_frequency, max_harmonic=5)
    harmonic_centroid = calculate_harmonic_centroid(h_freq, h_amp)
    spectral_centroid = calculate_spectral_centroid(frequency_axis, fft_magnitude)
    mean_relative_harmonic_deviation = calculate_mean_relative_harmonic_deviation(h_freq, h_freq[1])


    input_vector = np.array([
        safe_divide(h_amp[2], h_amp[1]),
        safe_divide(h_amp[3], h_amp[1]),
        safe_divide(h_amp[4], h_amp[1]),
        safe_divide(h_amp[5], h_amp[1]),
        log_h3_h2(h_amp[3], h_amp[2]),
        spectral_centroid,
        harmonic_centroid,
        mean_relative_harmonic_deviation
        ], dtype=float)



    X = []
    y = []

    # generating points from training data
    for recording_num in range(30):
        filename = generate_filename_from_numbers(notes[note_num], recording_num)

        row = dataset_features[dataset_features["filename"] == filename].iloc[0]

        X.append(row[feature_columns].to_numpy(dtype=float))

        string = positions[notes[note_num]][recording_num // 10]["S"]
        fret = positions[notes[note_num]][recording_num // 10]["F"]

        y.append(f"S{string}_F{fret}")

    X_train = np.array(X)
    y_train = np.array(y)

    X_test = input_vector.reshape(1, -1)

    # running the model
    scaler = StandardScaler()

    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    classifier = LogisticRegression()

    classifier.fit(X_train_std, y_train)

    # printing results
    prediction = classifier.predict(X_test_std)[0]

    probabilities = classifier.predict_proba(X_test_std)[0]
    classes = classifier.classes_

    predicted_index = np.argmax(probabilities)
    predicted_probability = probabilities[predicted_index]

    sorted_indices = np.argsort(probabilities)[::-1] # sorting descending

    second_index = sorted_indices[1]
    second_class = classes[second_index]
    second_probability = probabilities[second_index]


    first_string = prediction.split("_")[0][1:]
    first_fret = prediction.split("_")[1][1:]
    second_string = second_class.split("_")[0][1:]
    second_fret = second_class.split("_")[1][1:]

    print(f"Detected note: {note_name}")
    print(f"Predicted position: String {first_string} Fret {first_fret}")
    print(f"Model probability: {predicted_probability:.1%}")
    print(f"Next-best position: String {second_string} Fret {second_fret} ({second_probability:.1%})")
    print()

if __name__ == "__main__":
    main()
