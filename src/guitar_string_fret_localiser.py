# guitar_string_fret_localiser.py

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

if __package__:
    from .signal_processing import (
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
else:
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

FEATURES_CSV = Path(__file__).resolve().parents[1] / "data" / "features.csv"

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




def load_dataset_features(features_csv=FEATURES_CSV):
    """Load the existing feature table without changing its values or row order."""
    dataset_features = pd.read_csv(features_csv)
    dataset_features["filename"] = dataset_features["filename"].astype(str).str.strip()
    return dataset_features


def extract_recording_features(wav_path):
    """Return (detected note, raw eight-feature vector) using the original DSP."""
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

    return note_name, input_vector


def train_note_classifier(note_name, dataset_features=None):
    """Fit and return the original StandardScaler/LogisticRegression pipeline.

    Select exactly takes 01–10 at each of the note's three original positions,
    in the original order (30 rows). No evaluation split is introduced here.
    Supply a loaded feature DataFrame to reuse it when training several notes.
    """
    if note_name not in notes:
        raise ValueError(f"Unsupported note {note_name}; choose from {', '.join(notes)}")
    if dataset_features is None:
        dataset_features = load_dataset_features()
    else:
        dataset_features = dataset_features.copy()
        dataset_features["filename"] = dataset_features["filename"].astype(str).str.strip()

    X = []
    y = []

    # generating points from training data
    for recording_num in range(30):
        filename = generate_filename_from_numbers(note_name, recording_num)

        row = dataset_features[dataset_features["filename"] == filename].iloc[0]

        X.append(row[feature_columns].to_numpy(dtype=float))

        string = positions[note_name][recording_num // 10]["S"]
        fret = positions[note_name][recording_num // 10]["F"]

        y.append(f"S{string}_F{fret}")

    X_train = np.array(X)
    y_train = np.array(y)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression()),
    ])
    model.fit(X_train, y_train)
    return model


def predict_features(model, input_features):
    """Return (labels, probabilities) without fitting or updating the model.

    Accept one raw feature vector or a batch, ordered by feature_columns.
    Both outputs retain their batch dimension; probability columns follow
    model.classes_. Reuse the same fitted model for subsequent calls.
    """
    vectors = np.atleast_2d(np.asarray(input_features, dtype=float))
    return model.predict(vectors), model.predict_proba(vectors)


def main():
    # Keep the existing positional WAV argument and printed result format.
    if len(sys.argv) != 2:
        raise ValueError(
            "Usage: python guitar_string_fret_localiser.py <recording.wav>"
        )

    dataset_features = load_dataset_features()
    note_name, input_vector = extract_recording_features(sys.argv[1])
    model = train_note_classifier(note_name, dataset_features)
    predictions, all_probabilities = predict_features(model, input_vector)
    prediction = predictions[0]
    probabilities = all_probabilities[0]
    classes = model.classes_

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
