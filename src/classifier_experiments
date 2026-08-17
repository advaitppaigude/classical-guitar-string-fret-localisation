#classifier_experiments.py

import numpy as np
import pandas as pd
import math

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


notes = ["E4", "D3", "E3", "F4", "G3"]

positions =[[{"S": 1, "F": 0}, {"S": 2, "F": 5}, {"S": 3, "F": 9}],
            [{"S": 4, "F": 0}, {"S": 5, "F": 5}, {"S": 6, "F": 10}],
            [{"S": 4, "F": 2}, {"S": 5, "F": 7}, {"S": 6, "F": 12}],
            [{"S": 1, "F": 1}, {"S": 2, "F": 6}, {"S": 3, "F": 10}],
            [{"S": 3, "F": 0}, {"S": 4, "F": 5}, {"S": 5, "F": 10}]
            ]



feature_columns = [
    "h2_h1",
    "h3_h1",
    "h4_h1",
    "h5_h1",
    "log_h3_h2",
    "spectral_centroid",
    "harmonic_centroid",
    "inharmonicity",
]



num_correct_per_note = [0,0,0,0,0]


def log_h3_h2(h3_amp, h2_amp):
    epsilon = 1e-6
    return math.log((h3_amp + epsilon) / (h2_amp + epsilon))

def euclidean_distance(a, b):
    distance = 0
    for i in range(0, len(a)):
        distance += ((a[i] - b[i]) ** 2)
    return distance ** 0.5


def generate_filename_from_numbers(note_num, recording_num):
    string = positions[note_num][recording_num // 10]["S"]
    fret = positions[note_num][recording_num // 10]["F"]

    take_number = (recording_num % 10) + 1
    take = str(take_number)
    if take_number < 10:
        take = f"0{take}"

    filename = f"{notes[note_num]}_S{string}_F{fret}_{take}.wav"
    return filename



# driver code


features = pd.read_csv("features.csv")
features["filename"] = features["filename"].astype(str).str.strip()



# ITERATION 1 - k-NN
k = 3

def iteration_1():
    for note_num in range(0, len(notes)):
        for test_recording_num in range(0,30):
            test_filename = generate_filename_from_numbers(note_num, test_recording_num)
            # capture features from the csv row to form a "vector"
            row = features[features["filename"] == test_filename].iloc[0]
            test_vector = row[feature_columns].to_numpy(dtype=float)

            distances = []
            string_fret_positions = []

            training_vectors = []
            
            for recording_num in range(30):
                if recording_num != test_recording_num:
                    training_filename = generate_filename_from_numbers(
                        note_num, recording_num
                    )

                    row = features[
                        features["filename"] == training_filename
                    ].iloc[0]

                    vector = row[feature_columns].to_numpy(dtype=float)

                    string = positions[note_num][recording_num // 10]["S"]
                    fret = positions[note_num][recording_num // 10]["F"]

                    training_data.append({
                        "vector": vector,
                        "position": f"S{string}_F{fret}"
                    })

            training_vectors = np.array(training_vectors)

            # standardising data
            means = np.mean(training_vectors, axis = 0)
            stds = np.std(training_vectors, axis = 0)
            standardised_training_vectors = (training_vectors - means) / stds
            standardised_test_vector = (test_vector - means) / stds

            distances = []
            string_fret_positions = []

            for recording_num in range(len(training_data)):
                distance = euclidean_distance(standardised_training_vectors[recording_num], standardised_test_vector)

                distances.append(distance)
                string_fret_positions.append(training_data[recording_num]["position"]                )


            neighbours = list(zip(distances, string_fret_positions))
            neighbours.sort(key=lambda x: x[0])

            # evaluating how effective it is
            correct_nearest_neighbours = 0

            for _, position in neighbours[:k]:
                if position in test_filename:
                    correct_nearest_neighbours += 1

            if correct_nearest_neighbours > k // 2: # majority vote (no edge case with odd k)
                num_correct_per_note[note_num] += 1


# ITERATION 2 - rank-weighted position scoring
def iteration_2():
    for note_num in range(0, len(notes)):
        for test_recording_num in range(0,30):
            test_filename = generate_filename_from_numbers(note_num, test_recording_num)
            # capture features from the csv row to form a "vector"
            row = features[features["filename"] == test_filename].iloc[0]
            test_vector = row[feature_columns].to_numpy(dtype=float)


            training_data = []
            
            for recording_num in range(0, 30):
                if recording_num != test_recording_num:
                    # capture features from the csv row to form a "vector"
                    training_filename = generate_filename_from_numbers(note_num, recording_num)
                    row = features[features["filename"] == training_filename].iloc[0]

                    string = positions[note_num][recording_num // 10]["S"]
                    fret = positions[note_num][recording_num // 10]["F"]
                    vector = row[feature_columns].to_numpy(dtype=float)
                    training_data.append({"vector": vector, "position": f"S{string}_F{fret}"})

            training_vectors = np.array([item["vector"] for item in training_data])
            
            # standardising data
            means = np.mean(training_vectors, axis = 0)
            stds = np.std(training_vectors, axis = 0)
            standardised_training_vectors = (training_vectors - means) / stds
            standardised_test_vector = (test_vector - means) / stds


            neighbours = []

            for training_recording_num in range(0, len(training_data)):
                distance = euclidean_distance(standardised_training_vectors[training_recording_num], standardised_test_vector)
                neighbours.append({"distance": distance, "position": training_data[training_recording_num]["position"]})


            neighbours.sort(key = lambda x: x["distance"])

            string_fret_position_scores = {}
            string_fret_position_counts = {}
            for i in range(0, len(neighbours)):
                if neighbours[i]["position"] not in string_fret_position_scores:
                    string_fret_position_scores[neighbours[i]["position"]] = 0
                    string_fret_position_counts[neighbours[i]["position"]] = 0

                string_fret_position_scores[neighbours[i]["position"]] += len(neighbours) - i
                string_fret_position_counts[neighbours[i]["position"]] += 1
            
            #print(f"Scores after {string_fret_position_scores}")
            for position in string_fret_position_scores:
                string_fret_position_scores[position] /= string_fret_position_counts[position]

            # scoring positions based on list
            highest_score = -np.inf
            estimated_position = ""

            for position, score in string_fret_position_scores.items():
                if score > highest_score:
                    highest_score = score
                    estimated_position = position

            # collecting evaluation data
            #print(test_filename, "->", estimated_position, string_fret_position_scores)
            if estimated_position in test_filename:
                num_correct_per_note[note_num] += 1


# ITERATION 3 - distance-weighted position scoring
def iteration_3():
    for note_num in range(0, len(notes)):
        for test_recording_num in range(0,30):
            test_filename = generate_filename_from_numbers(note_num, test_recording_num)
            # capture features from the csv row to form a "vector"
            row = features[features["filename"] == test_filename].iloc[0]
            test_vector = row[feature_columns].to_numpy(dtype=float)


            training_data = []
            
            for recording_num in range(0, 30):
                if recording_num != test_recording_num:
                    # capture features from the csv row to form a "vector"
                    training_filename = generate_filename_from_numbers(note_num, recording_num)
                    row = features[features["filename"] == training_filename].iloc[0]

                    string = positions[note_num][recording_num // 10]["S"]
                    fret = positions[note_num][recording_num // 10]["F"]
                    vector = row[feature_columns].to_numpy(dtype=float)
                    training_data.append({"vector": vector, "position": f"S{string}_F{fret}"})

            training_vectors = np.array([item["vector"] for item in training_data])
            
            # standardising data
            means = np.mean(training_vectors, axis = 0)
            stds = np.std(training_vectors, axis = 0)
            standardised_training_vectors = (training_vectors - means) / stds
            standardised_test_vector = (test_vector - means) / stds


            # calculating and storign distances between test recording and training recordings
            neighbours = []
            for training_recording_num in range(0, len(training_data)):
                distance = euclidean_distance(standardised_training_vectors[training_recording_num], standardised_test_vector)
                neighbours.append({"distance": distance, "position": training_data[training_recording_num]["position"]})


            neighbours.sort(key = lambda x: x["distance"])

            string_fret_position_scores = {}
            string_fret_position_counts = {}
            for i in range(0, len(neighbours)):
                if neighbours[i]["position"] not in string_fret_position_scores:
                    string_fret_position_scores[neighbours[i]["position"]] = 0
                    string_fret_position_counts[neighbours[i]["position"]] = 0

                string_fret_position_scores[neighbours[i]["position"]] += distance_weighting(neighbours[i]["distance"])
                string_fret_position_counts[neighbours[i]["position"]] += 1
            
            #print(f"Scores after {string_fret_position_scores}")
            for position in string_fret_position_scores:
                string_fret_position_scores[position] /= string_fret_position_counts[position]

            # scoring positions based on list
            highest_score = -np.inf
            estimated_position = ""

            for position, score in string_fret_position_scores.items():
                if score > highest_score:
                    highest_score = score
                    estimated_position = position

            # collecting evaluation data
            #print(test_filename, "->", estimated_position, string_fret_position_scores)
            if estimated_position in test_filename:
                num_correct_per_note[note_num] += 1


# ITERATION 4 - logistic regression
def iteration_4():
    positions = {
        "E4": [{"S": 1, "F": 0}, {"S": 2, "F": 5}, {"S": 3, "F":  9}],
        "D3": [{"S": 4, "F": 0}, {"S": 5, "F": 5}, {"S": 6, "F": 10}],
        "E3": [{"S": 4, "F": 2}, {"S": 5, "F": 7}, {"S": 6, "F": 12}],
        "F4": [{"S": 1, "F": 1}, {"S": 2, "F": 6}, {"S": 3, "F": 10}],
        "G3": [{"S": 3, "F": 0}, {"S": 4, "F": 5}, {"S": 5, "F": 10}]
    }
    for note_num in range(0, len(notes)):
        for test_index in range(0,30):
            X = []
            y = []

            for recording_num in range(30):
                filename = generate_filename_from_numbers(note_num, recording_num)

                row = features[features["filename"] == filename].iloc[0]

                X.append(row[feature_columns].to_numpy(dtype=float))

                string = positions[notes[note_num]][recording_num // 10]["S"]
                fret = positions[notes[note_num]][recording_num // 10]["F"]

                y.append(f"S{string}_F{fret}")

            X = np.array(X)
            y = np.array(y)

            # training model on all other points
            X_train = np.delete(X, test_index, axis=0)
            y_train = np.delete(y, test_index)

            X_test = X[test_index].reshape(1, -1)

            scaler = StandardScaler()

            X_train_std = scaler.fit_transform(X_train)
            X_test_std = scaler.transform(X_test)

            classifier = LogisticRegression()

            classifier.fit(X_train_std, y_train)

            prediction = classifier.predict(X_test_std)[0]

            if prediction == y[test_index]:
                num_correct_per_note[note_num] += 1


# ITERATION 5 - SVM
def iteration_5():
    positions = {
        "E4": [{"S": 1, "F": 0}, {"S": 2, "F": 5}, {"S": 3, "F":  9}],
        "D3": [{"S": 4, "F": 0}, {"S": 5, "F": 5}, {"S": 6, "F": 10}],
        "E3": [{"S": 4, "F": 2}, {"S": 5, "F": 7}, {"S": 6, "F": 12}],
        "F4": [{"S": 1, "F": 1}, {"S": 2, "F": 6}, {"S": 3, "F": 10}],
        "G3": [{"S": 3, "F": 0}, {"S": 4, "F": 5}, {"S": 5, "F": 10}]
    }
    for note_num in range(0, len(notes)):
        for test_index in range(0,30):
            X = []
            y = []

            for recording_num in range(30):
                filename = generate_filename_from_numbers(note_num, recording_num)

                row = features[features["filename"] == filename].iloc[0]

                X.append(row[feature_columns].to_numpy(dtype=float))

                string = positions[notes[note_num]][recording_num // 10]["S"]
                fret = positions[notes[note_num]][recording_num // 10]["F"]

                y.append(f"S{string}_F{fret}")

            X = np.array(X)
            y = np.array(y)

            # training model on all other points
            X_train = np.delete(X, test_index, axis=0)
            y_train = np.delete(y, test_index)

            X_test = X[test_index].reshape(1, -1)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            svm_classifier = SVC(kernel='linear')
            svm_classifier.fit(X_train_scaled, y_train)
            [prediction] = svm_classifier.predict(X_test_scaled)
            if prediction == y[test_index]:
                num_correct_per_note[note_num] += 1



total = 0
for i in range(0,5):
    print(f"For note {notes[i]}, {num_correct_per_note[i]}/30 classifications were correct")
    total += num_correct_per_note[i]
print(f"On average, {total / 5}/30 classifications were correct")
