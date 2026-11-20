"""
modelling the arithmetic that will be implemented in hardware


quantise() - float -> signed fixed-point int

dequantise() - signed fixed-point int -> float

fixed_point_inference() - calculate class scores using ints

"""

import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path("fpga/results/numerical_exploration")
results_dir.mkdir(parents=True, exist_ok=True)

agreement_results = {}
weight_clipping_results = {}
feature_clipping_results = {}

def quantise(value, total_bits, frac_bits):
    q = int(round(value * (2 ** frac_bits)))

    # range saturation
    min = (-2 ** (total_bits - 1))
    max = ((2** (total_bits - 1)) - 1)
    if (q < min):
        return min
    elif (q > max):
        return max

    return q


def dequantise(value, frac_bits):
    return value / (2** (frac_bits))


def fixed_point_inference(quantised_features, quantised_weights, quantised_biases, acc_bits):
    scores = []
    predicted_class_index = 0

    for i in range(len(quantised_weights)):
        current = quantised_biases[i]

        """
        iterating backwards over features
        because fsm uses down-counter rather than up-counter

        due to saturation behaviour, iterating in ascending order
        may produce different results
        """
        for j in range(len(quantised_features) - 1, -1, -1):
            current += quantised_features[j] * quantised_weights[i][j]

            current = max(
                -(2 ** (acc_bits - 1)),
                min(current, (2 ** (acc_bits - 1)) - 1)
            )

        scores.append(current)

    for i in range(1, len(scores)):
        if scores[i] > scores[predicted_class_index]:
            predicted_class_index = i

    return (scores, predicted_class_index)



df = pd.read_csv("data/features.csv")


feature_bits_sweep = [1, 2, 4, 4, 5, 6, 8, 12, 16, 24]
feature_frac_sweep = [0, 1, 1, 2, 2, 3, 4, 8, 10, 16]

notes = ["E4", "D3", "E3", "F4", "G3"]

for note in notes:
    agreement_results[note] = []
    weight_clipping_results[note] = []
    feature_clipping_results[note] = []

    print(f"{note}:")
    model = np.load(f"fpga/models/{note}.npz")
    feature_columns = model["feature_columns"]


    weights = model["weights"]
    biases = model["biases"]
    means = model["means"]
    scales = model["scales"]
    classes = model["classes"]


    # Select all valid note recordings
    current_note_data = df[df["note"] == note].dropna(
        subset=list(feature_columns)
    )

    X_test = current_note_data[list(feature_columns)].to_numpy(dtype=float)



    print("Feature range:", np.min((X_test - means) / scales),
        np.max((X_test - means) / scales))

    print("Weight range:", weights.min(), weights.max())
    print("Bias range:", biases.min(), biases.max())

    for i in range(0, len(feature_bits_sweep)):

        FEATURE_BITS = feature_bits_sweep[i]
        FEATURE_FRAC = feature_frac_sweep[i]

        WEIGHT_BITS = feature_bits_sweep[i]
        WEIGHT_FRAC = feature_frac_sweep[i]

        PRODUCT_BITS = FEATURE_BITS + WEIGHT_BITS
        PRODUCT_FRAC = FEATURE_FRAC + WEIGHT_FRAC
        ACC_BITS = PRODUCT_BITS + 4

        quantised_biases = [quantise(b, ACC_BITS, PRODUCT_FRAC) for b in biases]
        quantised_weights = [[quantise(w, WEIGHT_BITS, WEIGHT_FRAC) for w in row] for row in weights]

        # int limits
        weight_min = -(2 ** (WEIGHT_BITS - 1))
        weight_max = (2 ** (WEIGHT_BITS - 1)) - 1

        feature_min = -(2 ** (FEATURE_BITS - 1))
        feature_max = (2 ** (FEATURE_BITS - 1)) - 1

        # checking weight clipping
        scaled_weights = np.rint(weights * (2 ** WEIGHT_FRAC))
        clipped_weights = np.sum(
            (scaled_weights < weight_min) |
            (scaled_weights > weight_max)
        )


        # checking feature clipping
        X_std = (X_test - means) / scales
        scaled_features = np.rint(X_std * (2 ** FEATURE_FRAC))

        clipped_features = np.sum(
            (scaled_features < feature_min) |
            (scaled_features > feature_max)
        )

        matches = 0

        for features in X_test:
            features_std = (features - means) / scales

            quantised_features = [quantise(f, FEATURE_BITS, FEATURE_FRAC) for f in features_std]

            scores, predicted_index = fixed_point_inference(quantised_features, quantised_weights, quantised_biases, ACC_BITS)

            float_scores = weights @ features_std + biases
            float_index = int(np.argmax(float_scores))

            matches += (float_index == predicted_index)

        print(f"Format {FEATURE_BITS}/{FEATURE_FRAC} | Agreement: {matches}/{len(X_test)}")
        print("Clipped weights:", clipped_weights, "/", weights.size)
        print(f"Clipped features: {clipped_features}/{X_std.size}")
        print()

        agreement_results[note].append(matches)
        weight_clipping_results[note].append(int(clipped_weights))
        feature_clipping_results[note].append(int(clipped_features))

formats = [
    f"B{bits}_F{frac}"
    for bits, frac in zip(feature_bits_sweep, feature_frac_sweep)
]

def save_results(results, filename):
    table = pd.DataFrame(results, index=formats)
    table.index.name = "Format (B/F)"

    table["Total"] = table.sum(axis=1)

    table.to_csv(results_dir / filename)


save_results(agreement_results, "prediction_agreement.csv")
save_results(weight_clipping_results, "clipped_weights.csv")
save_results(feature_clipping_results, "clipped_features.csv")