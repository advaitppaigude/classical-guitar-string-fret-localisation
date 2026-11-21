"""Compare the exported classifier formats on test_1.wav through test_30.wav."""

import argparse
from pathlib import Path

import numpy as np

from src.guitar_string_fret_localiser import extract_recording_features, feature_columns, positions

from .convert_models_to_rom import FORMATS, MODEL_DIR, NOTES, quantise
from .generate_regression_rom import expected_class


SWEEP_FORMATS = (*FORMATS, (12, 8), (16, 10))


def load_models():
    models = {}
    for note in NOTES:
        with np.load(MODEL_DIR / f"{note}.npz", allow_pickle=False) as model:
            if list(model["feature_columns"]) != feature_columns:
                raise ValueError(f"{note}: exported feature order differs from the WAV extractor")
            models[note] = {
                key: model[key] for key in ("means", "scales", "weights", "biases", "classes")
            }
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings_dir", type=Path, help="Directory containing the 30 WAV files")
    args = parser.parse_args()
    paths = [args.recordings_dir / f"test_{number}.wav" for number in range(1, 31)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.recordings_dir}: {', '.join(missing)}")

    models = load_models()
    labels = []
    floating_predictions = []
    fixed_predictions = {fmt: [] for fmt in SWEEP_FORMATS}

    for index, path in enumerate(paths):
        detected_note, vector = extract_recording_features(path)
        model = models[detected_note]
        standardised = (vector - model["means"]) / model["scales"]
        float_index = int(np.argmax(model["weights"] @ standardised + model["biases"]))
        floating_predictions.append((detected_note, model["classes"][float_index]))

        # Provisional labels: two sessions repeat the note and position order in the repo.
        note_index, position_index = divmod(index % (3 * len(NOTES)), 3)
        label_note = NOTES[note_index]
        position = positions[label_note][position_index]
        labels.append((label_note, f"S{position['S']}_F{position['F']}"))

        for bits, frac_bits in SWEEP_FORMATS:
            acc_bits = 2 * bits + 4
            features = [quantise(value, bits, frac_bits) for value in standardised]
            weights = [
                [quantise(value, bits, frac_bits) for value in row] for row in model["weights"]
            ]
            biases = [quantise(value, acc_bits, 2 * frac_bits) for value in model["biases"]]
            class_index = expected_class(features, weights, biases, acc_bits)
            fixed_predictions[bits, frac_bits].append((detected_note, model["classes"][class_index]))

    total = len(paths)
    float_correct = sum(pred == label for pred, label in zip(floating_predictions, labels, strict=True))
    print("Accuracy assumes both sessions repeat the repository's position order per note.")
    print(f"Floating-point accuracy: {float_correct}/{total} ({float_correct / total:.1%})")
    for (bits, frac_bits), predictions in fixed_predictions.items():
        correct = sum(pred == label for pred, label in zip(predictions, labels, strict=True))
        differences = [
            path.name for path, fixed, floating in zip(paths, predictions, floating_predictions, strict=True)
            if fixed != floating
        ]
        matches = total - len(differences)
        print(
            f"B{bits}_F{frac_bits}: {matches}/{total} agreement with float ({matches / total:.1%}); "
            f"{correct}/{total} accuracy ({correct / total:.1%})"
        )
        if differences:
            print(f"  Different from float: {', '.join(differences)}")

    headers = ["Recording", "Assumed label", "FP prediction"] + [
        f"B{bits}_F{frac_bits}" for bits, frac_bits in SWEEP_FORMATS
    ]
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for path, label, floating, *fixed in zip(
        paths, labels, floating_predictions, *fixed_predictions.values(), strict=True
    ):
        if any(prediction != floating for prediction in fixed):
            cells = [path.name, *["_".join(prediction) for prediction in (label, floating, *fixed)]]
            print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
