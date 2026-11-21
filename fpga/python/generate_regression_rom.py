"""Export the 150 original cases and optionally 30 later WAVs for an on-board self-test.

Each word has eight signed B-bit features in its low bits, starting with
feature 0, then two expected-class bits and three note bits. Pad to a
whole hex digit. Expected classes come from the fixed-point reference.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.guitar_string_fret_localiser import extract_recording_features, feature_columns

from .convert_models_to_rom import FORMATS, MODEL_DIR, NOTES, ROM_DIR, ROOT, quantise, write_mem


FEATURE_BITS = 6
FRAC_BITS = 3
ACC_BITS = 2 * FEATURE_BITS + 4
FEATURE_COUNT = 8


def expected_class(features, weights, biases, acc_bits=ACC_BITS):
    minimum = -(1 << (acc_bits - 1))
    maximum = (1 << (acc_bits - 1)) - 1
    scores = []
    for class_weights, score in zip(weights, biases, strict=True):
        # Match the RTL's descending index order and saturate after each add.
        for feature, weight in zip(reversed(features), reversed(class_weights), strict=True):
            score = max(minimum, min(maximum, score + feature * weight))
        scores.append(score)
    return max(range(len(scores)), key=scores.__getitem__)


def main():
    formats = {f"b{bits}_f{frac}": (bits, frac) for bits, frac in FORMATS}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=formats, default="b6_f3")
    parser.add_argument(
        "--recordings-dir", type=Path, help="Append test_1.wav through test_30.wav to the ROM"
    )
    args = parser.parse_args()
    bits, frac_bits = formats[args.format]
    acc_bits = 2 * bits + 4
    feature_mask = (1 << bits) - 1
    payload_bits = FEATURE_COUNT * bits
    word_bits = (payload_bits + 5 + 3) // 4 * 4

    dataset = pd.read_csv(ROOT.parent / "data/features.csv")
    later_rows = {note: [] for note in NOTES}
    if args.recordings_dir is not None:
        for index in range(1, 31):
            note, vector = extract_recording_features(args.recordings_dir / f"test_{index}.wav")
            later_rows[note].append(vector)

    words = []
    later_words = []
    for note_index, note in enumerate(NOTES):
        with np.load(MODEL_DIR / f"{note}.npz", allow_pickle=False) as model:
            columns = list(model["feature_columns"])
            rows = dataset.loc[dataset["note"] == note, columns].dropna()
            if rows.shape != (30, FEATURE_COUNT):
                raise ValueError(f"{note}: expected 30 rows of 8 features, found {rows.shape}")
            means, scales = model["means"], model["scales"]
            if args.recordings_dir is not None and columns != feature_columns:
                raise ValueError(f"{note}: exported feature order differs from the WAV extractor")
            weights = [
                [quantise(value, bits, frac_bits) for value in row]
                for row in model["weights"]
            ]
            biases = [quantise(value, acc_bits, 2 * frac_bits) for value in model["biases"]]

        for raw_rows, destination in ((rows.to_numpy(dtype=float), words), (later_rows[note], later_words)):
            for row in raw_rows:
                features = [quantise(value, bits, frac_bits) for value in (row - means) / scales]
                class_index = expected_class(features, weights, biases, acc_bits)
                word = (note_index << (payload_bits + 2)) | (class_index << payload_bits)
                for index, feature in enumerate(features):
                    word |= (feature & feature_mask) << (index * bits)
                destination.append(word)

    suffix = "_with_later" if args.recordings_dir is not None else ""
    path = ROM_DIR / f"{args.format}_regression{suffix}.mem"
    words.extend(later_words)
    write_mem(path, words, word_bits)
    print(f"Generated {len(words)} regression vectors in {path}")


if __name__ == "__main__":
    main()
