"""Rank fixed-point formats by prediction agreement, width, and clipping."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.guitar_string_fret_localiser import (
    extract_recording_features,
    feature_columns,
    load_dataset_features,
)

from .convert_models_to_rom import ROOT
from .generate_regression_rom import expected_class
from .sweep_later_session import load_models


DEFAULT_OUTPUT = ROOT / "results/numerical_exploration/bit_fraction_sweep.csv"
CLIP_GROUPS = ("later_feature", "original_feature", "weight", "bias")


def load_vectors(models, recordings):
    dataset = load_dataset_features()
    original = {
        note: dataset.loc[dataset["note"] == note, feature_columns].dropna().to_numpy()
        for note in models
    }
    later = {note: [] for note in models}
    for path in recordings:
        note, raw = extract_recording_features(path)
        later[note].append(raw)

    datasets = {"original": original, "later": later}
    for group, batches in datasets.items():
        for note, model in models.items():
            raw = np.asarray(batches[note], dtype=float).reshape(-1, len(feature_columns))
            standardised = (raw - model["means"]) / model["scales"]
            if not np.isfinite(standardised).all():
                raise ValueError(f"Non-finite standardised features in {group} data for {note}")
            batches[note] = standardised
    return datasets


def quantise_array(values, bits, frac_bits):
    scale = 1 << frac_bits
    rounded = np.rint(values * scale)
    limit = 1 << (bits - 1)
    clamped = rounded.clip(-limit, limit - 1)
    excess = np.abs(rounded - clamped)
    return clamped.astype(np.int64), int(np.count_nonzero(excess)), float(excess.max(initial=0) / scale)


def record_clipping(row, group, result):
    quantised, count, max_excess = result
    row[f"{group}_clips"] += count
    row[f"{group}_max_clip_excess"] = max(row[f"{group}_max_clip_excess"], max_excess)
    return quantised.tolist()


def evaluate_format(bits, frac_bits, models, datasets, references):
    acc_bits = 2 * bits + 4
    row = {"B": bits, "F": frac_bits, "acc_bits": acc_bits}
    for group in datasets:
        row[f"{group}_matches"] = 0
        row[f"{group}_total"] = sum(len(vectors) for vectors in datasets[group].values())
    for group in CLIP_GROUPS:
        row[f"{group}_clips"] = 0
        row[f"{group}_max_clip_excess"] = 0.0

    for note, model in models.items():
        weights = record_clipping(row, "weight", quantise_array(model["weights"], bits, frac_bits))
        biases = record_clipping(row, "bias", quantise_array(model["biases"], acc_bits, 2 * frac_bits))
        for group, batches in datasets.items():
            features = record_clipping(
                row, f"{group}_feature", quantise_array(batches[note], bits, frac_bits)
            )
            references_for_note = references[group][note].tolist()
            row[f"{group}_matches"] += sum(
                expected_class(vector, weights, biases, acc_bits) == reference
                for vector, reference in zip(features, references_for_note, strict=True)
            )
    return row


def ranking_key(row):
    return (
        -row["later_matches"],
        -row["original_matches"],
        row["B"],
        *(row[f"{group}_clips"] for group in CLIP_GROUPS),
        *(row[f"{group}_max_clip_excess"] for group in CLIP_GROUPS),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings_dir", type=Path, help="Directory with test_1.wav through test_30.wav")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="CSV output path")
    args = parser.parse_args()
    recordings = [args.recordings_dir / f"test_{index}.wav" for index in range(1, 31)]
    missing = [path.name for path in recordings if not path.is_file()]
    if missing:
        parser.error(f"Missing recordings in {args.recordings_dir}: {', '.join(missing)}")

    models = load_models()
    datasets = load_vectors(models, recordings)
    references = {
        group: {
            note: np.argmax(vectors @ models[note]["weights"].T + models[note]["biases"], axis=1)
            for note, vectors in batches.items()
        }
        for group, batches in datasets.items()
    }
    rows = [
        evaluate_format(bits, frac_bits, models, datasets, references)
        for bits in range(5, 17)
        for frac_bits in range(bits)
    ]
    rows.sort(key=lambda row: (*ranking_key(row), row["F"]))
    ranked = []
    previous_key = None
    for index, row in enumerate(rows, start=1):
        key = ranking_key(row)
        if key != previous_key:
            rank = index
            previous_key = key
        ranked.append({"rank": rank, **row})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=ranked[0])
        writer.writeheader()
        writer.writerows(ranked)

    print(f"Wrote {len(ranked)} formats to {args.output}")
    print("Ranking orders by later agreement, original agreement, lower B, then clipping.")
    print("Clipping order is later features, original features, weights, and biases.")
    print("Rank  Format    Later FP  Original FP  Clips (later/original/weights/biases)")
    for row in ranked:
        if row["rank"] > 10:
            break
        label = f"B{row['B']}_F{row['F']}"
        later = f"{row['later_matches']}/{row['later_total']}"
        original = f"{row['original_matches']}/{row['original_total']}"
        clips = "/".join(str(row[f"{group}_clips"]) for group in CLIP_GROUPS)
        print(f"{row['rank']:>4}  {label:<9} {later:>8}  {original:>11}  {clips}")


if __name__ == "__main__":
    main()
