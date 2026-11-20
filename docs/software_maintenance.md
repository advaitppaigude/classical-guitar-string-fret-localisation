# Existing software: setup and maintenance

The maintenance work prepares the original software and an empty `fpga/`
workspace. It does not implement the engineering tasks in the FPGA project
specification.

## Environment and dependencies

Tested with **CPython 3.14.4 on Linux x86_64**. There were no dependency manifests
or tests in the original repository. `requirements.txt` now pins the tested
direct and transitive dependencies; it is an environment snapshot, not a claim
that earlier Python or library versions produce identical fitted coefficients.

| Direct dependency | Used for |
| --- | --- |
| NumPy | Feature vectors and existing signal processing |
| pandas | Reading and selecting CSV data |
| SciPy | WAV input, FFTs, windows and peak detection |
| scikit-learn | StandardScaler, logistic regression and historical linear SVM |
| Matplotlib | Historical temporal-envelope plots |

Tests use the standard-library `unittest` framework and the existing scientific
dependencies. No FPGA tools or new testing frameworks are required.

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -s tests -v
```

## Running the original scripts

```bash
# Default: the existing logistic-regression leave-one-out experiment
python -m src.classifier_experiments

# Other runnable original experiments
python -m src.classifier_experiments --iteration 1  # 3-NN
python -m src.classifier_experiments --iteration 2  # rank-weighted scoring
python -m src.classifier_experiments --iteration 5  # linear SVM

# Original positional WAV argument and printed result format
python -m src.guitar_string_fret_localiser /path/to/recording.wav
```

Direct execution such as `python src/classifier_experiments.py` remains
supported. Script data paths resolve relative to the repository, so an absolute
script path also works from another working directory. Module commands should
be run from the repository root. WAV arguments resolve relative to the caller.

The repository includes `data/features.csv` and `data/metadata.csv`, but no WAVs.
To regenerate those CSVs, put the original recordings in `Recordings/` using
names such as `D3_S4_F0_01.wav`, then run:

```bash
python -m src.extract_metadata
python -m src.extract_features
```

These commands **overwrite** `data/metadata.csv` and `data/features.csv` with
results for the WAVs currently present. Do not regenerate from a partial set of
recordings if you want to retain the original dataset.

With the original recordings available, the existing plotting scripts run as:

```bash
python -m src.mean_sd_temporal_envelopes_overlay
python -m src.initial_temporal_secondary_peak_analysis
```

The first uses `data/metadata.csv` and the G3 recordings in `Recordings/`. The
second requires the earlier `Ei_1.wav` through `Ei_5.wav` recordings in
`Recordings/` and retains its original manually selected time windows. Both
display plots and need a graphical Matplotlib backend for interactive use.

## Reusing training and inference

```python
from src.guitar_string_fret_localiser import (
    extract_recording_features,
    feature_columns,
    load_dataset_features,
    predict_features,
    train_note_classifier,
)

dataset = load_dataset_features()  # Optional alternate CSV path accepted
model = train_note_classifier("D3", dataset)  # Train once for this note
vectors = dataset.loc[dataset["note"] == "D3", feature_columns].to_numpy(dtype=float)
labels, probabilities = predict_features(model, vectors)

# Reuse the fitted model for an individual vector:
labels, probabilities = predict_features(model, vectors[0])

# For audio, select a model using the detected note:
note, vector = extract_recording_features("/path/to/recording.wav")
models = {name: train_note_classifier(name, dataset) for name in ("E4", "D3", "E3", "F4", "G3")}
labels, probabilities = predict_features(models[note], vector)
```

`train_note_classifier(note_name, dataset_features=None)` returns a fitted
scikit-learn `Pipeline` with steps `scaler` (`StandardScaler()`) and `classifier`
(`LogisticRegression()`). Omitting the DataFrame loads `data/features.csv`.
Training selects exactly the original 30 filename-matched recordings for the
specified note: takes 01–10 at each of its three positions, in the original
order. Other rows are ignored by the same filename selection as before.

`predict_features(model, input_features)` accepts a raw vector with shape `(8,)`
or a batch with shape `(n, 8)`, applies the fitted scaler, and returns label and
probability arrays with shapes `(n,)` and `(n, 3)`. Probability columns follow
`model.classes_`. The function never trains or updates the model. Keep one fitted
model for each supported note and reuse it across calls.

The required feature order is unchanged:

```text
h2_h1, h3_h1, h4_h1, h5_h1, log_h3_h2,
spectral_centroid, harmonic_centroid, mean_relative_harmonic_deviation
```

`extract_recording_features(wav_path)` returns `(note_name, input_vector)` using
the original audio processing, analysis window and feature calculations. It does
not train. `load_dataset_features(features_csv=...)` reads the CSV and strips
filename whitespace as the original CLI did.

The command-line localiser remains a one-recording command: it trains the
detected note's model once and prints the same four result lines. Repeated
in-process predictions should use the functions above.

## Corrections and preserved behaviour

- `classifier_experiments.py` used `inharmonicity`, which is absent from the
  dataset. It now selects `mean_relative_harmonic_deviation`, matching the
  extractor and final localiser. No feature values or calculations changed.
- The historical script previously invoked no experiment and printed zero
  scores. Its entry point now runs the existing logistic-regression LOOCV by
  default, with optional selection of the other runnable original iterations.
  Importing it no longer prints an unevaluated summary.
- Historical iteration 1 referenced an uninitialised `training_data` and never
  populated its training-vector array. The list is now initialised within each
  fold and the existing vectors are collected from it. Its original majority
  scoring is retained.
- The localiser's original training block is reusable through a Pipeline;
  regression tests compare scaler statistics, classifier parameters, labels,
  and probabilities against the original procedure for all five notes.
- CSV and recording paths now refer to `data/` and `Recordings/` relative to the
  repository. Package and direct-script imports both work. Extraction imports
  are explicit, including pandas instead of obtaining it through a wildcard.
- Extraction file discovery, CSV loading and row accumulation now happen inside
  `main()`. Imports do not perform extraction; repeated runs do not reuse stale
  file lists or append previously processed rows.
- The feature extractor already calculated and wrote `rolloff_85` but omitted it
  from `DictWriter` fieldnames, causing CSV writing to fail. The header now
  includes this existing field in the same order as the tracked CSV. The
  classifier's eight input features remain unchanged.
- The root README's missing `docs/development_log.pdf` link now points to the
  existing DOCX development log.

Inference still trains on all 30 original rows for the detected note. Historical
logistic regression and SVM still perform 30 leave-one-out fits per note, each
using the other **29** rows; the scaler is fitted only on those training rows.
There are no new splits, filters, hyperparameters, preprocessing steps, or
feature-engineering changes. The tracked CSVs and signal-processing code are
unchanged.

## Verification and known limits

On the pinned environment, the original runnable experiments complete with:

| Existing experiment | E4 | D3 | E3 | F4 | G3 | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1: 3-NN with original majority scoring | 27 | 30 | 30 | 30 | 29 | 146/150 |
| 2: rank-weighted scoring | 24 | 29 | 28 | 29 | 27 | 137/150 |
| 4: logistic regression | 30 | 29 | 30 | 29 | 30 | 148/150 |
| 5: linear SVM | 29 | 30 | 30 | 30 | 30 | 149/150 |

Logistic regression and SVM reproduce the README's totals. The repaired 3-NN
iteration gives 146/150 with the current CSV, feature list and majority scoring;
the README historically reports 137/150 for 3-NN. That discrepancy is retained
and documented rather than changing features, data or scoring to match it.

Historical iteration 3 calls an absent `distance_weighting` function. Although
the README mentions inverse-square weighting, its original implementation and
zero-distance behaviour are unavailable. The function is left unchanged and is
not exposed as a runnable CLI choice. No replacement weighting rule is invented.

`features.csv` contains 150 labelled training rows plus three incomplete
`test_32.wav`–`test_34.wav` rows. The original filename selection excludes these;
the CSV is preserved. No later-session evaluation feature table or audio is
present, so the historical 27/30 later-session result cannot be rerun here.
Original audio extraction and plotting cannot be validated against the absent
recordings. A synthetic WAV regression checks audio I/O, detected note, complete
CSV output and repeated extraction, without claiming acoustic accuracy.

The **eight passing software regression tests** also check exact agreement with the original
training procedure, inference without fitting, filename selection, CLI output,
and the original LOOCV training rows. An additional temporary check compared the
unmodified Git HEAD localiser against the refactored direct-script and module
entry points on synthetic WAVs for all five notes: printed output agreed exactly
in every case, including execution from another working directory.

Commands executed successfully in the repository virtual environment:

```bash
python -m unittest discover -s tests -v
python src/classifier_experiments.py
python src/classifier_experiments.py --iteration 1
python src/classifier_experiments.py --iteration 2
python -m src.classifier_experiments --iteration 5
python -m pip install --no-index --dry-run -r requirements.txt
python -m pip check
python -m compileall -q src tests fpga/python
git diff --check
```

The baseline system interpreter had none of the five direct scientific
dependencies. They were installed into `.venv/`; an initial package download
timed out and a retry succeeded. The environment and generated caches are
ignored by Git. No FPGA tests or toolchains have been added or run.
