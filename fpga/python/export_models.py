"""
this exports the existing note-specific logistic-regression models to .npz

Run from the repository root:
    python fpga/python/export_models.py

fpga cannot run sklearn.LogisticRegression, but
can perform the arithmetic that the trained model represents


there are two components to my existing code:
1. StandardScaler() - turns features into standardised values based on
training set means and standard deviations
2. LogisticRegression() - learns coefficients/biases to distinguish 3 possible
string/fret positions

both of these will be fitted during training, but after this their params are fixed

for a given note, there will be 8 extracted input features that are considered, and 3 biases
therefore there are 24 weights in a 3x8 matrix M
the input will be vector v which has 8 rows and 1 column (since only 1 note)

therefore if you premultiply v by M, you get a 3x1 matrix/vector A
each row in A corresponds to one possible string/fret position

standardisation will occur in python and features will be sent to fpga
"""


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "data" / "features.csv"
OUTPUT_DIR = ROOT / "fpga" / "models"

FEATURE_COLUMNS = [
    "h2_h1", "h3_h1", "h4_h1", "h5_h1", "log_h3_h2",
    "spectral_centroid", "harmonic_centroid", "mean_relative_harmonic_deviation",
]

# Same ordering/positions as src/guitar_string_fret_localiser.py.
POSITIONS = {
    "E4": [(1, 0), (2, 5), (3, 9)],
    "D3": [(4, 0), (5, 5), (6, 10)],
    "E3": [(4, 2), (5, 7), (6, 12)],
    "F4": [(1, 1), (2, 6), (3, 10)],
    "G3": [(3, 0), (4, 5), (5, 10)],
}


def training_examples(data: pd.DataFrame, note: str):
    """Reproduce the main script's 30-recording training order."""
    rows, labels = [], []
    for string, fret in POSITIONS[note]:
        for take in range(1, 11):
            filename = f"{note}_S{string}_F{fret}_{take:02d}.wav"
            matches = data.loc[data["filename"] == filename]
            if len(matches) != 1:
                raise ValueError(f"Expected one record for {filename}, got {len(matches)}")
            values = matches.iloc[0][FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features in {filename}")
            rows.append(values)
            labels.append(f"S{string}_F{fret}")
    return np.stack(rows), np.asarray(labels)


def main():
    data = pd.read_csv(FEATURES_CSV)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for note in POSITIONS:
        X, y = training_examples(data, note)
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        classifier = LogisticRegression()
        classifier.fit(X_std, y)

        destination = OUTPUT_DIR / f"{note}.npz"
        np.savez(
            destination,
            weights=classifier.coef_,
            biases=classifier.intercept_,
            means=scaler.mean_,
            scales=scaler.scale_,
            classes=classifier.classes_,
            feature_columns=np.asarray(FEATURE_COLUMNS),
        )

        # check saved parameters without loading a scikit-learn model object
        with np.load(destination, allow_pickle=False) as model:
            standardized = (X - model["means"]) / model["scales"]
            scores = standardized @ model["weights"].T + model["biases"]
            predicted = model["classes"][np.argmax(scores, axis=1)]
        expected = classifier.predict(scaler.transform(X))
        np.testing.assert_array_equal(predicted, expected)
        np.testing.assert_allclose(
            scores, classifier.decision_function(scaler.transform(X)),
            rtol=1e-9, atol=1e-9,
        )
        print(f"{note}: exported {destination.relative_to(ROOT)} "
              f"| {len(X)} training samples | 30/30 reference predictions match")


if __name__ == "__main__":
    main()
