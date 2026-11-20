"""Software maintenance regressions; no FPGA or new numerical experiments."""

import csv
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import classifier_experiments as experiments
from src import extract_features, extract_metadata
from src import guitar_string_fret_localiser as localiser


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_ORDER = [
    "h2_h1", "h3_h1", "h4_h1", "h5_h1", "log_h3_h2",
    "spectral_centroid", "harmonic_centroid", "mean_relative_harmonic_deviation",
]
ORIGINAL_POSITIONS = {
    "E4": [(1, 0), (2, 5), (3, 9)],
    "D3": [(4, 0), (5, 5), (6, 10)],
    "E3": [(4, 2), (5, 7), (6, 12)],
    "F4": [(1, 1), (2, 6), (3, 10)],
    "G3": [(3, 0), (4, 5), (5, 10)],
}


def original_training_rows(dataset, note):
    """Reproduce the original CLI's filename selection, ordering, and labels."""
    vectors, labels = [], []
    for string, fret in ORIGINAL_POSITIONS[note]:
        for take in range(1, 11):
            filename = f"{note}_S{string}_F{fret}_{take:02d}.wav"
            row = dataset[dataset["filename"] == filename].iloc[0]
            vectors.append(row[FEATURE_ORDER].to_numpy(dtype=float))
            labels.append(f"S{string}_F{fret}")
    return np.array(vectors), np.array(labels)


class ClassifierRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = localiser.load_dataset_features()
        cls.originals = {}
        cls.models = {}
        for note in ORIGINAL_POSITIONS:
            vectors, labels = original_training_rows(cls.dataset, note)
            scaler = StandardScaler()
            classifier = LogisticRegression()
            classifier.fit(scaler.fit_transform(vectors), labels)
            cls.originals[note] = vectors, scaler, classifier
            cls.models[note] = localiser.train_note_classifier(note, cls.dataset)

    def test_training_and_predictions_match_original_for_every_note(self):
        self.assertEqual(localiser.feature_columns, FEATURE_ORDER)
        self.assertEqual(experiments.feature_columns, FEATURE_ORDER)
        for note, model in self.models.items():
            with self.subTest(note=note):
                vectors, scaler, classifier = self.originals[note]
                actual_scaler = model.named_steps["scaler"]
                actual_classifier = model.named_steps["classifier"]
                self.assertEqual(actual_scaler.n_samples_seen_, 30)
                self.assertEqual(actual_scaler.get_params(), scaler.get_params())
                self.assertEqual(actual_classifier.get_params(), classifier.get_params())
                for name in ("mean_", "var_", "scale_"):
                    np.testing.assert_array_equal(getattr(actual_scaler, name), getattr(scaler, name))
                for name in ("classes_", "coef_", "intercept_", "n_iter_"):
                    np.testing.assert_array_equal(getattr(actual_classifier, name), getattr(classifier, name))
                labels, probabilities = localiser.predict_features(model, vectors)
                standardised = scaler.transform(vectors)
                np.testing.assert_array_equal(labels, classifier.predict(standardised))
                np.testing.assert_array_equal(probabilities, classifier.predict_proba(standardised))

    def test_repeated_inference_never_fits_and_supports_single_vectors(self):
        model = self.models["E4"]
        vectors = self.originals["E4"][0]
        expected_labels, expected_probabilities = localiser.predict_features(model, vectors[:2])
        with (
            patch.object(model, "fit", side_effect=AssertionError("inference fitted pipeline")),
            patch.object(StandardScaler, "fit", side_effect=AssertionError("inference fitted scaler")),
            patch.object(LogisticRegression, "fit", side_effect=AssertionError("inference fitted classifier")),
        ):
            for index in range(2):
                labels, probabilities = localiser.predict_features(model, vectors[index])
                self.assertEqual(labels.shape, (1,))
                self.assertEqual(probabilities.shape, (1, 3))
                np.testing.assert_array_equal(labels, expected_labels[index:index + 1])
                np.testing.assert_allclose(probabilities, expected_probabilities[index:index + 1], atol=1e-14)

    def test_training_preserves_filename_selection_and_does_not_mutate_data(self):
        # Unrelated/incomplete rows must not become additional training samples.
        dataset = self.dataset.iloc[::-1].copy()
        dataset["filename"] = " " + dataset["filename"] + " "
        extra = dataset.iloc[[0]].copy()
        extra["filename"] = "E4_future_session.wav"
        extra["note"] = "E4"
        extra[FEATURE_ORDER] = 1000000.0
        dataset = pd.concat([dataset, extra], ignore_index=True)
        unchanged = dataset.copy(deep=True)
        model = localiser.train_note_classifier("E4", dataset)
        pd.testing.assert_frame_equal(dataset, unchanged)
        np.testing.assert_array_equal(model.named_steps["classifier"].coef_, self.models["E4"].named_steps["classifier"].coef_)
        self.assertEqual(model.named_steps["scaler"].n_samples_seen_, 30)

    def test_cli_argument_and_result_format_are_preserved(self):
        vector = self.originals["E4"][0][0]
        _, scaler, classifier = self.originals["E4"]
        probabilities = classifier.predict_proba(scaler.transform(vector.reshape(1, -1)))[0]
        prediction = classifier.predict(scaler.transform(vector.reshape(1, -1)))[0]
        second_index = np.argsort(probabilities)[::-1][1]
        string, fret = prediction.split("_")
        second_string, second_fret = classifier.classes_[second_index].split("_")
        expected = (
            "Detected note: E4\n"
            f"Predicted position: String {string[1:]} Fret {fret[1:]}\n"
            f"Model probability: {probabilities.max():.1%}\n"
            f"Next-best position: String {second_string[1:]} Fret {second_fret[1:]} ({probabilities[second_index]:.1%})\n\n"
        )
        output = io.StringIO()
        with (
            patch.object(sys, "argv", ["guitar_string_fret_localiser.py", "recording.wav"]),
            patch.object(localiser, "extract_recording_features", return_value=("E4", vector)) as extract,
            redirect_stdout(output),
        ):
            localiser.main()
        extract.assert_called_once_with("recording.wav")
        self.assertEqual(output.getvalue(), expected)


class HistoricalExperimentTests(unittest.TestCase):
    def test_knn_training_vectors_are_populated_for_each_fold(self):
        with (
            patch.object(experiments, "notes", ["E4"]),
            patch.object(experiments, "positions", [experiments.positions[0]]),
            patch.object(experiments, "num_correct_per_note", [0]),
        ):
            experiments.iteration_1()
            self.assertEqual(experiments.num_correct_per_note, [27])

    def test_original_loocv_fits_only_the_other_29_recordings(self):
        vectors, _ = original_training_rows(experiments.features, "D3")
        seen = []

        class TrackingScaler(StandardScaler):
            def fit_transform(self, values, y=None, **kwargs):
                seen.append(values.copy())
                return super().fit_transform(values, y, **kwargs)

        with (
            patch.object(experiments, "notes", ["D3"]),
            patch.object(experiments, "positions", [experiments.positions[1]]),
            patch.object(experiments, "num_correct_per_note", [0]),
            patch.object(experiments, "StandardScaler", TrackingScaler),
        ):
            experiments.iteration_4()
            self.assertEqual(experiments.num_correct_per_note, [29])
        self.assertEqual(len(seen), 30)
        for held_out, training in enumerate(seen):
            np.testing.assert_array_equal(training, np.delete(vectors, held_out, axis=0))

    def test_default_experiment_runs_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "src" / "classifier_experiments.py")],
                cwd=directory, text=True, capture_output=True, check=True, timeout=30,
            )
        self.assertEqual(result.stdout, (
            "For note E4, 30/30 classifications were correct\n"
            "For note D3, 29/30 classifications were correct\n"
            "For note E3, 30/30 classifications were correct\n"
            "For note F4, 29/30 classifications were correct\n"
            "For note G3, 30/30 classifications were correct\n"
            "On average, 29.6/30 classifications were correct\n"
        ))


class ExtractionMaintenanceTests(unittest.TestCase):
    def test_metadata_and_features_write_complete_csvs_without_stale_rows(self):
        # A short synthetic D3 WAV exercises existing audio I/O and CSV output.
        # It tests software execution, not acoustic classification accuracy.
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            rate = 48000
            times = np.arange(rate * 2) / rate
            signal = sum(np.sin(2 * np.pi * 146.8323839587038 * harmonic * times) / harmonic for harmonic in range(1, 6))
            signal *= np.exp(-times) * np.minimum(times / 0.02, 1)
            wavfile.write(directory / "D3_S4_F0_01.wav", rate, signal.astype(np.float32))
            metadata_path = directory / "metadata.csv"
            features_path = directory / "features.csv"
            with (
                patch.object(extract_metadata, "WAV_FOLDER", directory),
                patch.object(extract_metadata, "OUTPUT_CSV", metadata_path),
                patch.object(extract_features, "WAV_FOLDER", directory),
                patch.object(extract_features, "METADATA_CSV", metadata_path),
                patch.object(extract_features, "OUTPUT_CSV", features_path),
                redirect_stdout(io.StringIO()) as output,
            ):
                for _ in range(2):
                    extract_metadata.main()
                    extract_features.main()
                note, vector = localiser.extract_recording_features(directory / "D3_S4_F0_01.wav")
            self.assertNotIn("ERROR", output.getvalue())
            self.assertEqual(note, "D3")
            self.assertEqual(vector.shape, (8,))
            self.assertTrue(np.isfinite(vector).all())
            self.assertEqual(len(pd.read_csv(metadata_path)), 1)
            features = pd.read_csv(features_path)
            self.assertEqual(len(features), 1)
            self.assertTrue(np.isfinite(features[FEATURE_ORDER].to_numpy()).all())
            self.assertTrue(np.isfinite(features["rolloff_85"]).all())
            with (PROJECT_ROOT / "data" / "features.csv").open(newline="") as stream:
                expected_header = next(csv.reader(stream))
            self.assertEqual(list(features.columns), expected_header)


if __name__ == "__main__":
    unittest.main()
