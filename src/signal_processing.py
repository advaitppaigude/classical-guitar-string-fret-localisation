# signal_processing.py
"""
contains the following signal processing helper functions:
get_normalised_waveform(path)
get_rms_envelope(data, rate, window_s=0.02, hop_s=0.005)
find_onset_time(rms_times, rms_values)
find_rms_peak_and_time(rms_times, rms_values)
get_fft(rate, normalised_data, analysis_start_time, analysis_end_time)
find_h1_near_ideal(frequency_axis, fft_magnitude, f0_ideal, tolerance_hz=15)
find_harmonic_peaks(frequency_axis, fft_magnitude, f0, max_harmonic=5)
calculate_harmonic_centroid(h_freq, h_amp)
calculate_spectral_centroid(frequency_axis, fft_magnitude)
calculate_rolloff(frequency_axis, fft_magnitude, rolloff_percent=0.85)
calculate_mean_relative_harmonic_deviation(h_freq, h1)
safe_divide(a, b)
log_h3_h2(h3_amp, h2_amp)
harmonic_score(f0, frequency_axis, fft_magnitude, num_harmonics=5, tolerance_hz=5)
get_candidate_frequencies(frequency_axis, fft_magnitude)
find_fundamental(frequency_axis, fft_magnitude)
"""


import numpy as np
import math
import pandas as pd
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann
from scipy.signal import find_peaks



RMS_WINDOW_S = 0.03 # 30ms
RMS_HOP_S = 0.005 # 5ms

ANALYSIS_OFFSET_FROM_RMS_PEAK_S = 0.25
ANALYSIS_DURATION_S = 0.5

BIN_THRESHOLD_HZ = 10
ROLLOFF_PERCENT = 0.85

# limiting full spectral features to useful guitar range
SPECTRAL_MIN_HZ = 50
SPECTRAL_MAX_HZ = 5000


def get_normalised_waveform(path):
    rate, data = wavfile.read(path)

    # converting stereo to mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # converting datatype to float
    data = data.astype(np.float64)

    # removing DC offset
    data = data - np.mean(data)

    # normalising amplitude
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        raise ValueError(f"Silent file: {path.name}")
    data = data / max_abs

    return rate, data


def get_rms_envelope(data, rate, window_s=0.02, hop_s=0.005):
    window_samples = int(window_s * rate)
    hop_samples = int(hop_s * rate)

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("RMS window/hop too small")

    rms_values = []
    rms_times = []

    for start in range(0, len(data) - window_samples + 1, hop_samples):
        # constraining region
        end = start + window_samples
        window = data[start:end]

        # calculating rms and corresponding time axis
        rms = np.sqrt(np.mean(window ** 2))
        centre_time = (start + window_samples / 2) / rate

        rms_values.append(rms)
        rms_times.append(centre_time)

    return np.asarray(rms_times), np.asarray(rms_values)


def find_onset_time(rms_times, rms_values):
    noise_mask = rms_times <= 0.5
    baseline = np.mean(rms_values[noise_mask])

    peak_value = np.max(rms_values)

    # taking threshold as 10% of the way from baseline to peak
    threshold = baseline + 0.10 * (peak_value - baseline)

    onset_indices = np.where(rms_values > threshold)[0]

    if len(onset_indices) == 0:
        peak_index = np.argmax(rms_values)
        return rms_times[peak_index]

    return rms_times[onset_indices[0]]


def find_rms_peak_and_time(rms_times, rms_values):
    peak_index = np.argmax(rms_values)
    return rms_times[peak_index], rms_values[peak_index]


def get_fft(rate, normalised_data, analysis_start_time, analysis_end_time):
    # constraining analysis window
    analysis_start_index = int(analysis_start_time * rate)
    analysis_end_index = int(analysis_end_time * rate)

    if analysis_start_index < 0:
        raise ValueError("Analysis start index is negative")

    if analysis_end_index > len(normalised_data):
        raise ValueError("Analysis window exceeds recording length")

    if analysis_end_index <= analysis_start_index:
        raise ValueError("Invalid analysis window")

    time_domain_signal = normalised_data[analysis_start_index:analysis_end_index]

    time_domain_signal = time_domain_signal.astype(np.float64)

    """
    subtracting mean from data to remove the DC (0Hz) offset
    a large nonzero mean can create massive spike at 0Hz, which can leak into neighbouring
    low-frequency bins and obscure faint low-frequency components
    """
    time_domain_signal = time_domain_signal - np.mean(time_domain_signal)


    """
    applying Hann window to taper the edges of the data sample down to 0.
    this prevents sharp jumps at boundaries of samples which would otherwise occur if the
    edges of those segments were cut off abruptly.
    this reduces spectral leakage
    """
    window = hann(len(time_domain_signal))
    windowed_data = time_domain_signal * window

    # computing fft
    fft_values = rfft(windowed_data)
    fft_magnitude = np.abs(fft_values)

    max_mag = np.max(fft_magnitude)
    if max_mag == 0:
        raise ValueError("FFT magnitude is zero")

    # normalising so that fundamental magnitude ~= 1 if f0 is the largest peak
    fft_magnitude = fft_magnitude / max_mag

    frequency_axis = rfftfreq(len(windowed_data), d=1 / rate)

    return frequency_axis, fft_magnitude


def find_h1_near_ideal(frequency_axis, fft_magnitude, f0_ideal, tolerance_hz=15):
    """
    note is already known from the filename, so this is to find the exact frequency
    of the note that is recorded, which may vary due to tuning variations or string/fret position
    """
    mask = (
        (frequency_axis >= f0_ideal - tolerance_hz) &
        (frequency_axis <= f0_ideal + tolerance_hz)
    )

    indices = np.where(mask)[0]

    if len(indices) == 0:
        return np.nan, np.nan

    local_index = np.argmax(fft_magnitude[indices])
    h1_index = indices[local_index]

    h1_freq = frequency_axis[h1_index]

    return h1_freq, h1_index


def find_harmonic_peaks(frequency_axis, fft_magnitude, f0, max_harmonic=5):
    """
    finding amplitude and exact freqency of first n harmonics (maximum being 5, since
    amplitude of higher harmonics was found to be negligible during early testing)

    exact frequency can be used to find mean harmonic deviation from the ideal series
    """
    h_freq = {}
    h_amp = {}

    for k in range(1, max_harmonic + 1):
        expected_freq = k * f0

        harmonic_mask = (
            (frequency_axis >= expected_freq - BIN_THRESHOLD_HZ) &
            (frequency_axis <= expected_freq + BIN_THRESHOLD_HZ)
        )

        harmonic_indices = np.where(harmonic_mask)[0]

        if len(harmonic_indices) == 0:
            h_freq[k] = np.nan
            h_amp[k] = np.nan
            continue

        local_index = np.argmax(fft_magnitude[harmonic_indices])
        harmonic_index = harmonic_indices[local_index]

        h_freq[k] = frequency_axis[harmonic_index]
        h_amp[k] = fft_magnitude[harmonic_index]

    return h_freq, h_amp


def calculate_harmonic_centroid(h_freq, h_amp):
    """
    centroid using only detected harmonic peaks
    not the same as full spectral centroid but significantly cheaper
    """

    freqs = np.array([h_freq[k] for k in range(1, 6)], dtype=float)
    amps = np.array([h_amp[k] for k in range(1, 6)], dtype=float)

    valid = ~np.isnan(freqs) & ~np.isnan(amps)

    if not np.any(valid):
        return np.nan

    return np.sum(freqs[valid] * amps[valid]) / np.sum(amps[valid])


def calculate_spectral_centroid(frequency_axis, fft_magnitude):
    """
    full spectral centroid over a limited useful frequency range
    calculates weighted average of magnitude
    """

    mask = (
        (frequency_axis >= SPECTRAL_MIN_HZ) &
        (frequency_axis <= SPECTRAL_MAX_HZ)
    )

    freqs = frequency_axis[mask]
    mags = fft_magnitude[mask]

    if np.sum(mags) == 0:
        return np.nan

    return np.sum(freqs * mags) / np.sum(mags)


def calculate_rolloff(frequency_axis, fft_magnitude, rolloff_percent=0.85):
    """
    spectral rolloff: frequency below which rolloff_percent of spectral energy lies
    uses power spectrum, not magnitude
    """

    mask = (
        (frequency_axis >= SPECTRAL_MIN_HZ) &
        (frequency_axis <= SPECTRAL_MAX_HZ)
    )

    freqs = frequency_axis[mask]
    power = fft_magnitude[mask] ** 2

    total_energy = np.sum(power)

    if total_energy == 0:
        return np.nan

    cumulative_energy = np.cumsum(power)
    threshold = rolloff_percent * total_energy

    rolloff_index = np.searchsorted(cumulative_energy, threshold)

    if rolloff_index >= len(freqs):
        rolloff_index = len(freqs) - 1

    return freqs[rolloff_index]


def calculate_mean_relative_harmonic_deviation(h_freq, h1):
    """
    audio-derived proxy for inharmonicity:
    average absolute fractional deviation from exact integer harmonics

    for each harmonic k:
        deviation = |h_k - k h1| / (k h1)

    this is not a full piano-string inharmonicity coefficient B, but rather
    a practical measured harmonic deviation feature
    """

    deviations = []

    for k in range(2, 6):
        if np.isnan(h_freq[k]):
            continue

        expected = k * h1
        measured = h_freq[k]

        deviation = abs(measured - expected) / expected
        deviations.append(deviation)

    if len(deviations) == 0:
        return np.nan

    return np.mean(deviations)


def safe_divide(a, b):
    if b == 0 or np.isnan(a) or np.isnan(b):
        return np.nan
    return a / b


def log_h3_h2(h3_amp, h2_amp):
    epsilon = 1e-6
    return math.log((h3_amp + epsilon) / (h2_amp + epsilon))


def harmonic_score(f0, frequency_axis, fft_magnitude, num_harmonics=5, tolerance_hz=5):
    score = 0

    for harmonic_num in range(1, num_harmonics + 1):
        target_freq = harmonic_num * f0

        valid = (
            (frequency_axis >= target_freq - tolerance_hz) &
            (frequency_axis <= target_freq + tolerance_hz)
        )

        if np.any(valid):
            score += np.max(fft_magnitude[valid]) / harmonic_num

    return score


def get_candidate_frequencies(frequency_axis, fft_magnitude):
    min_f0 = 80
    max_f0 = 700

    valid = (
        (frequency_axis >= min_f0) &
        (frequency_axis <= max_f0)
    )

    valid_freqs = frequency_axis[valid]
    valid_mags = fft_magnitude[valid]

    peak_indices, properties = find_peaks(
        valid_mags,
        prominence=np.max(valid_mags) * 0.05
    )

    candidate_frequencies = valid_freqs[peak_indices]

    return candidate_frequencies


def find_fundamental(frequency_axis, fft_magnitude):

    best_f0 = None
    best_score = -1

    candidate_frequencies = get_candidate_frequencies(frequency_axis, fft_magnitude)

    for candidate_f0 in candidate_frequencies:
        score = harmonic_score(
            candidate_f0,
            frequency_axis,
            fft_magnitude
        )

        if score > best_score:
            best_score = score
            best_f0 = candidate_f0

    return best_f0
