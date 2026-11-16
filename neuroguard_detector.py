"""Exploratory, calibration-based EEG artifact-type *candidates*.

Not a validated clinical detector. No task labels are used to decide predictions.
"""
import numpy as np

FRONTAL = ('Fp1','Fp2','AF7','AF3','AF4','AF8')
TEMPORAL = ('F9','F10','T9','T10','FT7','FT8')


def features(signals_uv, channel_names, sfreq):
    """signals_uv shape (channels, samples), one 0.5-second window."""
    x = np.asarray(signals_uv, dtype=float)
    if x.ndim != 2 or x.shape[1] < 8 or not np.isfinite(x).all():
        raise ValueError('Invalid EEG window')
    x = x - np.median(x, axis=1, keepdims=True)
    names = list(channel_names)
    frontal = [names.index(c) for c in FRONTAL if c in names]
    temporal = [names.index(c) for c in TEMPORAL if c in names]
    if not frontal:
        raise ValueError('No frontal EEG channels available')
    p2p = np.ptp(x, axis=1)
    diff = np.mean(np.abs(np.diff(x, axis=1)), axis=1)
    # Welch-like short-window spectral estimate with Hann taper; no external dependency.
    taper = np.hanning(x.shape[1])
    spectrum = np.abs(np.fft.rfft(x * taper, axis=1)) ** 2
    hz = np.fft.rfftfreq(x.shape[1], 1.0 / sfreq)
    high = (hz >= 30) & (hz <= min(90, sfreq / 2))
    low = (hz >= 0.5) & (hz <= 8)
    def bandpower(idx, mask):
        return float(np.median(spectrum[np.ix_(idx, np.where(mask)[0])].sum(axis=1))) if idx and mask.any() else 0.0
    return {
        'frontal_p2p_uv': float(np.median(p2p[frontal])),
        'frontal_line_diff_uv': float(np.median(diff[frontal])),
        'temporal_line_diff_uv': float(np.median(diff[temporal])) if temporal else float(np.median(diff)),
        'frontal_low_power': bandpower(frontal, low),
        'temporal_high_power': bandpower(temporal if temporal else list(range(len(names))), high),
        'global_p2p_uv': float(np.median(p2p)),
    }


FEATURE_NAMES = ('frontal_p2p_uv','frontal_line_diff_uv','temporal_line_diff_uv',
                 'frontal_low_power','temporal_high_power','global_p2p_uv')


def calibrate(rest_features):
    """Only initial rest windows; no future or test labels."""
    if len(rest_features) < 20:
        raise ValueError('Need at least 20 initial-rest windows for calibration')
    return {k: max(float(np.median([r[k] for r in rest_features])), 1e-9) for k in FEATURE_NAMES}


def predict_candidates(feat, baseline):
    """Transparent provisional rules; thresholds must be validated, not optimized on test."""
    ratio = {k: float(feat[k] / baseline[k]) for k in FEATURE_NAMES}
    # Multi-label candidates: ocular and muscle artifacts can co-occur.
    ocular = ratio['frontal_p2p_uv'] >= 2.5 and ratio['frontal_low_power'] >= 2.0
    muscle = ratio['temporal_high_power'] >= 3.0 and ratio['temporal_line_diff_uv'] >= 1.8
    movement = ratio['global_p2p_uv'] >= 3.0 and not ocular and not muscle
    return {
        'ocular_candidate': bool(ocular),
        'muscle_candidate': bool(muscle),
        'movement_candidate': bool(movement),
        **{f'ratio_{k}': round(v, 4) for k, v in ratio.items()},
    }
