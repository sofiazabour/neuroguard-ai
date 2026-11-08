"""NeuroGuard's configurable EEG smoke test; not an emotion classifier.

The two-argument process() entry point remains compatible with HABS.
Run locally with: python process.py --data sample --out out/smoke
"""

import argparse
import json
import math
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

#: The loader that comes with the notebook travels with your job, because
#: `habs submit` uploads this repository. It reads the .record_map beside a
#: recording, so it knows which of the 84 channels are electrodes and which
#: are ECG, respiration, skin conductance or gaze -- and it puts the 166 Hz
#: gaze samples on the same timeline as the 1 kHz signals.
from starting_notebooks.bdf_loader import load_bdf

#: BDF headers do not agree on a unit. MNE wants volts.
TO_VOLTS = {"V": 1.0, "mV": 1e-3, "uV": 1e-6, "µV": 1e-6, "μV": 1e-6, "nV": 1e-9}

DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.toml")


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read and validate the same configuration locally and on HABS."""
    with Path(path).open("rb") as handle:
        config = tomllib.load(handle)
    required = {"dataset_variant", "max_recordings", "band_hz"}
    if set(config) != required:
        raise ValueError(f"Configuration must contain exactly: {', '.join(sorted(required))}")
    if config["dataset_variant"] not in ("CURATED", "PREPROCESSED"):
        raise ValueError("dataset_variant must be CURATED or PREPROCESSED")
    limit = config["max_recordings"]
    if type(limit) is not int or limit < 1:
        raise ValueError("max_recordings must be a positive integer")
    band = config["band_hz"]
    if (
        not isinstance(band, list)
        or len(band) != 2
        or any(type(x) not in (int, float) or not math.isfinite(x) for x in band)
        or not 0 < band[0] < band[1] <= 48
    ):
        raise ValueError("band_hz must contain two frequencies: 0 < low < high <= 48")
    return config


def alpha_power(bdf: Path, band_hz: tuple | list = (8.0, 12.0)) -> pd.DataFrame:
    """Absolute band power per electrode, in uV^2; alpha by default."""
    import mne
    from scipy.integrate import trapezoid

    raw, signals, channels = load_bdf(bdf)
    electrodes = channels.loc[channels.mne_type == "eeg", "bdf_label"].tolist()
    units = channels.set_index("bdf_label")["header_unit"].to_dict()
    rate = float(raw.info["sfreq"])
    if band_hz[1] > rate / 2:
        raise ValueError(f"Requested band exceeds the Nyquist frequency for {bdf.name}")

    volts = np.vstack([signals[ch].to_numpy() * TO_VOLTS[units[ch]] for ch in electrodes])
    eeg = mne.io.RawArray(volts, mne.create_info(electrodes, rate, "eeg"), verbose="ERROR")

    # A two-second window: long enough to resolve the alpha peak, short enough
    # that Welch averages over many of them.
    window = min(round(2 * rate), eeg.n_times)
    spectrum = eeg.compute_psd(method="welch", fmin=0.5, fmax=min(48.0, rate / 2),
                               n_fft=window, n_per_seg=window, n_overlap=window // 2,
                               verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    psd = psd * 1e12                                   # V^2/Hz -> uV^2/Hz
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if band.sum() < 2:
        raise ValueError(f"Not enough frequency bins for the requested band in {bdf.name}")
    if not np.isfinite(psd).all():
        raise ValueError(f"Non-finite EEG spectrum in {bdf.name}; inspect signal quality")

    return pd.DataFrame({
        "channel": electrodes,
        "band_uV2": trapezoid(psd[:, band], x=freqs[band], axis=1),
        "seconds": eeg.n_times / rate,
    })


def process(data_dir: Path, out_dir: Path, *, config_path: Path = DEFAULT_CONFIG) -> None:
    """Read recordings from data_dir, write your results into out_dir.

    data_dir is the dataset root, read-only and identical for every team:
    data_dir/CURATED/P001/..., and on the platform also labels_train.csv,
    markers.csv and participants_train.csv. Anything you write into out_dir
    is kept.

    The same code runs against sample/, which has the same layout.
    """
    started = datetime.now(timezone.utc)
    clock = perf_counter()
    config = load_config(config_path)
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    # Named rather than globbed over everything. The dataset is 351 GB: the
    # emotion-video recordings are 150-280 MB each, resting-state about 20.
    recordings = sorted(data_dir.glob(
        f"{config['dataset_variant']}/P*/P*_task-resting-state_*.bdf"
    ))
    if not recordings:
        raise FileNotFoundError(
            f"No resting-state BDF files found in {data_dir / config['dataset_variant']}. "
            "Use --data sample for the included smoke test, or check your dataset path."
        )
    selected = recordings[:config["max_recordings"]]
    for bdf in selected:
        if not bdf.with_suffix(".record_map").is_file():
            raise FileNotFoundError(f"Missing record map: {bdf.with_suffix('.record_map')}")
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f"Output directory must be empty to preserve earlier runs: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(recordings)} resting-state recording(s) found; "
          f"using {len(selected)} from {config['dataset_variant']}", flush=True)

    per_channel, summary = [], []
    for bdf in selected:
        print(f"Loading {bdf.name}", flush=True)
        table = alpha_power(bdf, config["band_hz"])
        table.insert(0, "recording", bdf.stem)
        per_channel.append(table)

        strongest = table.loc[table.band_uV2.idxmax()]
        summary.append({
            "recording": bdf.stem,
            "participant": bdf.parent.name,
            "seconds": round(float(table.seconds.iloc[0]), 1),
            "electrodes": len(table),
            "band_uV2_mean": float(table.band_uV2.mean()),
            "strongest_channel": strongest.channel,
            "strongest_uV2": float(strongest.band_uV2),
        })
        print(f"  {bdf.stem}: {summary[-1]['band_uV2_mean']:.1f} uV^2 mean, "
              f"strongest {strongest.channel} at {strongest.band_uV2:.1f}", flush=True)

    pd.DataFrame(summary).to_csv(out_dir / "band_power.tsv", sep="\t", index=False)
    pd.concat(per_channel).to_csv(out_dir / "band_power_by_channel.tsv", sep="\t", index=False)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parent,
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip())
    except (OSError, subprocess.SubprocessError):
        commit = None  # HABS can run a git archive without a .git directory.
        dirty = None
    manifest = {
        "status": "completed",
        "started_at_utc": started.isoformat(),
        "elapsed_seconds": round(perf_counter() - clock, 3),
        "git_commit": commit,
        "git_dirty": dirty,
        "python_version": sys.version.split()[0],
        "packages": {name: version(name) for name in ("numpy", "pandas", "scipy", "mne", "pyedflib")},
        "config": config,
        "data_dir": str(data_dir.resolve()),
        "recordings_found": len(recordings),
        "recordings_processed": len(selected),
        "inputs": [{"path": str(bdf.relative_to(data_dir)), "bytes": bdf.stat().st_size}
                   for bdf in selected],
        "outputs": ["band_power.tsv", "band_power_by_channel.tsv"],
    }
    (out_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote results and run.json to {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset root, e.g. sample or /data")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="TOML configuration path")
    args = parser.parse_args()
    try:
        process(args.data, args.out, config_path=args.config)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
