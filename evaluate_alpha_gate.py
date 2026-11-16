"""Measure alpha-power availability with and without the Fp1 alert gate."""

import json
from collections import deque
from pathlib import Path

import numpy as np
from pyedflib import EdfReader
from scipy.integrate import trapezoid
from scipy.signal import welch

from process import TO_VOLTS

FOLDER = Path(__file__).resolve().parent / "sample" / "CURATED" / "P001"
CHANNEL = "Fp1"
THRESHOLD_UV = 200.0
WINDOW_S = 0.5
BLOCK_S = 2
ALPHA_BAND_HZ = (8.0, 12.0)


def alpha_power_uv2(samples_uv, rate):
    """Integrate the Welch power spectral density over 8–12 Hz."""
    frequencies, psd = welch(
        samples_uv, fs=rate, nperseg=samples_uv.size, window="hamming"
    )
    in_band = (frequencies >= ALPHA_BAND_HZ[0]) & (frequencies <= ALPHA_BAND_HZ[1])
    return float(trapezoid(psd[in_band], x=frequencies[in_band]))


def report(label, blocks):
    passing = [block for block in blocks if block[3] is not None]
    print(f"\n{label}")
    print(f"  Puissance alpha sans filtrage des alertes : {len(blocks)} blocs")
    if not blocks:
        return

    print(f"  Avec exclusion : {len(passing)}/{len(blocks)} blocs "
          f"({len(passing) / len(blocks):.1%})")
    print(f"  Moyenne sans exclusion : {np.mean([row[2] for row in blocks]):.2f} µV²")
    if passing:
        print(f"  Moyenne après exclusion : "
              f"{np.mean([row[3] for row in passing]):.2f} µV²")
    else:
        print("  Puissance après exclusion : indisponible")


def main():
    metadata = json.loads(
        (FOLDER / "P001_day-1_metadata.json").read_text(encoding="utf-8")
    )
    segment = metadata["dataset_export"]["segments"]["artefact"]
    origin_s = segment["metadata_time_origin_s"]
    task_intervals = {10000: [], 10006: []}
    for event in metadata["events"]:
        marker_id = event["marker_id"]
        duration = event["duration_s"]
        if marker_id in task_intervals and duration is not None and duration > 0:
            start = event["onset_s"] - origin_s
            task_intervals[marker_id].append((start, start + duration))

    rolling_blocks = []
    windows_per_block = round(BLOCK_S / WINDOW_S)
    recent_windows = deque(maxlen=windows_per_block)
    with EdfReader(str(FOLDER / segment["file"])) as reader:
        channel = reader.getSignalLabels().index(CHANNEL)
        rate = reader.getSampleFrequency(channel)
        to_uv = TO_VOLTS[reader.getPhysicalDimension(channel)] * 1e6
        subwindow_size = round(WINDOW_S * rate)
        # Ignore any samples added to pad the final BDF record.
        total = min(segment["n_samples"], reader.getNSamples()[channel])

        # Replay one 500 ms window at a time. Buffer only the latest two seconds.
        for start in range(0, total - subwindow_size + 1, subwindow_size):
            values_uv = reader.readSignal(channel, start, subwindow_size) * to_uv
            if not np.isfinite(values_uv).all():
                raise ValueError(f"Non-finite data in the window starting at {start / rate:g} s.")

            alert = np.ptp(values_uv) >= THRESHOLD_UV
            recent_windows.append((values_uv, alert))
            if len(recent_windows) < windows_per_block:
                continue

            end = start + subwindow_size
            block_start = end - windows_per_block * subwindow_size
            baseline_power = alpha_power_uv2(
                np.concatenate([samples for samples, _ in recent_windows]), rate
            )
            # The gated calculation requires four consecutive unflagged windows.
            gated_power = None if any(flag for _, flag in recent_windows) else baseline_power
            rolling_blocks.append((block_start / rate, end / rate, baseline_power, gated_power))

    print(f"P001 · {CHANNEL} · seuil {THRESHOLD_UV:g} µV par fenêtre de {WINDOW_S:g} s")
    fixed_blocks = rolling_blocks[::windows_per_block]
    report("Enregistrement complet · blocs fixes", fixed_blocks)
    report("Enregistrement complet · blocs glissants (pas de 500 ms)", rolling_blocks)

    demo_blocks = [row for row in fixed_blocks if row[0] >= 70 and row[1] <= 80]
    demo_rolling = [row for row in rolling_blocks if row[0] >= 70 and row[1] <= 80]
    report("Extrait de la démo (70–80 s) · blocs fixes", demo_blocks)
    report("Extrait de la démo (70–80 s) · blocs glissants", demo_rolling)

    for marker_id, label in [(10000, "Repos annoté"), (10006, "Consigne de clignement")]:
        task_blocks = [
            row for row in rolling_blocks
            if any(start <= row[0] and row[1] <= end
                   for start, end in task_intervals[marker_id])
        ]
        report(f"{label} · blocs glissants entièrement dans la période", task_blocks)

    for start, end, _, gated_power in demo_rolling:
        if gated_power is not None:
            print(f"\nMesure alpha disponible dans l’extrait : "
                  f"{start:g}–{end:g} s | {gated_power:.2f} µV²")


if __name__ == "__main__":
    main()
