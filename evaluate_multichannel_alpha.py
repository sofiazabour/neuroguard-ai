"""Compare single-channel and per-channel alert rules on P001."""

import json

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pyedflib import EdfReader

from evaluate_alpha_gate import BLOCK_S, FOLDER, THRESHOLD_UV, WINDOW_S, alpha_power_uv2
from process import TO_VOLTS

POSTERIOR_CHANNELS = ("O1", "O2", "POz")
MIN_POSTERIOR_CHANNELS = 2


def report(label, selected, fp1_ok, local_ok, channels_ok):
    total = int(selected.sum())
    print(f"\n{label} · {total} blocs")
    if total == 0:
        return

    global_ok = fp1_ok & local_ok
    for name, available in [
        ("Fp1 seul (test précédent)", fp1_ok),
        ("Alpha postérieur avec veto Fp1", global_ok),
        ("Alpha postérieur sans veto Fp1", local_ok),
        ("Blocs récupérés en retirant le veto Fp1", local_ok & ~fp1_ok),
    ]:
        count = int((available & selected).sum())
        print(f"  {name} : {count}/{total} ({count / total:.1%})")

    counts = channels_ok[:, selected].sum(axis=1)
    print("  Canaux postérieurs sous le seuil sur les 2 s : " + " | ".join(
        f"{label} {int(count)}/{total}"
        for label, count in zip(POSTERIOR_CHANNELS, counts)
    ))


def main():
    metadata = json.loads(
        (FOLDER / "P001_day-1_metadata.json").read_text(encoding="utf-8")
    )
    segment = metadata["dataset_export"]["segments"]["artefact"]

    with EdfReader(str(FOLDER / segment["file"])) as reader:
        labels = reader.getSignalLabels()
        indices = [labels.index(name) for name in ("Fp1", *POSTERIOR_CHANNELS)]
        rate = reader.getSampleFrequency(indices[0])
        if any(reader.getSampleFrequency(index) != rate for index in indices):
            raise ValueError("The selected EEG channels have different sampling rates.")
        window_size = round(rate * WINDOW_S)
        n_samples = min(segment["n_samples"], *(reader.getNSamples()[i] for i in indices))
        n_windows = n_samples // window_size
        windows_per_block = round(BLOCK_S / WINDOW_S)
        if n_windows < windows_per_block:
            raise ValueError("The recording is too short for an alpha block.")

        signals_uv = np.stack([
            reader.readSignal(index, 0, n_windows * window_size)
            * TO_VOLTS[reader.getPhysicalDimension(index)] * 1e6
            for index in indices
        ])

    # Each electrode keeps its own alert history, in the original time order.
    windows = signals_uv.reshape(len(indices), n_windows, window_size)
    window_ok = np.isfinite(windows).all(axis=-1) & (np.ptp(windows, axis=-1) < THRESHOLD_UV)
    block_ok = sliding_window_view(window_ok, windows_per_block, axis=-1).all(axis=-1)
    fp1_ok = block_ok[0]
    channels_ok = block_ok[1:]
    local_ok = channels_ok.sum(axis=0) >= MIN_POSTERIOR_CHANNELS
    recovered = local_ok & ~fp1_ok

    starts = np.arange(fp1_ok.size) * window_size / rate
    ends = starts + windows_per_block * window_size / rate
    demo = (starts >= 70) & (ends <= 80)

    print(f"P001 · seuil {THRESHOLD_UV:g} µV sur {WINDOW_S:g} s · "
          f"blocs glissants de {BLOCK_S:g} s")
    print(f"Alpha postérieur disponible sur au moins {MIN_POSTERIOR_CHANNELS} "
          f"canaux parmi {', '.join(POSTERIOR_CHANNELS)}.")
    report("Enregistrement complet", np.ones(starts.shape, dtype=bool),
           fp1_ok, local_ok, channels_ok)
    report("Extrait de la démo (70–80 s)", demo, fp1_ok, local_ok, channels_ok)

    origin = segment["metadata_time_origin_s"]
    for marker, name in [(10000, "Repos annoté"), (10006, "Consigne de clignement")]:
        selected = np.zeros(starts.shape, dtype=bool)
        for event in metadata["events"]:
            duration = event["duration_s"]
            if event["marker_id"] == marker and duration is not None and duration > 0:
                start = event["onset_s"] - origin
                selected |= (start <= starts) & (ends <= start + duration)
        report(name, selected, fp1_ok, local_ok, channels_ok)

    # Compute per-channel power for accepted blocks using contiguous samples.
    posterior_power = np.full(channels_ok.shape, np.nan)
    block_size = windows_per_block * window_size
    for channel, values in enumerate(signals_uv[1:]):
        for index in np.flatnonzero(local_ok & channels_ok[channel]):
            start = int(index) * window_size
            posterior_power[channel, index] = alpha_power_uv2(
                values[start:start + block_size], rate
            )

    print("\nPuissance alpha par canal dans les blocs récupérés de la démo")
    recovered_indices = np.flatnonzero(demo & recovered)
    if not recovered_indices.size:
        print("  Aucun bloc récupéré dans cet extrait.")
    for index in recovered_indices:
        powers = [
            f"{name} {power:.2f} µV²"
            for name, power in zip(POSTERIOR_CHANNELS, posterior_power[:, index])
            if np.isfinite(power)
        ]
        print(f"  {starts[index]:g}–{ends[index]:g} s | {' | '.join(powers)}")


if __name__ == "__main__":
    main()
