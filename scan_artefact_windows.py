import json
from pathlib import Path
from time import perf_counter, sleep
import numpy as np
import pandas as pd

from process import TO_VOLTS
from starting_notebooks.bdf_loader import load_bdf

WINDOW_S = 0.5
THRESHOLD_UV = 200.0
CHANNEL = "Fp1"

# Read the metadata and locate the blink instruction in the BDF timeline.
folder = Path("sample/CURATED/P001")
metadata = json.loads(
    (folder / "P001_day-1_metadata.json").read_text(encoding="utf-8")
)
segment = metadata["dataset_export"]["segments"]["artefact"]
blink = next(event for event in metadata["events"] if event["marker_id"] == 10006)
blink_start = blink["onset_s"] - segment["metadata_time_origin_s"]
blink_end = blink_start + blink["duration_s"]

# Load Fp1, convert it to microvolts, and find the number of samples per window.
raw, signals, channels = load_bdf(folder / segment["file"])
sample_rate = raw.info["sfreq"]
unit = channels.set_index("bdf_label").loc[CHANNEL, "header_unit"]
fp1_uv = signals[CHANNEL].to_numpy() * TO_VOLTS[unit] * 1e6
samples_per_window = round(WINDOW_S * sample_rate)

if not np.isfinite(fp1_uv).all():
    raise ValueError(f"{CHANNEL} contains missing or non-finite values.")

# Compute the peak-to-peak amplitude in each complete, non-overlapping window.
rows = []
for first in range(0, len(fp1_uv) - samples_per_window + 1, samples_per_window):
    stop = first + samples_per_window
    start_s = first / sample_rate
    end_s = stop / sample_rate
    values = fp1_uv[first:stop]
    amplitude_uv = values.max() - values.min()

    rows.append({
        "start_s": start_s,
        "end_s": end_s,
        "peak_to_peak_uv": amplitude_uv,
        "alert": amplitude_uv >= THRESHOLD_UV,
        "inside_blink_instruction": blink_start <= start_s and end_s <= blink_end,
    })

windows = pd.DataFrame(rows)

# Save the CSV with rounded amplitudes. Threshold comparisons use full precision.
output = Path("out/fp1-windows.csv")
output.parent.mkdir(exist_ok=True)
windows.round({"peak_to_peak_uv": 2}).to_csv(output, index=False)

# Select windows fully inside either of P001's two eyes-open rest periods.
at_rest = (
    ((windows["start_s"] >= 9.928) & (windows["end_s"] <= 69.928))
    | ((windows["start_s"] >= 145.533) & (windows["end_s"] <= 205.533))
)
during_blink = windows["inside_blink_instruction"]

# Compare all three thresholds on the same set of windows.
print(f"\nFenêtres complètes : {len(windows)}")
print("Comparaison des seuils")
for threshold in (150, 200, 300):
    alerts = windows["peak_to_peak_uv"] >= threshold
    print(
        f"{threshold} µV | "
        f"total : {alerts.sum()}/{len(windows)} | "
        f"consigne : {(alerts & during_blink).sum()}/{during_blink.sum()} | "
        f"repos : {(alerts & at_rest).sum()}/{at_rest.sum()}"
    )

print(f"\nTableau enregistré dans {output} (colonne alert à {THRESHOLD_UV:g} µV)")




# Baseline observation on P001 (Fp1, 500 ms windows):
# Raising the threshold from 150 to 300 µV reduced alerts during annotated
# rest periods from 53/238 to 11/238. It also reduced alerts during the blink
# instruction from 6/10 to 0/10. At 200 µV, the counts were 40/238 and 4/10.
# The annotations describe task periods. Individual blinks and artifacts
# during rest would need to be verified before estimating detection accuracy.

# Channel metadata: P001_day-1_task-artefact_multimodal.record_map
# Gaze: stored 166 Hz, effective 166.358 Hz, onset 0.002 s

# Fenêtres complètes : 424
# Comparaison des seuils
# 150 µV | total : 153/424 | consigne : 6/10 | repos : 53/238
# 200 µV | total : 112/424 | consigne : 4/10 | repos : 40/238
# 300 µV | total : 37/424 | consigne : 0/10 | repos : 11/238

# Tableau enregistré dans out/fp1-windows.csv (colonne alert à 300 µV)

# Replay ten seconds and make a decision when each window becomes available.
replay_start_s = 70.0
replay_end_s = 80.0
first_sample = round(replay_start_s * sample_rate)
last_sample = round(replay_end_s * sample_rate)

print(f"\nSimulation de 70 à 80 s — seuil : {THRESHOLD_UV:g} µV")
clock_start = perf_counter()

for first in range(first_sample, last_sample, samples_per_window):
    stop = first + samples_per_window

    # Wait until the last sample of this window would have arrived.
    arrival_s = (stop - first_sample) / sample_rate
    sleep(max(0.0, clock_start + arrival_s - perf_counter()))

    values = fp1_uv[first:stop]
    amplitude_uv = values.max() - values.min()
    status = "ALERTE" if amplitude_uv >= THRESHOLD_UV else "Sous le seuil"

    print(
        f"{first / sample_rate:5.1f}–{stop / sample_rate:5.1f} s | "
        f"{amplitude_uv:6.1f} µV | {status}",
        flush=True,
    )