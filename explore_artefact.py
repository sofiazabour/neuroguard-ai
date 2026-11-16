import json
from pathlib import Path

import matplotlib.pyplot as plt

from process import TO_VOLTS
from starting_notebooks.bdf_loader import load_bdf

# 1. Find the blink instruction and locate it on the BDF timeline.
folder = Path("sample/CURATED/P001")
metadata = json.loads(
    (folder / "P001_day-1_metadata.json").read_text(encoding="utf-8")
)
segment = metadata["dataset_export"]["segments"]["artefact"]
blink = next(event for event in metadata["events"] if event["marker_id"] == 10006)

blink_start = blink["onset_s"] - segment["metadata_time_origin_s"]
blink_end = blink_start + blink["duration_s"]

# 2. Load the recording and convert the Fp1 measurements to microvolts.
_, signals, channels = load_bdf(folder / segment["file"])
time = signals["time_s"]
unit = channels.set_index("bdf_label").loc["Fp1", "header_unit"]
fp1_uv = signals["Fp1"] * TO_VOLTS[unit] * 1e6

# 3. Plot 20 seconds of EEG and shade the blink instruction period.
excerpt = (time >= 65) & (time < 85)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(time[excerpt], fp1_uv[excerpt], color="black", linewidth=0.8)
ax.axvspan(
    blink_start, blink_end, color="orange", alpha=0.2,
    label="Consigne de clignement",
)

# 4. Measure the peak-to-peak amplitude in two 500 ms windows.
for start, label, color in [
    (65.0, "Fenêtre au repos", "tab:blue"),
    (75.5, "Fenêtre pendant la consigne", "tab:red"),
]:
    end = start + 0.5
    values = fp1_uv[(time >= start) & (time < end)]
    peak_to_peak = values.max() - values.min()
    print(f"{label} ({start:g}–{end:g} s) : "
          f"{len(values)} mesures, {peak_to_peak:.1f} µV")
    ax.axvspan(start, end, color=color, alpha=0.35, label=label)

ax.set(xlabel="Temps depuis le début du fichier (s)", ylabel="Fp1 (µV)")
ax.legend()
output = Path("out/first-blink.png")
output.parent.mkdir(exist_ok=True)
fig.savefig(output, dpi=150, bbox_inches="tight")
print(f"Graphique enregistré dans {output}")