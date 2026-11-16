import argparse
import json
from pathlib import Path

import numpy as np
from pyedflib import EdfReader

from process import TO_VOLTS

CHANNEL = "Fp1"
WINDOW_S = 0.5
THRESHOLD_UV = 200.0
TASKS = {10000: "repos", 10006: "clignement"}


def evaluate_recording(metadata_path, segment, events):
    # Convert task times from the full session to the BDF timeline.
    origin = segment["metadata_time_origin_s"]
    intervals = {task: [] for task in TASKS.values()}

    for event in events:
        task = TASKS.get(event["marker_id"])
        duration = event["duration_s"]
        if task is not None and duration is not None and duration > 0:
            start_s = event["onset_s"] - origin
            intervals[task].append((start_s, start_s + duration))

    # Read only Fp1 and convert its values to microvolts.
    with EdfReader(str(metadata_path.parent / segment["file"])) as reader:
        channel = reader.getSignalLabels().index(CHANNEL)
        rate = reader.getSampleFrequency(channel)
        unit = reader.getPhysicalDimension(channel)
        count = min(segment["n_samples"], reader.getNSamples()[channel])
        signal_uv = reader.readSignal(channel, 0, count) * TO_VOLTS[unit] * 1e6

    # Apply the same threshold to every complete 500 ms window.
    size = round(WINDOW_S * rate)
    count_windows = len(signal_uv) // size
    windows = signal_uv[:count_windows * size].reshape(count_windows, size)

    alerts = np.ptp(windows, axis=1) >= THRESHOLD_UV
    starts = np.arange(count_windows) * size / rate
    ends = (np.arange(count_windows) + 1) * size / rate

    results = []
    for task, periods in intervals.items():
        # Count only windows fully inside one of the annotated intervals.
        selected = np.zeros(count_windows, dtype=bool)
        for start_s, end_s in periods:
            selected |= (starts >= start_s) & (ends <= end_s)

        total = int(selected.sum())
        flagged = int((alerts & selected).sum())
        results.append(f"{task} : {flagged}/{total}" if total else f"{task} : N/D")

    print(f"{metadata_path.name} | {' | '.join(results)}")


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=Path, default=Path("sample"))
args = parser.parse_args()

processed = 0
for metadata_path in sorted((args.data / "CURATED").glob("P*/*_metadata.json")):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    segment = metadata["dataset_export"]["segments"].get("artefact")

    if segment is None or not (metadata_path.parent / segment["file"]).is_file():
        continue

    evaluate_recording(metadata_path, segment, metadata["events"])
    processed += 1

if processed == 0:
    parser.exit(
        1, f"Aucun enregistrement artefact trouvé dans {args.data / 'CURATED'}\n"
    )