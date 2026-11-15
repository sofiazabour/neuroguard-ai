from pathlib import Path
from time import perf_counter, sleep

import numpy as np
from pyedflib import EdfReader

from process import TO_VOLTS

BDF = (
    Path(__file__).resolve().parent / "sample" / "CURATED" / "P001"
    / "P001_day-1_task-artefact_multimodal.bdf"
)
CHANNEL = "Fp1"
WINDOW_S = 0.5
THRESHOLD_UV = 200.0
START_S = 70.0
END_S = 80.0

processing_times_ms = []
alert_count = 0

with EdfReader(str(BDF)) as reader:
    # Find Fp1 and convert seconds to sample indices.
    channel_index = reader.getSignalLabels().index(CHANNEL)
    sample_rate = reader.getSampleFrequency(channel_index)
    unit = reader.getPhysicalDimension(channel_index)
    to_microvolts = TO_VOLTS[unit] * 1e6
    samples_per_window = round(WINDOW_S * sample_rate)
    first_sample = round(START_S * sample_rate)
    last_sample = round(END_S * sample_rate)

    if last_sample > reader.getNSamples()[channel_index]:
        raise ValueError("The replay interval extends beyond the recording.")

    print(f"Rejeu de {START_S:g} à {END_S:g} s | canal {CHANNEL}", flush=True)
    replay_started = perf_counter()

    for start in range(
        first_sample, last_sample - samples_per_window + 1, samples_per_window
    ):
        stop = start + samples_per_window

        # Wait until the complete window would have arrived.
        arrival_s = (stop - first_sample) / sample_rate
        sleep(max(0.0, replay_started + arrival_s - perf_counter()))

        # Read one window and time the alert decision.
        started = perf_counter()
        values_uv = reader.readSignal(channel_index, start, samples_per_window)
        values_uv = values_uv * to_microvolts
        amplitude_uv = values_uv.max() - values_uv.min()
        alert = amplitude_uv >= THRESHOLD_UV
        elapsed_ms = (perf_counter() - started) * 1000

        processing_times_ms.append(elapsed_ms)
        alert_count += int(alert)
        status = "ALERTE" if alert else "Sous le seuil"
        print(
            f"{start / sample_rate:.1f}–{stop / sample_rate:.1f} s | "
            f"{amplitude_uv:.1f} µV | {status} | {elapsed_ms:.2f} ms",
            flush=True,
        )

print(f"Fenêtres : {len(processing_times_ms)} | Alertes : {alert_count}")
print(
    f"Temps de lecture et de décision (p95) : "
    f"{np.percentile(processing_times_ms, 95):.2f} ms"
)