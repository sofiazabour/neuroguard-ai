import argparse
import json
from pathlib import Path

import numpy as np
from pyedflib import EdfReader

from process import TO_VOLTS

FOLDER = Path(__file__).resolve().parent / "sample" / "CURATED" / "P001"
CHANNEL = "Fp1"
WINDOW_S = 0.5
THRESHOLD_UV = 200.0
START_S = 70.0
END_S = 80.0


def main():
    parser = argparse.ArgumentParser(description="Visual P001 EEG replay.")
    parser.add_argument("--save", type=Path, help="Save the final frame as a PNG.")
    args = parser.parse_args()

    import matplotlib

    if args.save:
        matplotlib.use("Agg")

    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Patch

    metadata = json.loads(
        (FOLDER / "P001_day-1_metadata.json").read_text(encoding="utf-8")
    )
    segment = metadata["dataset_export"]["segments"]["artefact"]
    blink = next(event for event in metadata["events"] if event["marker_id"] == 10006)
    blink_start = blink["onset_s"] - segment["metadata_time_origin_s"]
    blink_end = blink_start + blink["duration_s"]

    with EdfReader(str(FOLDER / segment["file"])) as reader:
        channel_index = reader.getSignalLabels().index(CHANNEL)
        rate = reader.getSampleFrequency(channel_index)
        to_uv = TO_VOLTS[reader.getPhysicalDimension(channel_index)] * 1e6
        first = round(START_S * rate)
        stop = round(END_S * rate)
        size = round(WINDOW_S * rate)

        if stop > reader.getNSamples()[channel_index]:
            raise ValueError("The replay interval extends beyond the recording.")

        n_windows = (stop - first) // size
        time_s = (first + np.arange(n_windows * size)) / rate
        raw_uv = np.full(time_s.shape, np.nan)
        forwarded_uv = np.full(time_s.shape, np.nan)

        fig, (raw_ax, forwarded_ax) = plt.subplots(
            2, 1, sharex=True, sharey=True, figsize=(11, 6), layout="constrained"
        )
        if not args.save and type(fig.canvas) is FigureCanvasAgg:
            raise SystemExit("Affichage indisponible. Lance la démo avec --save demo.png.")

        raw_ax.set(title="P001 · Fp1 · Signal brut", ylabel="Amplitude (µV)")
        forwarded_ax.set(
            title="Flux pour l’analyse · un blanc = une fenêtre écartée",
            xlabel="Temps dans l’enregistrement (s)",
            ylabel="Amplitude (µV)",
            xlim=(START_S, END_S),
        )
        blink_band = raw_ax.axvspan(
            blink_start, blink_end, color="orange", alpha=0.15,
            label="Consigne de clignement",
        )
        alert_band = Patch(color="tab:red", alpha=0.25, label="Fenêtre écartée")
        raw_ax.legend(handles=[blink_band, alert_band], loc="upper right")
        raw_line, = raw_ax.plot(time_s, raw_uv, color="black", linewidth=0.8)
        forwarded_line, = forwarded_ax.plot(
            time_s, forwarded_uv, color="tab:blue", linewidth=0.8
        )
        status = fig.suptitle(f"En attente de la première fenêtre · seuil {THRESHOLD_UV:g} µV")
        alert_count = 0

        def update(index):
            nonlocal alert_count
            start = index * size
            end = start + size
            values = reader.readSignal(channel_index, first + start, size) * to_uv
            amplitude = np.ptp(values)
            raw_uv[start:end] = values

            # Keep flagged windows out of the forwarded stream.
            if amplitude >= THRESHOLD_UV:
                alert_count += 1
                raw_ax.axvspan(time_s[start], (first + end) / rate,
                               color="tab:red", alpha=0.25)
                decision = "ALERTE · fenêtre écartée"
            else:
                forwarded_uv[start:end] = values
                decision = "Fenêtre transmise"

            raw_line.set_ydata(raw_uv)
            forwarded_line.set_ydata(forwarded_uv)
            raw_ax.relim()
            raw_ax.autoscale_view(scalex=False)
            status.set_text(
                f"{decision} · {amplitude:.1f} µV (seuil {THRESHOLD_UV:g}) · "
                f"{alert_count} alertes sur {index + 1} fenêtres"
            )

            if index + 1 == n_windows:
                print(
                    f"{n_windows} fenêtres | {alert_count} écartées | "
                    f"{n_windows - alert_count} transmises",
                    flush=True,
                )

        if args.save:
            for index in range(n_windows):
                update(index)
            fig.savefig(args.save, dpi=150)
        else:
            # Poll one complete window at the nominal 500 ms cadence.
            animation = FuncAnimation(
                fig, update, init_func=lambda: (), frames=n_windows,
                interval=1000 * size / rate, repeat=False,
            )
            plt.show()


if __name__ == "__main__":
    main()
