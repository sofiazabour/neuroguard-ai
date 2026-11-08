"""Small numerical integration tests using generated EEG, without dataset access."""

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pyedflib import FILETYPE_BDFPLUS, highlevel

from process import alpha_power, load_config, process


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.out = self.root / "out"

    def write_recording(self):
        folder = self.data / "CURATED" / "P001"
        folder.mkdir(parents=True)
        bdf = folder / "P001_day-1_task-resting-state_multimodal.bdf"
        rate = 128
        times = np.arange(rate * 6) / rate
        signals = [10 * np.sin(2 * np.pi * hz * times) for hz in (10, 20)]
        headers = [highlevel.make_signal_header(
            label, dimension="uV", sample_frequency=rate,
            physical_min=-100, physical_max=100,
            digital_min=-8388608, digital_max=8388607,
        ) for label in ("O1", "O2")]
        highlevel.write_edf(str(bdf), signals, headers, file_type=FILETYPE_BDFPLUS)
        bdf.with_suffix(".record_map").write_text(json.dumps({
            "TotalSamples": len(times),
            "stream_map": {"synthetic": {"EEG": {
                "channel_indices": [0, 1],
                "correction": {"RecordType": "EEG", "Channels": [
                    {"name": "O1", "units": "uV"},
                    {"name": "O2", "units": "uV"},
                ]},
            }}},
        }), encoding="utf-8")
        return bdf

    def test_alpha_recovers_known_sine_power(self):
        table = alpha_power(self.write_recording()).set_index("channel")
        self.assertAlmostEqual(table.loc["O1", "band_uV2"], 50.0, delta=0.1)
        self.assertLess(table.loc["O2", "band_uV2"], 0.01)

    def test_custom_band_and_manifest_through_cli(self):
        self.write_recording()
        config = self.root / "custom.toml"
        config.write_text(
            'dataset_variant = "CURATED"\nmax_recordings = 1\nband_hz = [18, 22]\n',
            encoding="utf-8",
        )
        result = subprocess.run([
            sys.executable, str(Path(__file__).resolve().parents[1] / "process.py"),
            "--data", str(self.data), "--out", str(self.out), "--config", str(config),
        ], cwd=self.root, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = pd.read_csv(self.out / "band_power.tsv", sep="\t")
        self.assertEqual(summary.loc[0, "strongest_channel"], "O2")
        manifest = json.loads((self.out / "run.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["config"]["band_hz"], [18, 22])
        self.assertEqual(manifest["recordings_processed"], 1)
        self.assertEqual(len(pd.read_csv(self.out / "band_power_by_channel.tsv", sep="\t")), 2)

    def test_no_matching_recordings_has_clear_error_and_no_output(self):
        with self.assertRaisesRegex(FileNotFoundError, "No resting-state BDF"):
            process(self.data, self.out)
        self.assertFalse(self.out.exists())

    def test_missing_sidecar_is_reported_before_loading(self):
        bdf = self.write_recording()
        bdf.with_suffix(".record_map").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing record map"):
            process(self.data, self.out)
        self.assertFalse(self.out.exists())

    def test_previous_results_are_preserved(self):
        self.write_recording()
        self.out.mkdir()
        sentinel = self.out / "previous-result.txt"
        sentinel.write_text("keep me", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Output directory must be empty"):
            process(self.data, self.out)
        self.assertEqual(sentinel.read_text(), "keep me")

    def test_preprocessed_variant_and_recording_limit(self):
        bdf = self.write_recording()
        destination = self.data / "PREPROCESSED"
        shutil.copytree(self.data / "CURATED", destination)
        second = destination / "P002" / bdf.name.replace("P001", "P002")
        second.parent.mkdir()
        shutil.copyfile(bdf, second)
        shutil.copyfile(bdf.with_suffix(".record_map"), second.with_suffix(".record_map"))
        config = self.root / "preprocessed.toml"
        config.write_text(
            'dataset_variant = "PREPROCESSED"\nmax_recordings = 1\nband_hz = [8, 12]\n',
            encoding="utf-8",
        )
        process(self.data, self.out, config_path=config)
        manifest = json.loads((self.out / "run.json").read_text())
        self.assertEqual(manifest["recordings_found"], 2)
        self.assertEqual(manifest["recordings_processed"], 1)
        self.assertTrue(manifest["inputs"][0]["path"].startswith("PREPROCESSED/P001/"))

    def test_invalid_configuration_is_rejected(self):
        for setting in (
            'dataset_variant = "UNKNOWN"\nmax_recordings = 3\nband_hz = [8, 12]\n',
            'dataset_variant = "CURATED"\nmax_recordings = 0\nband_hz = [8, 12]\n',
            'dataset_variant = "CURATED"\nmax_recordings = true\nband_hz = [8, 12]\n',
            'dataset_variant = "CURATED"\nmax_recordings = 3\nband_hz = [12, 8]\n',
            'dataset_variant = "CURATED"\nmax_recordings = 3\nband_hz = [8, nan]\n',
            'dataset_variant = "CURATED"\nmax_recordings = 3\nband_hz = [8, 12]\ntypo = 1\n',
        ):
            with self.subTest(setting=setting):
                config = self.root / "invalid.toml"
                config.write_text(setting, encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_config(config)


if __name__ == "__main__":
    unittest.main()
