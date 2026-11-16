"""NeuroGuard exploratory multi-type EEG artifact detection (HABS entry point).

Usage: python process.py --data data --out out/neuroguard
Output: per-window candidates, task-interval diagnostics, and run manifest.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from neuroguard_detector import features, calibrate, predict_candidates

TO_UV = {'V':1e6,'mV':1e3,'uV':1.,'µV':1.,'μV':1.,'nV':1e-3}
TASKS = {10000:'rest',10006:'blink',10008:'horizontal_eye',10010:'vertical_eye',
         10012:'head_rotation',10014:'head_tilting',10016:'jaw_clenching',10018:'swallowing'}
EEG = set('Fp1 Fp2 AF7 AF3 AF4 AF8 F9 F7 F5 F3 F1 Fz F2 F4 F6 F8 F10 FT7 FC5 FC3 FC1 FCz FC2 FC4 FC6 FT8 T9 T7 C5 C3 C1 C2 C4 C6 T8 T10 TP7 CP5 CP3 CP1 CP2 CP4 CP6 TP8 P9 P7 P5 P3 P1 Pz P2 P4 P6 P8 P10 PO7 PO5 PO3 POz PO4 PO6 PO8 O1 O2'.split())


def intervals(metadata, duration):
    seg = metadata['dataset_export']['segments']['artefact']
    origin = float(seg['metadata_time_origin_s'])
    out = []
    for e in metadata.get('events', []):
        task = TASKS.get(int(e['marker_id']))
        if task is None or e.get('duration_s') is None:
            continue
        start = float(e['onset_s']) - origin
        end = start + float(e['duration_s'])
        if end > 0 and start < duration:
            out.append((max(0.,start), min(duration,end),task))
    return sorted(out)


def task_for_window(start, end, periods):
    matching = [name for a,b,name in periods if start >= a and end <= b]
    return matching[0] if len(matching) == 1 else 'unlabeled'


def analyze(bdf, metadata, participant, window_s=0.5):
    from pyedflib import EdfReader
    rows = []
    with EdfReader(str(bdf)) as reader:
        labels = reader.getSignalLabels()
        indices = [i for i,c in enumerate(labels) if c in EEG]
        channels = [labels[i] for i in indices]
        if not any(c in channels for c in ('Fp1','Fp2','AF7','AF3','AF4','AF8')):
            raise ValueError(f'No frontal EEG channels in {bdf}')
        rates = [float(reader.getSampleFrequency(i)) for i in indices]
        if max(rates)-min(rates) > 1e-6:
            raise ValueError('EEG channels have different sample rates')
        sfreq = rates[0]
        size = int(round(sfreq * window_s))
        n = min(int(reader.getNSamples()[i]) for i in indices)
        periods = intervals(metadata, n/sfreq)
        calibration = []
        for start in range(0, n-size+1, size):
            stop = start+size
            a,b = start/sfreq,stop/sfreq
            task = task_for_window(a,b,periods)
            signal = []
            for i in indices:
                unit = reader.getPhysicalDimension(i).strip()
                if unit not in TO_UV:
                    raise ValueError(f'Unknown EEG unit {unit!r} in {bdf.name}')
                signal.append(reader.readSignal(i,start,size)*TO_UV[unit])
            t0=perf_counter()
            f=features(np.asarray(signal),channels,sfreq)
            elapsed=(perf_counter()-t0)*1000
            row={'participant':participant,'start_s':round(a,4),'end_s':round(b,4),
                 'task_interval':task,'feature_ms':round(elapsed,3),**f}
            rows.append(row)
            # The initial rest interval only, before the first non-rest task.
            nonrest_starts=[x for x,y,name in periods if name != 'rest']
            first_task=min(nonrest_starts) if nonrest_starts else float('inf')
            if task == 'rest' and b <= first_task:
                calibration.append(f)
    baseline=calibrate(calibration)
    for row in rows:
        row.update(predict_candidates(row,baseline))
    return rows,baseline


def process(data_dir: Path, out_dir: Path) -> None:
    data_dir,out_dir=Path(data_dir),Path(out_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        raise ValueError(f'Output directory must be empty: {out_dir}')
    # Small, deterministic default for bounded HABS compute; override via env only locally if needed.
    recordings=sorted(data_dir.glob('CURATED/P*/*_task-artefact_*.bdf'))
    if not recordings:
        raise FileNotFoundError('No CURATED artefact BDF recordings found')
    selected=recordings[:3]
    out_dir.mkdir(parents=True,exist_ok=True)
    all_rows=[]; baselines=[]
    started=perf_counter()
    for bdf in selected:
        participant=bdf.parent.name
        metadata_path=bdf.parent/f'{participant}_day-1_metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        rows,base=analyze(bdf,metadata,participant)
        all_rows.extend(rows)
        baselines.append({'participant':participant,**base})
        print(f'{participant}: {len(rows)} windows',flush=True)
    result=pd.DataFrame(all_rows)
    result.to_csv(out_dir/'artifact_windows.csv',index=False)
    pd.DataFrame(baselines).to_csv(out_dir/'calibration.csv',index=False)
    keys=['ocular_candidate','muscle_candidate','movement_candidate']
    summary=(result[result.task_interval!='unlabeled']
             .groupby(['participant','task_interval'],as_index=False)[keys].mean())
    summary.to_csv(out_dir/'task_interval_summary.csv',index=False)
    manifest={'status':'completed','timestamp_utc':datetime.now(timezone.utc).isoformat(),
              'participants':[b.parent.name for b in selected],
              'recordings_found':len(recordings),'recordings_processed':len(selected),
              'windows':len(result),'elapsed_seconds':round(perf_counter()-started,2),
              'window_seconds':0.5,'method':'provisional_calibrated_heuristics',
              'note':'Task intervals are NOT window-level artifact ground truth. Candidate outputs are not validated.',
              'outputs':['artifact_windows.csv','calibration.csv','task_interval_summary.csv']}
    (out_dir/'run.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    process(args.data,args.out)

if __name__=='__main__':
    main()
