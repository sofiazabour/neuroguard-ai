import numpy as np
from neuroguard_detector import features, calibrate, predict_candidates
from process import task_for_window

def test_detector_calibration_and_candidates():
    fs=1000
    t=np.arange(500)/fs
    x=np.stack([10*np.sin(2*np.pi*3*t), 8*np.sin(2*np.pi*3*t),
                5*np.sin(2*np.pi*40*t),5*np.sin(2*np.pi*40*t)])
    names=['Fp1','Fp2','F9','F10']
    baseline=features(x,names,fs)
    cal=calibrate([baseline]*30)
    assert not any(predict_candidates(baseline,cal)[k] for k in
                   ('ocular_candidate','muscle_candidate','movement_candidate'))
    assert predict_candidates(features(x*np.array([[5],[5],[1],[1]]),names,fs),cal)['ocular_candidate']
    assert predict_candidates(features(x*np.array([[1],[1],[5],[5]]),names,fs),cal)['muscle_candidate']

def test_task_interval_alignment():
    periods=[(10.,70.,'rest'),(75.,80.,'blink')]
    assert task_for_window(75.,75.5,periods)=='blink'
    assert task_for_window(74.75,75.25,periods)=='unlabeled'
