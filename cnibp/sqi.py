import numpy as np
from scipy.signal import find_peaks

def signal_quality_index(ppg, fs):
    x = np.asarray(ppg).astype(np.float32)
    if np.std(x) < 1e-6:  # flat
        return 0.0
    x = (x - x.mean()) / (x.std() + 1e-8)

    # peak count plausibility (heart rate bounds 40–200 bpm)
    peaks, _ = find_peaks(x, distance=int(0.25*fs))
    duration_s = len(x) / fs
    bpm = 60.0 * len(peaks) / max(duration_s, 1e-6)
    hr_ok = 40 <= bpm <= 200

    # morphology: skew/kurt bounds
    skew = ((x**3).mean())
    kurt = ((x**4).mean())
    skew_ok = -1.5 <= skew <= 1.5
    kurt_ok = 1.5 <= kurt <= 6.5

    # peak-to-peak interval variance
    if len(peaks) >= 3:
        ipi = np.diff(peaks) / fs
        rr_cv = np.std(ipi) / (np.mean(ipi) + 1e-6)
        ipi_ok = rr_cv < 0.6
    else:
        ipi_ok = False

    score = 0.25*hr_ok + 0.25*skew_ok + 0.25*kurt_ok + 0.25*ipi_ok
    return float(score)
