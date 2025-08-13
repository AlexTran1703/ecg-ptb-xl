import numpy as np

def average_rr_seconds(r_peaks, fs=100):
    """
    Compute average RR interval (in seconds) from R-peak indices.
    """
    r_peaks = np.asarray(r_peaks, dtype=int)
    if r_peaks.size < 2:
        # Fallback: assume ~60 bpm if not enough beats
        return 1.0
    rr_samples = np.diff(r_peaks)
    # Robust average (median) against outliers, convert to seconds
    rr_sec = np.median(rr_samples) / float(fs)
    # Clamp to reasonable physiologic range (30–220 bpm)
    rr_sec = float(np.clip(rr_sec, 60/220.0, 60/30.0))
    return rr_sec



def rr_scaled_durations(avg_rr,
                        base={"P":0.09, "Q":0.04, "S":0.04, "T":0.16},
                        scale={"P":0.9, "Q":0.8, "S":0.8, "T":1.2},
                        clamps={"P":(0.05,0.14), "Q":(0.02,0.08), "S":(0.02,0.08), "T":(0.12,0.30)}):
    """
    Scale P/Q/S/T search-window durations (seconds) from avg RR.
    - base: durations at RR=1.0 s (≈60 bpm)
    - scale: how strongly each wave scales with RR
    - clamps: (min, max) hard limits for each duration
    """
    factor = avg_rr / 1.0  # baseline 1s
    d = {}
    for k in base:
        val = base[k] * factor * scale[k]
        lo, hi = clamps[k]
        d[k] = float(np.clip(val, lo, hi))
    return d