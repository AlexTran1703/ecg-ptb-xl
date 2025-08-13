import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from utils.utils import average_rr_seconds, rr_scaled_durations

"""" 
Based on lead II only
"""

def detect_pqst(signal, r_peaks, fs=100, smoothing_filter_config={"window_length": 11, "polyorder": 3}, length={"begin": 0, "end": 1000} , plot=False):
    """
    Low-compute ECG delineation:
    SavitzkyGolay smoothing -> derivative -> adaptive thresholds.
    
    Parameters
    ----------
    signal : 1D np.array
        Raw ECG signal (float).
    r_peaks : list[int]
        Indices of detected R peaks.
    fs : int or float
        Sampling frequency in Hz.
        
    Returns
    -------
    results : dict
        Dictionary with lists for P, Q, S, T peaks (indices).
    """
    signal = np.asarray(signal).flatten()  # Ensure signal is a 1D array
    signal = signal[length["begin"]:length["end"]]  # Slice the signal to the specified length
    # --- Step 1: Smooth for clean derivative
    smoothed = savgol_filter(signal, window_length=smoothing_filter_config["window_length"], polyorder=smoothing_filter_config["polyorder"])

    # --- Step 2: Derivative for slope info
    deriv = np.gradient(smoothed)

    P_waves, Q_waves, S_waves, T_waves = [], [], [], []
    # duration = {"P": 0.3,
    #              "Q": 0.06,
    #              "S": 0.06,
    #              "T": 0.2}
    duration = rr_scaled_durations(average_rr_seconds(r_peaks, fs))
    if plot:
        colors = ["blue", "orange", "purple", "brown"]
        labels = ["P peak", "Q point", "S point", "T peak"]
        shown_labels = set()
        plt.figure(figsize=(12, 6))
        plt.plot(signal, label='Raw ECG Signal')
        plt.plot(smoothed, label='Smoothed Signal', alpha=0.7)
        plt.plot(deriv, label='Derivative', alpha=0.5)
        plt.scatter(r_peaks, signal[r_peaks], color='red', label='R-peaks', zorder=5)
        plt.title('ECG Signal with R-peaks')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
    for r in r_peaks:
        if r < length["begin"] or r > length["end"]:
            break
        r_amp = smoothed[r]

        # --- Step 3: Find Q (max slope before R crossing below alpha*R_amp)
        q_search_start = max(0, r - int(duration["Q"] * fs))  # 50 ms before R
        q_search_end   = r
        if q_search_end > q_search_start:
            q_idx_rel = np.argmin(smoothed[q_search_start:q_search_end])
            q_idx = q_search_start + q_idx_rel
            Q_waves.append(q_idx)
        else:
            q_idx = r
            Q_waves.append(r)

        # --- Step 4: Find S (after R)
        s_search_start = r
        s_search_end = min(len(smoothed), r + int(duration["S"] * fs))
        
        if s_search_end > s_search_start:  # Check if the window is valid
            s_idx_rel = np.argmin(deriv[s_search_start:s_search_end])
            s_idx = s_search_start + s_idx_rel
            S_waves.append(s_idx)
        else:
            s_idx = r
            S_waves.append(r)  # If window is invalid, use R as S

        # --- Step 5: Find P (low amp peak before Q, ~120 ms window)
        p_search_start = max(0, q_idx - int(duration["P"] * fs))
        p_search_end   = q_idx - int(0.04 * fs)  # avoid overlap
        if p_search_end > p_search_start:
            p_idx_rel = np.argmax(smoothed[p_search_start:p_search_end])
            p_idx = p_search_start + p_idx_rel
            P_waves.append(p_idx)
        else:
            p_idx = 1
            P_waves.append(1)

        # --- Step 6: Find T (after S, ~200–400 ms window)
        t_search_start = s_idx + int(0.08 * fs)
        t_search_end   = min(len(smoothed), s_idx + int(duration["T"] * fs))
        if t_search_end > t_search_start:
            t_idx_rel = np.argmax(smoothed[t_search_start:t_search_end])
            t_idx = t_search_start + t_idx_rel
            T_waves.append(t_idx)
        else:
            t_idx = len(signal)-1
            T_waves.append(len(signal)-1)
            
        if plot:
            
            for x, y, color, label in zip(
                [p_idx, q_idx, s_idx, t_idx],
                [smoothed[p_idx], smoothed[q_idx], smoothed[s_idx], smoothed[t_idx]],
                colors,
                labels
            ):
                if label in shown_labels:
                    plt.scatter(x, y, color=color, zorder=5)
                else:
                    plt.scatter(x, y, color=color, label=label, zorder=5)
                    shown_labels.add(label)
    if plot:
        plt.legend()
        plt.show()
    # --- Step 7: Create intervals dictionary

    intervals = {
            "P": P_waves,
            "Q": Q_waves,
            "R": r_peaks,
            "S": S_waves,
            "T": T_waves,
        }
    return intervals