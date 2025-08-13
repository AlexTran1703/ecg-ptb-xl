import numpy as np
import matplotlib.pyplot as plt

def segment_ecg_intervals(ecg, intervals, buffer=50):
    """
    Segment PR, QRS, ST, and QT intervals per lead into fixed shapes.

    Parameters
    ----------
    ecg : np.ndarray
        Shape (1, length, 12), single ECG record.
    intervals : dict[list]
        {"P": [list of P indices], "Q": [list of Q indices], "R": [list of R indices], "S": [list of S indices], "T": [list of T indices]}
        Output from detect_pqst_slope
    Returns
    -------
    segments : dict
        Keys: 'PR', 'QRS', 'ST', 'QT'.
        Shapes: PR -> (12, B, 100), QRS -> (12, B, 100),
                ST -> (12, B, 100), QT -> (12, B, 200)
    """
    if ecg.ndim == 3 and ecg.shape[1] == 1000:
        length = ecg.shape[1]
        first_key, first_value = next(iter(intervals.items()))
        n_beats = len(first_value)
        n_leads = ecg.shape[2]
    elif ecg.ndim == 2 and ecg.shape[0] == 1000:
        ecg = ecg[np.newaxis, ...]  # Add batch dimension
        length = ecg.shape[1]
        first_key, first_value = next(iter(intervals.items()))
        n_beats = len(first_value)
        n_leads = ecg.shape[2]
    else:
        raise ValueError("ECG data must be in shape (1, length, num_leads) or (length, num_leads)")

    # Preallocate zero arrays
    PR_arr  = np.zeros((n_leads, n_beats, buffer))
    QRS_arr = np.zeros((n_leads, n_beats, buffer))
    ST_arr  = np.zeros((n_leads, n_beats, buffer))
    QT_arr  = np.zeros((n_leads, n_beats, buffer))
    annotation = {"P": 0,
                  "Q": 1,
                  "R": 2,
                  "S": 3,
                  "T": 4}

    for lead_idx in range(n_leads):
        for beat_idx, beat in enumerate(zip(*intervals.values())):
            P, Q, R, S, T = beat[annotation["P"]], beat[annotation["Q"]], beat[annotation["R"]], beat[annotation["S"]], beat[annotation["T"]]

            # --- PR interval
            PR_seg = ecg[0, P:Q, lead_idx]
            PR_seg = PR_seg[:buffer]  # truncate if longer
            PR_arr[lead_idx, beat_idx, :len(PR_seg)] = PR_seg

            # --- QRS interval
            QRS_seg = ecg[0, Q:S, lead_idx]
            QRS_seg = QRS_seg[:buffer]
            QRS_arr[lead_idx, beat_idx, :len(QRS_seg)] = QRS_seg

            # --- ST interval
            ST_seg = ecg[0, S:T, lead_idx]
            ST_seg = ST_seg[:buffer]
            ST_arr[lead_idx, beat_idx, :len(ST_seg)] = ST_seg

            # --- QT interval
            QT_seg = ecg[0, Q:T, lead_idx]
            QT_seg = QT_seg[:buffer]
            QT_arr[lead_idx, beat_idx, :len(QT_seg)] = QT_seg

    segments = {
        "PR": PR_arr,
        "QRS": QRS_arr,
        "ST": ST_arr,
        "QT": QT_arr,
    }
    np_segments = np.stack((PR_arr, QRS_arr, ST_arr, QT_arr), axis=1)
    # for k, v in segments.items():
    #     print(f"{k}: {v.shape}")
    return segments, np_segments


def segment_ECG_beats(ecg, r_peaks, start_idx=40, end_idx=80, plot={"plot": False, "lead": "All"}):
    """
    Segment ECG beats around R-peaks into fixed-length intervals.
    
    Parameters
    ----------
    ecg : np.ndarray
        Shape (1, length, 12) or (length, 12), single ECG record.
    r_peaks : list
        List of indices where R-peaks are detected.
    start_idx : int
        Number of samples before the R-peak to include in the segment.
    end_idx : int
        Number of samples after the R-peak to include in the segment.
    plot : dict
        Dictionary with keys 'plot' (bool) to enable plotting and 'lead' (int or None) to specify which lead to plot.
    Returns
    -------
    PQRSTs : list
        List of segments for each lead, where each segment is a 1D array of fixed length.
    np_segments : np.ndarray
        Numpy array of shape (num_segments, segment_length, num_leads).
    """
    ecg_leads = {"I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5, "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}
    # The shape of ecg is (1, 1000, 12), so length is 1000 and number of channels is 12
    if ecg.ndim == 3 and ecg.shape[1] == 1000:
        length = ecg.shape[1]
    elif ecg.ndim == 2 and ecg.shape[0] == 1000:
        ecg = ecg[np.newaxis, ...]  # Add batch dimension
        length = ecg.shape[1]
    elif ecg.ndim == 1:
        ecg = ecg[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions
        length = ecg.shape[1]
    else:
        raise ValueError("Unsupported ECG shape.")
    PQRSTs = []
    expected_segment_length = start_idx + end_idx
    if plot["plot"]:
        plt.figure()
    # Iterate through each lead (channel) first
    for lead_idx in range(ecg.shape[2]):  # Loop through each lead (channel)
        segments_for_this_lead = []
        #plt.figure()
        # Iterate through each R-peak and extract segments for this lead
        for i, r_peak in enumerate(r_peaks):
            start_beat = np.max([0, r_peak - start_idx])
            end_beat = np.min([r_peak + end_idx, length])
            if end_beat == length:
                continue
            # Check if we need padding at the start of the segment (if r_peak - start_idx is negative)
            if r_peak - start_idx < 0:
                start_padding = np.zeros(abs(r_peak - start_idx))  # Create padding at the start
                segment = ecg[0, start_beat:end_beat, lead_idx]  # Extract the segment
                segment = np.concatenate((start_padding, segment))  # Add start padding
            else:
                # Extract the segment without start padding if no negative index
                segment = ecg[0, start_beat:end_beat, lead_idx]
            
            # Pad the segment if it's shorter than the expected segment length at the end
            if expected_segment_length > len(segment):
                padding = np.zeros(expected_segment_length - len(segment))  # Create padding
                segment = np.concatenate((segment, padding))  # Pad the segment

            # Make sure the segment has the expected length
            assert len(segment) == expected_segment_length, f"Segment length mismatch: {len(segment)} != {expected_segment_length}"

            segments_for_this_lead.append(segment)
            
            if plot["plot"] and (plot["lead"] == "All" or lead_idx == ecg_leads[plot["lead"]]):
                plt.plot(segment)
        
        # Add the segments for this lead (as a 2D array)
        PQRSTs.append(np.array(segments_for_this_lead))
    if plot["plot"]:
        plt.title(f"ECG Segments for Lead {ecg_leads[plot['lead']]}" if plot["lead"] != "All" else "ECG Segments")
        plt.show()
    return PQRSTs, np.asarray(PQRSTs)

def convert_to_beat_index(intervals, start_idx=40, end_idx=100):
    converted_intervals = {}
    duplicate_R = [i for i in intervals["R"]]
    for key, original_indices in intervals.items():
        converted_intervals[key] = []
        # For each beat, convert the original index to the corresponding beat index
        for i, original_idx in enumerate(original_indices):
            if key == 'Q' or key == 'P' or key == 'R':
                # For Q and S, we need to adjust the index to be within the beat segment
                index_within_beat = intervals[key][i] - intervals['R'][i] + start_idx
            else:
                # For R, S, and T, we can directly use the index
                index_within_beat = intervals[key][i] - duplicate_R[i] + start_idx
            converted_intervals[key].append(index_within_beat)
            
    return converted_intervals
