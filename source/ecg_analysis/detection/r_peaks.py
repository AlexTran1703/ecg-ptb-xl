
import numpy as np

"""" 
Based on lead II only
"""

def detect_r_peaks(ecg_, fs=100, length={"begin": 0, "end": 1000}, plot=False):
    """
    Simple R-peak detector using diff, squaring, moving window integration, and adaptive thresholding.
    Parameters:
    ecg_ : array-like
        Input ECG signal.
    fs : int, optional
        Sampling frequency in Hz (default is 100 Hz).
    Returns: indices of detected R-peaks.
    """
    def simple_diff(signal):
        diff = []
        for i in range(1, len(signal)):
            diff.append(signal[i] - signal[i-1])
        return np.array(diff)
    
    ecg_ = np.asarray(ecg_).flatten()  # Ensure ecg_ is a 1D array
    ecg_ = ecg_[length["begin"]:length["end"]]  # Slice the ECG signal

    diff = simple_diff(ecg_)
    diff = np.append(diff, 0)  # Append zero to match original length

    # Step 2: Squaring
    squared = diff ** 2

    # Step 3: Moving window integration
    window_size = int(0.150 * fs)  # 150 ms window
    integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

    # Step 4: Simple adaptive thresholding and refractory period
    threshold = np.mean(integrated) * 1.5
    refractory_period = int(0.10 * fs)  # 250 ms

    peaks = []
    last_peak = -refractory_period

    for i in range(1, len(integrated) - 1):
        if integrated[i] > threshold and integrated[i] > integrated[i - 1] and integrated[i] > integrated[i + 1]:
            if i - last_peak > refractory_period:
                # Local search in raw signal for actual peak in a small window
                window = ecg_[i-10:i+10]
                if len(window) == 20:
                    true_peak = i - 10 + np.argmax(window)
                    peaks.append(true_peak)
                    last_peak = true_peak
    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(ecg_, label='ECG Signal')
        plt.plot(integrated, label='Integrated Signal', alpha=0.5)
        plt.scatter(peaks, ecg_[peaks], color='red', label='Detected R-peaks')
        plt.title('R-peak Detection')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid()
        plt.show()
    return np.array(peaks), integrated, squared, diff