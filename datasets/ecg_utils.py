from scipy import signal


def lp_filter(fs, fc, data):
    w = fc/(fs/2)
    b, a = signal.butter(4, w, 'low')
    output = signal.filtfilt(b, a, data)
    return output


def bandstop_filter(fs, f0, Q, data):
    w = f0/(fs/2)
    b, a = signal.iirnotch(w, Q)
    output = signal.filtfilt(b, a, data)
    return output


def baseline_fix(fs, data):
    # remove baseline drift in ecg
    nyq_rate = fs / 2.0
    # The desired width of the transition from pass to stop.
    width = 5.0 / nyq_rate
    # The desired attenuation in the stop band, in dB.
    ripple_db = 60.0
    # Compute the order and Kaiser parameter for the FIR filter.
    O, beta = signal.kaiserord(ripple_db, width)
    # The cutoff frequency of the filter.
    cutoff_hz = 4.0
    ###Use firwin with a Kaiser window to create a lowpass FIR filter.###
    taps = signal.firwin(O, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero=False)
    # Use lfilter to filter x with the FIR filter.
    y_filt = signal.lfilter(taps, 1.0, data)
    # yff = scipy.fftpack.fft(y_filt)
    # Plot filtered outputs
    return y_filt


def get_abnormal_r_peaks(ecg, lead_idx=10):
    lead = ecg[lead_idx]

    mov_baseline = [lead.max() for k in range(len(lead))]
    mov_baseline = [x*0.8 for x in mov_baseline]

    window = []
    peaklist = []
    listpos = 0  # We use a counter to move over the different data columns
    for datapoint in lead:
        rollingmean = mov_baseline[listpos]  # Get local mean

        if (datapoint < rollingmean) and (len(window) < 1):  # If no detectable R-complex activity -> do nothing
            listpos += 1

        elif (datapoint > rollingmean):  # If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1

        else:  # If signal drops below local mean -> determine highest point
            maximum = max(window)
            beatposition = listpos - len(window) + (
                window.index(max(window)))  # Notate the position of the point on the X-axis
            peaklist.append(beatposition)  # Add detected peak to list
            window = []  # Clear marked ROI
            listpos += 1

    return peaklist



