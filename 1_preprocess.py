'''
preprocess.py
Code for preprocessing the EEG data

Before running code, please download raw data from here
https://www.physionet.org/content/sleep-edfx/1.0.0/#ref1

Code is a modified version of sample code provided in an MNE tutorial:
https://mne.tools/stable/auto_tutorials/clinical/60_sleep.html

COMP9417 project 21T2
Arthur Sze (z5165205) and William Yang (zxxxxxxx)
Contact: z5165205@ad.unsw.edu.au, xxxxxxx
'''
# %%
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.time_frequency import psd_welch

# Set directory of raw data (use data in sleep-cassette folder) and pre-processed data
rawDir = 'F:\\sleep-edf-database-expanded-1.0.0\\sleep-cassette'
destDir = 'E:\\HDD documents\\University\\comp9417\\comp9417-project-21t2'


# %%


def eeg_power_band(epochs):
    """EEG relative power band feature extraction.

    This function takes an ``mne.Epochs`` object and creates EEG features based
    on relative power in specific frequency bands that are compatible with
    scikit-learn.

    Parameters
    ----------
    epochs : Epochs
        The data.

    Returns
    -------
    X : numpy array of shape [n_samples, 5]
        Transformed data.
    """
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.0],
                  "theta": [4.0, 8.0],
                  "alpha": [8.0, 12.0],
                  "sigma": [12.0, 16.0],
                  "beta": [16.0, 30.0],
                  "gamma": [30.0, 40.0]}

    psds, freqs = psd_welch(epochs, picks='eeg', fmin=0.5, fmax=40.)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)
    
# Match each psg file with corresponding hypnogram (sleep stage annotation) file. We only use the 2nd night recording for each patient, which is identified by the '2'.
psg_hyp = []
for file in os.listdir(rawDir):
    if file.endswith("PSG.edf") and file[5] == '2':
        psgTemp = file
    elif file.endswith("Hypnogram.edf") and file[5] == '2':
        if psgTemp[0:5] == file[0:5]:
            psg_hyp.append((psgTemp, file))

rows = []
# psg_hyp = [('SC4001E0-PSG.edf', 'SC4001EC-Hypnogram.edf')]    # For debugging with 1 patient
mapping = {'EOG horizontal': 'eog',
           'Resp oro-nasal': 'resp',
           'EMG submental': 'emg',
           'Temp rectal': 'misc',
           'Event marker': 'misc'}
for psgFile, hypFile in psg_hyp:
    raw = mne.io.read_raw_edf(os.path.join(rawDir, psgFile))

    # Double check if sampling freq is 100 Hz
    fs = int(raw.info['sfreq'])
    assert(fs == 100)
    assert(raw['EEG Fpz-Cz'][1][100] == 1)

    annot = mne.read_annotations(os.path.join(rawDir, hypFile))
    raw.set_annotations(annot, emit_warning = False)
    raw.set_channel_types(mapping)

    # Plotting
    # raw.plot(start=60, duration=60,
    #            scalings=dict(eeg=1e-4, resp=1e3, eog=1e-4, emg=1e-7,
    #                          misc=1e-1))

    annotation_desc_2_event_id = {'Sleep stage W': 0,
                              'Sleep stage 1': 1,
                              'Sleep stage 2': 2,
                              'Sleep stage 3': 3,
                              'Sleep stage 4': 3,
                              'Sleep stage R': 4}
    
    annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)
    raw.set_annotations(annot, emit_warning=False)
    
    events, _ = mne.events_from_annotations(
        raw, event_id=annotation_desc_2_event_id, chunk_duration=30.)

    event_id = {'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3/4': 3,
            'Sleep stage R': 4}

    # plot events
    # fig = mne.viz.plot_events(events, event_id=event_id,
    #                         sfreq=raw.info['sfreq'],
    #                         first_samp=events[0, 0])

    # keep the color-code for further plotting
    stage_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

    # Currently the code breaks if the PSG file does not include at least one of each sleep stage (eg: for pID = 20, there is no sleep stage 3/4). I don't know how to fix this so we just ignore that file.
    try:
        epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0., tmax=tmax, baseline=None, picks = ['EEG Fpz-Cz'])

        print(epochs)

        # Get power
        X = eeg_power_band(epochs)
        X_sum = 0.0
        
        pID = psgFile[3:5]
        for i in range(0, X.shape[0]):
            X_sum = sum(X[i, :])
            rows.append([pID,
                        X[i, 0], X[i, 1], X[i, 2], X[i, 3], X[i, 4], X[i, 5],
                        X[i, 0]/X_sum, X[i, 1]/X_sum, X[i, 2]/X_sum, X[i, 3]/X_sum, X[i, 4]/X_sum, X[i, 5]/X_sum,
                        list(epochs[i].event_id.values())[0]])
    except Exception as e:
        print(e)


# Each row represents a 30s epoch from a patient
# A patient has 100's of epochs
# pID: Patient ID
# X1-6: Features (power levels of delta, theta, alpha, sigma, beta, gamma bands)
# Sleep Stage: Output classification (Wake, S1, S2, S3 + S4, REM as 0, 1, 2, 3, 4 respectively) 
df = pd.DataFrame(data = rows, columns = ['pID',
                                            'delta', 'theta', 'alpha', 'sigma', 'beta', 'gamma',
                                            'delta/all', 'theta/all', 'alpha/all', 'sigma/all', 'beta/all', 'gamma/all',
                                            'Sleep Stage'])
df.to_csv(os.path.join(destDir, 'data.csv'), index = False)

# %%
