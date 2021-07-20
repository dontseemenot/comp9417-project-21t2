import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Set directory of raw data (use data in sleep-cassette folder) and pre-processed data
rawDir = "./data/sleep-edf-database-expanded-1.0.0/sleep-cassette"
dest_path = "./data/preprocessed/all_data.pkl"

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

all_epochs = []
Npatients = len(psg_hyp)

for i, (psgFile, hypFile) in enumerate(psg_hyp):
    pID = psgFile[3:5]    
    print(f"Patient {i:3d}/{Npatients}")
    # get our data into epochs
    raw = mne.io.read_raw_edf(os.path.join(rawDir, psgFile))
    annot = mne.read_annotations(os.path.join(rawDir, hypFile))
    raw.set_annotations(annot, emit_warning = False)
    raw.set_channel_types(mapping)
    # Double check if sampling freq is 100 Hz
    fs = int(raw.info['sfreq'])
    assert(fs == 100)
    assert(raw['EEG Fpz-Cz'][1][100] == 1)
    # remove head and tail of our recording 
    annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)
    raw.set_annotations(annot, emit_warning=False)
    # split into epochs
    events, _ = mne.events_from_annotations(raw, event_id=event_id_mapping, chunk_duration=30.)
    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included
    try:
        # Currently the code breaks if the PSG file does not include at least one of each sleep stage (eg: for pID = 20, there is no sleep stage 3/4). I don't know how to fix this so we just ignore that file.
        epochs = mne.Epochs(raw=raw, events=events, tmin=0., tmax=tmax, baseline=None, picks = ['EEG Fpz-Cz'])
    except Exception as ex:
        print(ex)
    
    # push all our epochs into a list
    epochs.drop_bad()
    Nepochs = len(epochs)
    for i in range(Nepochs):
        epoch = epochs[i]
        classification = epoch.events[0,-1]
        eeg_data = epoch.get_data("EEG Fpz-Cz") * 1e6 # scale to uV (originally Volts)
        eeg_data = np.squeeze(eeg_data)
        all_epochs.append((int(pID), classification, eeg_data))


with open(dest_path, "wb+") as fp:
    pickle.dump(all_epochs, fp)