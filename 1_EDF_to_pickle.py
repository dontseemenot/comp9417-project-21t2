import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Set directory of raw data (use data in sleep-cassette folder) and pre-processed data
rawDir = "F:\sleep-edf-database-expanded-1.0.0\sleep-cassette"
dest_path = "F:\sleep-edf-database-expanded-1.0.0\sleep-cassette\\all_data.pkl"

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

annotation_desc_2_event_id = {'Sleep stage W': 0,
                              'Sleep stage 1': 1,
                              'Sleep stage 2': 2,
                              'Sleep stage 3': 3,
                              'Sleep stage 4': 3,
                              'Sleep stage R': 4}

event_id = {'Sleep stage W': 0,
        'Sleep stage 1': 1,
        'Sleep stage 2': 2,
        'Sleep stage 3/4': 3,
        'Sleep stage R': 4}


all_epochs = []
Npatients = len(psg_hyp)

# Keep to 30s duration
epoch_duration = 30.0

for i, (psgFile, hypFile) in enumerate(psg_hyp):
    pID = psgFile[3:5]    
    print(f"Patient {i:3d}/{Npatients}")
    # get our data into epochs
    raw = mne.io.read_raw_edf(os.path.join(rawDir, psgFile))
    annot = mne.read_annotations(os.path.join(rawDir, hypFile))
    raw.set_annotations(annot, emit_warning = False)
    raw.set_channel_types(mapping)
    # Double check if sampling freq is really 100 Hz, because sometimes it may not be
    fs = int(raw.info['sfreq'])
    assert(fs == 100)
    assert(raw['EEG Fpz-Cz'][1][100] == 1)
    # remove head and tail of our recording 
    annot.crop(annot[1]['onset'] - 30 * 60, annot[-2]['onset'] + 30 * 60)
    raw.set_annotations(annot, emit_warning=False)
    # split into epochs
    events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=epoch_duration)
    tmax = epoch_duration - 1. / raw.info['sfreq']  # tmax in included
    try:
        # Currently the code breaks if the PSG file does not include at least one of each sleep stage (eg: for pID = 20, there is no sleep stage 3/4). I don't know how to fix this so we just ignore that file.
        epochs = mne.Epochs(raw=raw, events=events, tmin=0., tmax=tmax, baseline=None, picks = ['EEG Fpz-Cz'])
    except Exception as ex:
        print(ex)
        print(f'Files {psgFile} {hypFile} ignored because an instance of a particular sleep stage was not found')
    
    # Push all our epochs into a list. I don't believe any epochs are designated as 'bad', but we will resolve this issue in the preprocessing file.
    epochs.drop_bad()
    
    epochs_data = epochs.get_data("EEG Fpz-Cz")
    epochs_data = np.squeeze(epochs_data)
    epochs_y = epochs.events[:,-1].flatten()
    Nepochs = epochs_data.shape[0]
    
    for i in range(Nepochs):
        all_epochs.append((int(pID), epochs_y[i], epochs_data[i,:].flatten()))

with open(dest_path, "wb+") as fp:
    pickle.dump(all_epochs, fp)