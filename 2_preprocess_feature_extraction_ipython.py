# %%
import pickle
import numpy as np
from scipy.signal import welch as psd_welch
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from scipy.signal import butter as butter_filter
from scipy.signal import sosfilt
from scipy.signal import ricker   # for our wavelet transform

# Change location of raw data and save data files here
raw_data_path = "F:/sleep-edf-database-expanded-1.0.0/sleep-cassette/all_data.pkl"
subband_data_pickle_path = "F:/sleep-edf-database-expanded-1.0.0/sleep-cassette/subband_data.pkl"
subband_data_csv_path = "F:/sleep-edf-database-expanded-1.0.0/sleep-cassette/subband_data.csv"

with open(raw_data_path, "rb") as fp:
    data = pickle.load(fp)

# assign slice for each epoch for our patient
# we need to create our concatentated arrays dynamically per patient
# this is because all of the data takes up 2GB with float32
# if we are splitting it into 6 subbands, that would take 12GB 
# doing it per patient is more memory efficient
patient_data_running_len = {}
patient_data_indices = {}
patient_data_x = {}
patient_data_y = {}
for pid, y, x_data in data:
    Ydata = patient_data_y.setdefault(pid, [])
    Xdata = patient_data_x.setdefault(pid, [])
    indices = patient_data_indices.setdefault(pid, [])
    # create index
    start_idx = patient_data_running_len.setdefault(pid, 0)
    end_idx = start_idx+len(x_data)
    index = slice(start_idx,end_idx)
    # add entries
    indices.append(index)
    Xdata.append(x_data)
    Ydata.append(y)
    # update our index
    patient_data_running_len[pid] = end_idx

# 10th order butterworth filter
Nfilter = 10
Fs = 100

band_filters_sos = [
    
    # Fitler for each subband
    butter_filter(Nfilter, (0.5, 4), btype="bandpass", fs=Fs, output='sos'),
    butter_filter(Nfilter, (4,8),    btype="bandpass", fs=Fs, output='sos'),
    butter_filter(Nfilter, (8,12),   btype="bandpass", fs=Fs, output='sos'),
    butter_filter(Nfilter, (12,16),  btype="bandpass", fs=Fs, output='sos'),
    butter_filter(Nfilter, (16,30),  btype="bandpass", fs=Fs, output='sos'),
    butter_filter(Nfilter, (30,40),  btype="bandpass", fs=Fs, output='sos'),
]

# Fitler inclusive of all bands
allband_filter_sos = butter_filter(Nfilter, (0.5, 40), btype="bandpass", fs=Fs, output='sos')

wavelet_widths = np.arange(0, 10)
wavelet_fir_filters = [ricker(2**(i+3), 2**i) for i in wavelet_widths]

# energy features
def get_energy(x):
    return np.sum(np.abs(x))

# get power of our signal
def get_power(x):
    X = np.abs(x)**2
    N = x.shape[0]
    Y = (1/(2*N+1))*np.sum(X)
    return Y

# entropy features (in progress)
# https://en.wikipedia.org/wiki/Sample_entropy
# Unfortunately this takes ~1 secs per epoch to compute so we ignore this feature for now
def get_entropy(x):
    N = len(x)
    m = 2                     # Segment size of 2
    r = 0.2 * np.std(x)       # 20% of std
    xmi = np.array([x[i : i + m] for i in range(N - m)])
    xmj = np.array([x[i : i + m] for i in range(N - m + 1)])
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])
    m += 1
    xm = np.array([x[i : i + m] for i in range(N - m + 1)])
    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    return -np.log(A / B)

# hjorth features
def get_mobility(y):
    dy_dt = np.gradient(y)
    return (np.var(dy_dt)/np.var(y))**0.5

def get_hjorth_features(x):
    dx_dt = np.gradient(x)
    x_activity = np.var(x)
    x_mobility = get_mobility(x)
    x_complexity = get_mobility(dx_dt)/x_mobility
    return (x_activity, x_mobility, x_complexity)

def get_zcr(x):
    sign = x[0]
    zcr = 0
    for point in x:
        if np.sign(point) != sign and np.sign(point) != 0:
            zcr += 1
            sign = np.sign(point)
    return zcr

# we have 6 subbands
# x_subbands.shape = (6, N)
# y_data.shape = (K,)
# indices.shape = (K,)
# K*len(epoch) = N
# divide our subbands into per epoch
def generate_band_epochs(x_bands, y_data, indices, band):
    for idx, y in zip(indices, y_data):
        if band == "subband":
            x_epoch_bands = x_bands[:,idx]
        elif band == "allband":
            x_epoch_bands = x_bands[idx]
        yield (x_epoch_bands, y)
        # print(x_epoch_bands, y)

# for each epoch in our subbands, get our features
# power, entropy, hjorth, std
# and their relative counterparts
# each of these measurements are of shape (Nsubbandsx1) 
# except hjorth, that is (Nsubbands,3), where it is the 3 hjorth measurements
# where Nsubbands = 6
def extract_features_from_subbands(x_subbands, y_data, indices):
    # go through each epoch and extract features
    # generate our per epoch features


    # In this loop, there are 6 subbands per 30s epoch. Each loop covers one subband and its features.
    for x_epoch_subbands, y in generate_band_epochs(x_subbands, y_data, indices, "subband"):
        energy = np.array([get_energy(x) for x in x_epoch_subbands])
        power = np.array([get_power(x) for x in x_epoch_subbands])
        #rel_energy = energy/np.sum(energy)
        rel_power = power/np.sum(power)
        features = (energy, power, rel_power)   # Size 3 for each subband
        # entropy = np.array([get_entropy(x) for x in x_epoch_subbands])
        # std = np.std(x_epoch_subbands, axis=1)
        #features = (energy, entropy, std)
        #print("subband feats: ", features)
        yield (features, y)

def extract_features_from_allband(x_allband, y_data, indices):
    for x_epoch_allband, y in generate_band_epochs(x_allband, y_data, indices, "allband"):
        #print("x_epoch_allband: ", x_epoch_allband)
        activity, mobility, complexity = get_hjorth_features(x_epoch_allband)
        zcr = get_zcr(x_epoch_allband)
        features = (activity, mobility, complexity, zcr)    # Size 2 for the allband
        yield (features, y)

# allocate our numpy arrays to store the time domain data for each patient
# then filter them into subbands and divide into epochs
# for each epoch we do our preprocessing
epoch_data = []

pids = list(patient_data_x.keys())
for pid in tqdm(pids):
# for pid, i in zip(pids, range(1)):    # This line is for debugging purposes
    x_data_refs = patient_data_x[pid]
    y_data = patient_data_y[pid]
    idxs = patient_data_indices[pid]
    
    # allocate our contiguous array for subband filtering
    Ndata = sum((len(x) for x in x_data_refs))
    x_data = np.zeros((Ndata,), dtype=np.float32)
    for idx, x_data_ref in zip(idxs, x_data_refs):
        x_data[idx] = x_data_ref
    #print(f"pid={pid}, sizeof(x_data)={Ndata*4*1e-6}MB")
    
    # normalise per patient
    # a, b = np.min(x_data), np.max(x_data)
    # x_data = (x_data-a) * (1/(b-a))
    x_data = x_data* 1e-6; # convert back to uV
    
    # filter into our subbands
    x_subband_data = []
    x_allband_data = []
    for band_filter_sos in band_filters_sos:
        x_subband = sosfilt(band_filter_sos, x_data)
        x_subband_data.append(x_subband)
    
    x_allband_data = sosfilt(allband_filter_sos, x_data)
    # for wavelet_filter in wavelet_fir_filters:
    #     x_subband = np.convolve(x_data, wavelet_filter, mode='same')
    #     x_subband_data.append(x_subband)
        
    x_subband_data = np.array(x_subband_data)
    x_allband_data = np.array(x_allband_data)
    # for each subband get our features
    for (features, y), (features2, y2) in zip(extract_features_from_subbands(x_subband_data, y_data, idxs), extract_features_from_allband(x_allband_data, y_data, idxs)):
        assert(y == y2)
        feature_vec = [f.flatten() for f in features]
        feature_vec = np.concatenate(feature_vec).flatten()
        feature2_vec = np.array(features2)
        epoch_data.append((pid, *feature_vec, *features2, y))


with open(subband_data_pickle_path, "wb+") as fp:
    pickle.dump(epoch_data, fp)

columns = [
    # Subband features: energy, power, relative power to all power
    "delta_energy", "theta_energy", "alpha_energy", "sigma_energy", "beta_energy", "gamma_energy",
    "delta_power", "theta_power", "alpha_power", "sigma_power", "beta_power", "gamma_power",
    "delta_power_rel", "theta_power_rel", "alpha_power_rel", "sigma_power_rel", "beta_power_rel", "gamma_power_rel",
    "activity", "mobility", "complexity", "zcr"]

columns = ["pid", *columns, "sleep_stage"]

df = pd.DataFrame(epoch_data, columns=columns)
df = df.dropna()
df.to_csv(subband_data_csv_path, index=False)
