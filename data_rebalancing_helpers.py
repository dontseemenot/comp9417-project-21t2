
# %%
'''
This code does 3 things
1) Balances dataset:
2) Cross validation based on inter-patient method
3) Train on simple classifiers (not yet implemented)
'''
# %%
import numpy as np
import pandas as pd
import math
from sklearn.utils import resample
from scipy.stats import skew
# %%
# Helper functions
def get_sleep_info_per_patient(df):
    pids = df['pid'].unique()
    info = []
    for pid in pids:
    #for pid, i in zip(pids, range(1)): # debugging
        pid_df = df.loc[df['pid'] == pid]
        W = (pid_df.loc[df['sleep_stage'] == 0]).shape[0]
        S1 = (pid_df.loc[df['sleep_stage'] == 1]).shape[0]
        S2 = (pid_df.loc[df['sleep_stage'] == 2]).shape[0]
        SWS = (pid_df.loc[df['sleep_stage'] == 3]).shape[0]
        R = (pid_df.loc[df['sleep_stage'] == 4]).shape[0]
        assert((W + S1 + S2 + SWS + R) == pid_df.shape[0])
        info.append([pid, W, S1, S2, SWS, R])

    return info

def remove_patients_with_missing_stages(df, info):
    bad_patients = [p[0] for p in info if 0 in p[1:None]]
    for pid in bad_patients:
        df = df.drop(df[df.pid == pid].index)
    return df

def remove_extra_patients(df, num_to_remove):
    num_to_remove *= -1
    bad_patients = df['pid'].value_counts().index.tolist()[num_to_remove:None]
    for pid in bad_patients:
        df = df.drop(df[df.pid == pid].index)
    return df

def resample_df(df, info):
    threshold = math.ceil(np.mean([np.mean([epoch[1:None] for epoch in info])]))
    df_new = pd.DataFrame()
    pids = df['pid'].unique()
    for pid in pids:
        df_pid = df.loc[df['pid'] == pid]
        for sleep_stage in range(0, 5):
            df_epoch = df_pid.loc[df_pid['sleep_stage'] == sleep_stage]
            if df_epoch.shape[0] > threshold:
                df_epoch = resample(df_epoch, n_samples = threshold, replace = False, random_state = 42)
            else:
                df_epoch = resample(df_epoch, n_samples = threshold, replace = True,  random_state = 42)
            #print(df_epoch)
            df_new = pd.concat([df_new, df_epoch])
    num_classes = 5
    assert(df_new['sleep_stage'].nunique() == num_classes)    # 5 output classes
    a = [x for x in df_new['sleep_stage'].value_counts()]
    b = threshold * len(pids)
    assert(a == b for a in a)  # Ensure each patient in resampled dataset has equal distribution of sleep stage classes

    return df_new

def transform(X, Y):
    #print(np.min(X))
    assert(np.min(X) > 0)

    X0 = X
    X1 = np.log10(X0)
    X0 = (X0-np.mean(X0, axis=0))/np.std(X0, axis=0)
    X1 = (X1-np.mean(X1, axis=0))/np.std(X1, axis=0)

    Nepochs, Nfeatures = X.shape
    print(X.shape)

    # reduce skew
    X1_skew = skew(X1, axis=0)
    X0_skew = skew(X0, axis=0)
    skew_delta = np.abs(X0_skew)-np.abs(X1_skew)
    skew_delta = skew_delta
    #indices where we have non-beneficial log transform
    bad_log_i = np.where(skew_delta < 0)
    # print(bad_log_i)

    # ignore log transform where it worsens skew
    X1[:,bad_log_i] = X0[:,bad_log_i]
    Y1 = Y

    return X1, Y1

# Does the following:
# 1) Obtain number of epochs per sleep stage per patient
# 2) Filter out patients with a missing sleep stage (79 -> 68)
# 3) Reduce last 2 patients (68 -> 66) so 6-Fold CV can be performed later
# 4) Resample based on average number of epochs per sleep stage averaged across all patients. This comes out to be 247 epochs per sleep stage per patient.
def rebalance_df(df):
    train_size = 44
    valid_size = 11
    test_size = 11
    total_size = train_size + valid_size + test_size # = 66

    info = get_sleep_info_per_patient(df)
    df = remove_patients_with_missing_stages(df, info)
    assert(df['pid'].nunique() >= total_size)
    df = remove_extra_patients(df, df['pid'].nunique() - total_size)

    info = get_sleep_info_per_patient(df)
    df = resample_df(df, info)
    return df

def df_to_array(df):
    X = []
    Y = []
    groups = []

    for row in df.to_numpy():
        X.append(row[1:-1])
        Y.append(int(row[-1]))
        groups.append(row[0])

    X = np.asarray(X)
    Y = np.asarray(Y)       
    groups = np.asarray(groups)
    return X, Y, groups
