# %%
#
import numpy as np
import pandas as pd
from scipy.signal.ltisys import dfreqresp
from sklearn.utils import resample
from sklearn.model_selection import LeavePGroupsOut
import math
# %%
def get_sleep_info_per_patient(df):
    max_W = 0
    max_S1 = 0
    max_S2 = 0
    max_SWS = 0
    max_R = 0
    max_all = 0
    pids = df['pid'].unique()
    info = []
    for pid in pids:
    #for pid, i in zip(pids, range(1)):
        pid_df = df.loc[df['pid'] == pid]
        #print(pid_df)
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
    bad_patients = [p[0] for p in info if 0 in p[1:None]]
    for pid in bad_patients:
        df = df.drop(df[df.pid == pid].index)
    return df

def resample_df(df):
    
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
# Does few things
# 1) Obtain number of epochs per sleep stage per patient
# 2) Filter out patients with a missing sleep stage (79 -> 68)
# 3) Reduce last 2 patients (68 -> 66) so 5-Fold CV can be performed later
# 4) Resample based on average number of epochs per sleep stage averaged across all patients
def rebalance_df(df):
    train_size = 44
    valid_size = 11
    test_size = 11
    train_and_valid_size = train_size + valid_size  # = 55
    total_size = train_size + valid_size + test_size # = 66

    info = get_sleep_info_per_patient(df)
    df = remove_patients_with_missing_stages(df, info)
    assert(df['pid'].nunique() >= total_size)
    df = remove_extra_patients(df, total_size - df['pid'].nunique())

    df = resample_df(df)
    return df


data_csv = 'E:/HDD documents/University/comp9417/comp9417-project-21t2/data/subband_data.csv'
df = pd.read_csv(data_csv)
num_features = df.shape[1] - 2 # Minus pID and pClass column

df = rebalance_df(df)


# %%
X = []
y = []
groups = []
for row in df.to_numpy():
    X.append(row[1:-1])
    y.append(int(row[-1]))
    groups.append(int(row[0]) % 5)
X = np.asarray(X)   # Use asarray to avoid making copies of array
X = X.reshape(X.shape[0], X.shape[1], 1)
y = np.asarray(y)       
y = y.reshape(y.shape[0], 1)  
groups = np.asarray(groups)
#for ((train_index, test_index), iteration, f) in zip(lpgo.split(X, y, groups), range(9), os.listdir('./CAP results/balanced data epoch 80 rem models/')):
# %%
lpgo = LeavePGroupsOut(n_groups = 5)
lpgo.get_n_splits(X, y, groups)
for train_index, test_index in lpgo.split(X, y, groups):    # returns generators
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    print(f'X_train {X[train_index].shape} y_train {y[train_index].shape} X_test {X[test_index].shape} y_test {y[test_index].shape}')


