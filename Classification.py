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
import matplotlib.pyplot as plt
import math
from sklearn.utils import resample
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, label_binarize
from statistics import mean
import pickle
from sklearn.neural_network import MLPClassifier
import os

from sklearn import tree
from scipy.stats import skew
from collections import Counter
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
    print(bad_log_i)

    # # plot effect of log transform
    # plt.figure()
    # plt.scatter(np.arange(Nfeatures), skew_delta)
    # plt.axhline(0, c="black", linestyle="--")
    # plt.title("Skewness improvement with log transform")
    # plt.show()

    # i = np.argmax(np.abs(X1_skew))
    # plt.figure()
    # _ = plt.hist(X1[:,i], bins=100, label="Log transform")
    # _ = plt.hist(X0[:,i], bins=100, label="Original")
    # plt.legend()
    # plt.title(f"Histogram of worst skewness after log transform - X{i}")
    # plt.show()

    # i = np.argmin(np.abs(X1_skew))
    # plt.figure()
    # _ = plt.hist(X1[:,i], bins=100, label="Log transform")
    # _ = plt.hist(X0[:,i], bins=100, label="Original")
    # plt.legend()
    # plt.title(f"Histogram of best skewness after log transform - X{i}")
    # plt.show()

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

# Transform categorical data into one hot encoding
# Required for GridCVSearch with roc_auc as scorer
# e.g: [0, 1, 2, 3, 4] = [[1, 0, 0, 0, 0], ... , [0, 0, 0, 0, 1]]
# def inner_cv_roc_auc_scorer(Y_test, Y_pred):
#     Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
#     Y_pred = label_binarize(Y_pred, classes=[0, 1, 2, 3, 4])
#     return roc_auc_score(Y_test, Y_pred, multi_class="ovr", average="macro")

def inner_cv_roc_auc_scorer(clf, X_test, Y_test):
    Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
    Y_pred_proba = clf.predict_proba(X_test)
    return roc_auc_score(Y_test, Y_pred_proba, multi_class="ovr", average="macro")

def outer_cv_roc_auc_scorer(clf, X_test, Y_test):
    Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
    Y_pred_proba = clf.predict_proba(X_test)
    return roc_auc_score(Y_test, Y_pred_proba, multi_class="ovr", average = None)

# (Change this if needed) load csv file with epoch features
data_csv = 'E:/HDD documents/University/comp9417/comp9417-project-21t2/data/subband_data.csv'
df = pd.read_csv(data_csv)
df = df.dropna()

df = rebalance_df(df)
X, Y, groups = df_to_array(df)
X1, Y1 = transform(X, Y)
# Because sklearn does not support cross_val_score() function with GroupKFold CV, we have to do this manually
outer_cv = GroupKFold(n_splits = 3)
inner_cv = GroupKFold(n_splits = 4)
results_path = './models/'
# %%
lr_classifiers = []
lr_best_params = []
lr_performance_metrics = []
for train_valid_i, test_i in outer_cv.split(X1, Y1, groups = groups):
    X_train_valid, X_test = X1[train_valid_i], X1[test_i]
    Y_train_valid, Y_test = Y1[train_valid_i], Y1[test_i]
    groups_train_valid = groups[train_valid_i]
    
    lr = LogisticRegression(max_iter = 10000)
    param_grid = {'C': [0.1, 1, 10]}
    clf = GridSearchCV(estimator = lr, param_grid = param_grid, cv = inner_cv, scoring = inner_cv_roc_auc_scorer, verbose = 3)
    clf.fit(X_train_valid, Y_train_valid, groups = groups_train_valid)
    lr_best_params.append(clf.best_params_)
    lr_classifiers.append(clf)
    # Y_pred = lr_clf.predict(X_test)

    # Performance metrics
    Y_pred = clf.predict(X_test)
    roc_auc = outer_cv_roc_auc_scorer(clf, X_test, Y_test)
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, target_names = ['W', 'S1', 'S2', 'SWS', 'R'])
    cm = confusion_matrix(Y_test, Y_pred)
    lr_performance_metrics.append([roc_auc, acc, report, cm])
    
np.save(os.path.join(results_path, 'lr_classifiers'), lr_classifiers)
np.save(os.path.join(results_path, 'lr_best_params'), lr_best_params)
np.save(os.path.join(results_path, 'lr_performance_metrics'), lr_performance_metrics)
# %%
pca_classifiers = []
pca_best_params = []
pca_performance_metrics = []
for train_valid_i, test_i in outer_cv.split(X1, Y1, groups = groups):
    X_train_valid, X_test = X1[train_valid_i], X1[test_i]
    Y_train_valid, Y_test = Y1[train_valid_i], Y1[test_i]
    groups_train_valid = groups[train_valid_i]
    pca = PCA()
    param_grid = {'n_components': [5, 10, 15, 20]}
    clf = GridSearchCV(estimator = pca, param_grid = param_grid, cv = inner_cv, scoring = inner_cv_roc_auc_scorer, verbose = 3)
    clf.fit(X_train_valid, Y_train_valid, groups = groups_train_valid)

    pca_best_params.append(clf.best_params_)
    pca_classifiers.append(clf)
    # Y_pred = lr_clf.predict(X_test)

    # Performance metrics
    Y_pred = clf.predict(X_test)
    roc_auc = outer_cv_roc_auc_scorer(clf, X_test, Y_test)
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, target_names = ['W', 'S1', 'S2', 'SWS', 'R'])
    cm = confusion_matrix(Y_test, Y_pred)
    pca_performance_metrics.append([roc_auc, acc, report, cm])
    
    # Testing code
np.save(os.path.join(results_path, 'pca_classifiers'), pca_classifiers)
np.save(os.path.join(results_path, 'pca_best_params'), pca_best_params)
np.save(os.path.join(results_path, 'pca_performance_metrics'), pca_performance_metrics)
# %%
mlp_classifiers = []
mlp_best_params = []
mlp_performance_metrics = []
for train_valid_i, test_i in outer_cv.split(X1, Y1, groups = groups):
    X_train_valid, X_test = X1[train_valid_i], X1[test_i]
    Y_train_valid, Y_test = Y1[train_valid_i], Y1[test_i]
    groups_train_valid = groups[train_valid_i]
    mlp = MLPClassifier(max_iter = 1000)
    param_grid = {'hidden_layer_sizes' : [10, 50, 100]}
    clf = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=inner_cv, scoring=inner_cv_roc_auc_scorer, verbose=3)
    clf.fit(X_train_valid, Y_train_valid, groups=groups_train_valid)

    mlp_best_params.append(clf.best_params_)
    mlp_classifiers.append(clf)

    # Performance metrics
    Y_pred = clf.predict(X_test)
    roc_auc = outer_cv_roc_auc_scorer(clf, X_test, Y_test)
    acc = accuracy_score(Y_test, Y_pred)
    report = classification_report(Y_test, Y_pred, target_names = ['W', 'S1', 'S2', 'SWS', 'R'])
    cm = confusion_matrix(Y_test, Y_pred)
    mlp_performance_metrics.append([roc_auc, acc, report, cm])
    
np.save(os.path.join(results_path, 'mlp_classifiers'), mlp_classifiers)
np.save(os.path.join(results_path, 'mlp_best_params'), mlp_best_params)
np.save(os.path.join(results_path, 'mlp_performance_metrics'), mlp_performance_metrics)
# %%
