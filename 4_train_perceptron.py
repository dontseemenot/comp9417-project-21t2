# %%
'''
This code does 2 things
1) Run rebalancing scripts
2) Cross validation based on inter-patient method
3) Train on simple classifiers (not yet implemented)
'''
# %% 1) run rebalancing scripts
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch._C import Value
from data_rebalancing_helpers import *

# (Change this if needed) load csv file with epoch features
# data_csv = 'E:/HDD documents/University/comp9417/comp9417-project-21t2/data/subband_data.csv'
data_csv = './data/subband_data.csv'
df = pd.read_csv(data_csv)
df = df.dropna()

# df = rebalance_df(df)
X, Y, groups = df_to_array(df)
X1, Y1 = transform(X, Y)

# %% Get our roc-auc scorer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
from collections import Counter
def get_roc_auc(y_true, y_pred):
    N = y_true.shape[0]
    y_true_probs = np.zeros((N,5))
    y_true_probs[np.arange(0,N), y_true] = 1
    auc = roc_auc_score(y_true_probs, y_pred, multi_class="ovo", average=None)
    # return auc
    return np.mean(auc)

roc_scorer = make_scorer(get_roc_auc, greater_is_better=True, needs_proba=True)

# %% 3) cross validation grid search
from sklearn.model_selection import GroupKFold, cross_val_score, GridSearchCV
from statistics import mean

# Inter-patient method with nested CV
roc_auc_scores = []
best_params = []

from importlib import reload  
import perceptron_estimator
import torch
import torch.nn as nn
reload(perceptron_estimator)


estimator = perceptron_estimator.PytorchEstimator(total_epochs=1)
param_grid = {
    'hidden_count': [10, 22, 100], 
    'activation': [nn.ReLU, nn.Sigmoid, nn.PReLU]}

# Because sklearn does not support cross_val_score() function with GroupKFold CV, we have to do this manually
outer_cv = GroupKFold(n_splits=3)
inner_cv = GroupKFold(n_splits=4)

# %% 4) Start training
all_classifiers = []
for train_valid_i, test_i in outer_cv.split(X1, Y1, groups = groups):
    print("Starting outer cross validation")
    X_train_valid, X_test = X1[train_valid_i], X1[test_i]
    Y_train_valid, Y_test = Y1[train_valid_i], Y1[test_i]
    groups_train_valid = groups[train_valid_i]

    # get our best estimator
    clf = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=inner_cv, scoring=roc_scorer, verbose=3)
    clf.fit(X_train_valid, Y_train_valid, groups=groups_train_valid)
    all_classifiers.append(clf)

    best_param = clf.best_params_
    best_params.append(best_param)
    roc = roc_auc_score(Y_test, clf.predict_proba(X_test), multi_class='ovr')
    avg_roc = np.mean(roc)
    roc_auc_scores.append(avg_roc)


roc_auc = mean(roc_auc_scores)
print(f'ROC AUC across all folds: {roc_auc_scores}\nMean: {roc_auc}')
print(f'Best params: {best_params}')
