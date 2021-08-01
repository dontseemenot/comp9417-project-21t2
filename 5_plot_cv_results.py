# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, label_binarize
from statistics import mean
import pickle
import os
from tabulate import tabulate

import seaborn as sn

# for pickling
def inner_cv_roc_auc_scorer(clf, X_test, Y_test):
    Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
    Y_pred_proba = clf.predict_proba(X_test)
    return roc_auc_score(Y_test, Y_pred_proba, multi_class="ovr", average="macro")

def outer_cv_roc_auc_scorer(clf, X_test, Y_test):
    Y_test = label_binarize(Y_test, classes=[0, 1, 2, 3, 4])
    Y_pred_proba = clf.predict_proba(X_test)
    return roc_auc_score(Y_test, Y_pred_proba, multi_class="ovr", average = None)

# convert summary string into a list of values
def get_summary(summary):
    lines = summary.split("\n")
    lines = [l for l in lines if len(l) > 0]

    y_precision_scores = []
    y_recall_scores = []
    y_f1_scores = []

    for i in range(1,1+5):
        l = lines[i]
        l = l.strip().split(' ')
        l = [x for x in l if len(x) > 0]
        y = l[0]
        values = [float(x) for x in l[1:]]
        precision, recall, f1, support = values
        y_precision_scores.append(precision)
        y_recall_scores.append(recall)
        y_f1_scores.append(f1)

    l = lines[6]
    l = l.split(' ')
    l = [x for x in l if len(x) > 0]
    acc = float(l[1])

    l = lines[6]
    l = l.split(' ')
    l = [x for x in l if len(x) > 0]
    acc = float(l[1])
    
    y_precision_scores = np.array(y_precision_scores)
    y_recall_scores = np.array(y_recall_scores)
    y_f1_scores = np.array(y_f1_scores)
    
    return (y_precision_scores, y_recall_scores, y_f1_scores, acc)

# load all of our cross validation data
folders = "results_intra_bal results_intra_imbal results_inter".split(" ")
names = "mlp lr dt two_layer_mlp".split(" ")
models = {}
for folder in folders:
    M = {}
    models[folder] = M
    for name in names:
        base_name = f"./{folder}/{name}"
        best_params = np.load(f"{base_name}_best_params.npy", allow_pickle=True)
        clfs = np.load(f"{base_name}_classifiers.npy", allow_pickle=True)
        perf_metrics = np.load(f"{base_name}_performance_metrics.npy", allow_pickle=True)

        M[name] = (best_params, clfs, perf_metrics)

# convert folder and model names for plotting
model_name = {
    "lr": "logistic regression", 
    "mlp": "multilayer perceptron", 
    "dt": "decision tree",
    "two_layer_mlp": "two layer multilayer perceptron"
}
method_name = {
    "results_intra_imbal": "imbalanced intra-patient",
    "results_intra_bal": "balanced intra-patient",
    "results_inter": "balanced inter-patient"}

def plot_results(best_params, clfs, perf_metrics):
    nb_clfs = len(clfs)

    roc_avg = []
    acc_avg = []
    conf_mat_avg = []
    precision_avg = []
    recall_avg = []
    f1_avg = []

    for i in range(nb_clfs):
        print(f"Fold={i}")
        roc, acc, summary, conf_mat  = perf_metrics[i]
        print(best_params[i])
        roc_avg.append(roc)
        acc_avg.append(acc)
        conf_mat_avg.append(conf_mat)
        # we get accuracy again
        precision, recall, f1, _ = get_summary(summary)
        precision_avg.append(precision)
        recall_avg.append(recall)
        f1_avg.append(f1)

    roc_avg = np.mean(roc_avg, axis=0)
    acc_avg = np.mean(acc_avg, axis=0)
    conf_mat_avg = np.mean(conf_mat_avg, axis=0)
    precision_avg = np.mean(precision_avg, axis=0)
    recall_avg = np.mean(recall_avg, axis=0)
    f1_avg = np.mean(f1_avg, axis=0)



    print(best_params)
    print(acc_avg)
    print(tabulate(
        [roc_avg, precision_avg, recall_avg],
        tablefmt="simple",
        floatfmt=".2f"))

    label_names = ["Awake", "Stage 1", "Stage 2", "Stage 3/4", "REM"]

    def norm_conf_mat(conf_mat):
        a = np.sum(conf_mat, axis=1)
        return conf_mat/a

    fig = plt.figure(dpi=100)
    sn.heatmap(norm_conf_mat(conf_mat_avg), annot=True, fmt=".3f")
    plt.xticks(np.arange(5)+0.5, label_names)
    plt.yticks(np.arange(5)+0.5, label_names, rotation=0)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.title(f"Confusion matrix for {model_name[model_key]}\nwith {method_name[method_key]} dataset")
    plt.show()

# %% for every single model plot the results
for method_key in models.keys():
    for model_key in models[method_key].keys():
        print(f"Method={method_key}, Model={model_key}")
        best_params, clfs, perf_metrics = models[method_key][model_key]
        plot_results(best_params, clfs, perf_metrics)
