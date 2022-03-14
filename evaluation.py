import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate


def evaluate(labels, predict):
    accuracy = accuracy_score(labels, predict)
    precision = precision_score(labels, predict)
    recall = recall_score(labels, predict)
    f1 = f1_score(labels, predict)
    auc = roc_auc_score(labels, predict)
    fpr, tpr, thresholds = roc_curve(labels, predict)
    thresholds = thresholds[np.where(tpr == recall)][0]
    fpr = fpr[np.where(tpr == recall)][0]
    tpr = tpr[np.where(tpr == recall)][0]
    return accuracy, precision, recall, f1, auc, fpr, tpr, thresholds


def get_cross_var_score(estimator, X, y, scoring, cv):
    cv_score = cross_validate(estimator=estimator, X=X, y=y, scoring=scoring, cv=cv)
    time_fit = np.mean(cv_score['fit_time'])
    time_score = np.mean(cv_score['score_time'])
    accuracy_cv = np.mean(cv_score['test_accuracy'])
    precision_cv = np.mean(cv_score['test_precision'])
    recall_cv = np.mean(cv_score['test_recall'])
    f1_cv = np.mean(cv_score['test_f1'])
    auc_cv = np.mean(cv_score['test_roc_auc'])
    return [time_fit, time_score, accuracy_cv, precision_cv, recall_cv, f1_cv, auc_cv]
