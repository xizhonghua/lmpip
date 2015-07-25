#!/usr/bin/env python

'''
In this scipt, SVM's SVR model is used, which can be tuned with different parameters

@Author : Nikesh Bajaj

'''
# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
import xgboost as xgb


def gini(expected, predicted):
    assert expected.shape[0] == predicted.shape[0], 'unequal number of rows'

    _all = np.asarray(np.c_[
        expected,
        predicted,
        np.arange(expected.shape[0])], dtype=np.float)

    _EXPECTED = 0
    _PREDICTED = 1
    _INDEX = 2

    # sort by predicted descending, then by index ascending
    sort_order = np.lexsort((_all[:, _INDEX], -1 * _all[:, _PREDICTED]))
    _all = _all[sort_order]

    total_losses = _all[:, _EXPECTED].sum()
    gini_sum = _all[:, _EXPECTED].cumsum().sum() / total_losses
    gini_sum -= (expected.shape[0] + 1.0) / 2.0
    return gini_sum / expected.shape[0]

def gini_normalized(expected, predicted):
    return gini(expected, predicted) / gini(expected, expected)

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../data/train.csv")
test  = pd.read_csv("../data/test.csv")

labels = train.Hazard
train.drop('Hazard', axis=1, inplace=True)

columns = train.columns
test_ind = test.Id

train = np.array(train)
test = np.array(test)

for i in range(train.shape[1]):
    if type(train[1,i]) is str:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[:,i]) + list(test[:,i]))
        train[:,i] = lbl.transform(train[:,i])
        test[:,i] = lbl.transform(test[:,i])

train = train.astype(float)
test = test.astype(float)

print(train.shape)
print(test.shape)

mn = np.mean(train,axis=0)
st = np.std(train,axis=0)
train =(train-mn)/(st)
test =(test-mn)/(st)

train_scores = []
test_scores = []

for i in range(0, 10):
    Xtr, Xts, ytr, yts = train_test_split(train, labels, test_size=.2)

    clf = xgb.XGBRegressor(max_depth=6, n_estimators=100)

    clf.fit(Xtr, ytr)
    ytp = clf.predict(Xtr)
    ysp = clf.predict(Xts)

    train_score = gini_normalized(ytr, ytp)
    test_score = gini_normalized(yts, ysp)

    print('Tr Score: ', train_score)
    print('Va Score: ', test_score)

    train_scores.append(train_score)
    test_scores.append(test_score)

print("mean train score:", np.mean(train_scores))
print("mean test score:", np.mean(test_scores))

print('Training on full dataset')

clf.fit(train,labels)

yp = clf.predict(test)

preds = pd.DataFrame({"Id": test_ind, "Hazard": yp})
preds = preds.set_index('Id')
preds.to_csv('Benchmark_SVM.csv')
