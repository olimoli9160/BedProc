# encoding: utf-8

import numpy as np  # linear algebra library
import xgboost as xgb  # ensemble boosted tree model (don't have to import this yet!)
import pandas as pd  # data processing library, converts data into data frames and allows for manipulation of these frames
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import matplotlib as plt
import os
from math import sqrt

TARGET = "Verschil_bedtijd_dag"
NFOLDS = 5
SEED = 0
NROWS = None

pd.set_option('expand_frame_repr', False)  # shows dataframe without wrapping to next line
pd.set_option('chained_assignment', None)
pd.set_option('display.max_columns', 500)
#np.set_printoptions(precision=6)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
data_file = os.path.join(__location__, 'LISS_bedtime_final.csv')

df = pd.read_csv(data_file)
df = df.drop("Unnamed: 0", 1) # drop redundant indexing
#print(df)

#-----------------------------Split Dataframe into Train and Test Sets------------------------------#
targets = np.log(df[TARGET]+1)

train, test = train_test_split(df, train_size = 0.7)
y_train = np.log(train[TARGET]+1)
y_test = np.log(test[TARGET]+1)
print("Value Range of dependent variable (Verschil_bedtijd_dag) in Training Set after Logarithmic Transformation: " + str(y_train.min()) + " - " + str(y_train.max())+"\n")
print("Standard Deviation of dependent variable (Verschil_bedtijd_dag) in Training Set: "+str(y_train.std())+'\n')

train.drop([TARGET], axis=1, inplace=True)

ntrain = train.shape[0]
ntest = test.shape[0]

x_train = np.array(df[:train.shape[0]])
x_test = np.array(df[train.shape[0]:])
#print("X_train listings: "+ str(int(x_train.size / 133)) + "   X_test listings:" + str(int(x_test.size / 133)))

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)


#---------------------- Model Constructors -------------------------------#
class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))

class SklearnWrapper(object):
    def __init__(self, model, seed=0, params=None):
        if model != LinearRegression:
            params['random_state'] = seed
        self.model = model(**params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, x):
        return self.model.predict(x)

#--------------------- Specific Model Parameteres -------------------------#

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.1,
    'objective': 'reg:linear',
    'max_depth': 6,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}

rd_params={
    'alpha': 10
}

ls_params={
    'alpha': 0.005,
    'max_iter': 10000
}

lr_params={
    'normalize': True
}

#----------------- Cross-Validation Mdoel Selection -----------------------------#

def get_oof(model): # get best setup out of folds of cross validation
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf): # issue with indices not matching, causing NaN predictions
        x_tr = x_train[train_index]
        #print(x_tr)
        y_tr = targets[train_index]
        #print(y_tr)
        x_te = x_train[test_index]
        #print(x_te)

        model.train(x_tr, y_tr)

        oof_train[test_index] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

xg = XgbWrapper(seed=SEED, params=xgb_params)
rd = SklearnWrapper(model=Ridge, seed=SEED, params=rd_params)
ls = SklearnWrapper(model=Lasso, seed=SEED, params=ls_params)
lr = SklearnWrapper(model=LinearRegression, seed=SEED, params=lr_params)

xg_oof_train, xg_oof_test = get_oof(xg)
rd_oof_train, rd_oof_test = get_oof(rd)
ls_oof_train, ls_oof_test = get_oof(ls)
lr_oof_train, lr_oof_test = get_oof(lr)

print("Linear Regression Results:") # RMSE = Root mean squared error
print("LR-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, lr_oof_train))))
print("LR-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, lr_oof_test)))+"\n")


print("XGBoost Results:")
print("XG-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
print("XG-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, xg_oof_test)))+"\n")


print("Ridge Results:")
print("RD-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
print("RD-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, rd_oof_test)))+"\n")


print("Lasso Results:")
print("LS-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
print("LS-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, ls_oof_test)))+"\n")

