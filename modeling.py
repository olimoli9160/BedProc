# encoding: utf-8

import numpy as np  # linear algebra library
import xgboost as xgb  # ensemble boosted tree model (don't have to import this yet!)
import pandas as pd  # data processing library, converts data into data frames and allows for manipulation of these frames
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
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

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
data_file = os.path.join(__location__, 'LISS_bedtime_final.csv')

df = pd.read_csv(data_file)
df = df.drop("Unnamed: 0", 1) # drop redundant indexing

##### Droping those entries which have 0.0 for procrastination time##### Models performed considerably worse
#df = df.loc[df[TARGET] != 0.0]
#df = df.reset_index(drop=True)
#print(df.shape[0])

print("Dataset, Test, and Train Set Specifications: ")
print("Full Dataset size: " + str(df.shape[0]) + " rows")
print("Full Dataset size: " + str(round(df.shape[0] /5)) + " participants\n")


#-----------------------------Split Dataframe into Train and Test Sets------------------------------#
targets = df[TARGET]

def splitOnParticipants(dataframe, train_size):
    num_participants = dataframe.shape[0] / 5 # number of days
    split = round((num_participants * train_size) * 5)

    train = dataframe.ix[:split-1]
    test = dataframe.ix[split:]
    print("Training Set size: " + str(train.shape[0]) + " rows")
    print("Training Set size: " + str(round(train.shape[0]/ 5)) + " participants\n")

    print("Test Set size: " + str(test.shape[0]) + " rows")
    print("Test Set size: " + str(round(test.shape[0] / 5)) + " participants\n")

    return train, test


def splitOnDay(dataframe, dayNumber):
    train = pd.DataFrame()
    test = pd.DataFrame()
    for i in range(0, dataframe.shape[0]):
        if i % dayNumber == 0:
            test = test.append(dataframe.ix[i])
        else:
            train = train.append(dataframe.ix[i])

    print("Training Set size: " + str(train.shape[0]) + " rows")
    print("Training Set size: " + str(round(train.shape[0] / 5)) + " participants\n")

    print("Test Set size: " + str(test.shape[0]) + " rows")
    print("Test Set size: " + str(round(test.shape[0] / 5)) + " participants\n")
    return train, test

run = 1
while(run <= 2): # log all modelling runs

    print("\nModeling Run: "+str(run)+"\n")
    if run < 2:
        train, test = splitOnParticipants(df, 0.7)
    else:
        train, test = splitOnDay(df, 5)
    y_train = train[TARGET]
    y_test = test[TARGET]
    print("Value Range of dependent variable (Verschil_bedtijd_dag) in Training Set: " + str(y_train.min()) + " - " + str(y_train.max())+"\n")
    print("Standard Deviation of dependent variable (Verschil_bedtijd_dag) in Training Set: "+str(y_train.std())+'\n')

    train.drop([TARGET], axis=1, inplace=True)
    test.drop([TARGET], axis=1, inplace=True)

    ntrain = train.shape[0]
    ntest = test.shape[0]

    x_train = np.array(train)
    x_test = np.array(test)
    #print("X_train listings: "+ str(x_train.shape[0]) + "   X_test listings: " + str(x_test.shape[0]))

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
        'nrounds': 500,
        "booster": "gblinear"
    }

    rd_params={
        'alpha': 10
    }

    ls_params={
        'alpha': 0.05,
        'max_iter': 100000
    }

    lr_params={
        'normalize': True
    }

    #----------------- Cross-Validation Mdoel Selection -----------------------------#

    def crossValidationModels(model): # get best setup out of folds of cross validation
        best_train = np.zeros((ntrain,))
        best_test = np.zeros((ntest,))
        best_test_skf = np.empty((NFOLDS, ntest)) # mean of values per fold (stratified k-fold)

        for fold, (train_index, test_index) in enumerate(kf):
            x_tr = x_train[train_index]
            #print(x_tr)
            y_tr = targets[train_index]
            #print(y_tr)
            x_te = x_train[test_index]
            #print(x_te)

            model.train(x_tr, y_tr)

            best_train[test_index] = model.predict(x_te)
            best_test_skf[fold, :] = model.predict(x_test)

        best_test[:] = best_test_skf.mean(axis=0)
        return best_train.reshape(-1, 1), best_test.reshape(-1, 1)

    xg = XgbWrapper(seed=SEED, params=xgb_params)
    rd = SklearnWrapper(model=Ridge, seed=SEED, params=rd_params)
    ls = SklearnWrapper(model=Lasso, seed=SEED, params=ls_params)
    lr = SklearnWrapper(model=LinearRegression, seed=SEED, params=lr_params)

    lr_oof_train, lr_oof_test = crossValidationModels(lr)
    xg_oof_train, xg_oof_test = crossValidationModels(xg)
    rd_oof_train, rd_oof_test = crossValidationModels(rd)
    ls_oof_train, ls_oof_test = crossValidationModels(ls)


    #------------------------ Results --------------------------#

    resultTrainDf = pd.DataFrame()
    resultTestDf = pd.DataFrame()

    resultTrainDf['Actual_Train'] = y_train
    resultTrainDf['LR_Pred_Train'] = lr_oof_train
    resultTrainDf['XG_Pred_Train'] = xg_oof_train
    resultTrainDf['RD_Pred_Train'] = rd_oof_train
    resultTrainDf['LS_Pred_Train'] = ls_oof_train

    resultTestDf['Actual_Test'] = y_test
    resultTestDf['LR_Pred_Test'] = lr_oof_test
    resultTestDf['XG_Pred_Test'] = xg_oof_test
    resultTestDf['RD_Pred_Test'] = rd_oof_test
    resultTestDf['LS_Pred_Test'] = ls_oof_test

    print(round(resultTrainDf, 6))
    print(round(resultTestDf, 6))
    print('\n')

    def mape(a, b): # Mean absolute percentage error (I'll write my own version of the function later or cite this one)
        mask = a != 0
        return (np.fabs(a[mask] - b[mask])/a[mask]).mean() * 100 # returned value is a percentage

    print("Linear Regression Results:") # RMSE = Root mean squared error, MAE = Mean Absolute Error, MAPE = Mean Absolute Percent Error
    print("LR-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, lr_oof_train))))
    print("LR-CV MAE: {}".format(mean_absolute_error(y_train, lr_oof_train)))
    print("LR-CV MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["LR_Pred_Train"]), 2))+"\n")

    print("LR-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, lr_oof_test))))
    print("LR-Test MAE: {}".format(mean_absolute_error(y_test, lr_oof_test)))
    print("LR-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["LR_Pred_Test"]), 2))+"\n")

    print("XGBoost Results:")
    print("XG-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, xg_oof_train))))
    print("XG-CV MAE: {}".format(mean_absolute_error(y_train, xg_oof_train)))
    print("XG-CV MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["XG_Pred_Train"]), 2))+"\n")

    print("XG-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, xg_oof_test))))
    print("XG-Test MAE: {}".format(mean_absolute_error(y_test, xg_oof_test)))
    print("XG-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["XG_Pred_Test"]), 2))+"\n")

    print("Ridge Results:")
    print("RD-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, rd_oof_train))))
    print("RD-CV MAE: {}".format(mean_absolute_error(y_train, rd_oof_train)))
    print("RD-CV MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["RD_Pred_Train"]), 2))+"\n")

    print("RD-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, rd_oof_test))))
    print("RD-Test MAE: {}".format(mean_absolute_error(y_test, rd_oof_test)))
    print("RD-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["RD_Pred_Test"]), 2))+"\n")

    print("Lasso Results:")
    print("LS-CV RMSE: {}".format(sqrt(mean_squared_error(y_train, ls_oof_train))))
    print("LS-CV MAE: {}".format(mean_absolute_error(y_train, ls_oof_train)))
    print("LS-CV MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["LR_Pred_Train"]), 2))+"\n")

    print("LS-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, ls_oof_test))))
    print("LS-Test MAE: {}".format(mean_absolute_error(y_test, ls_oof_test)))
    print("LS-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["LS_Pred_Test"]), 2))+"\n")
    run += 1