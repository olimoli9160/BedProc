# encoding: utf-8

import numpy as np  # linear algebra library
import xgboost as xgb  # ensemble boosted tree model (don't have to import this yet!)
import pandas as pd  # data processing library, converts data into data frames and allows for manipulation of these frames
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
import matplotlib as plt
import os
import math

# ---------------------- Model Constructors -------------------------------#
class XgbWrapper(object):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train, weights):
        dtrain = xgb.DMatrix(x_train, label=y_train, weight=weights)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


class SklearnWrapper(object):
    def __init__(self, model, seed=0, params=None):
        self.model = model(**params)

    def train(self, x_train, y_train):
        self.model.fit(x_train, y_train)
        self.coef_ = self.model.coef_
        self.coefs = dict(zip(feature_names, self.model.coef_))

    def predict(self, x):
        return self.model.predict(x)

# --------------------- Specific Model Parameteres -------------------------#

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'eta': 0.05,
    'gamma': 0,
    'objective': 'multi:softmax',
    'num_class': 4,
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'mlogloss',
    'nrounds': 500,
    'booster': 'gbtree',
}

ls_params = {
    'alpha' : 0.05,
    'max_iter': 100000
}

BIN_DEPENDENCY = "Verschil_bedtijd_dag"
TARGET = "procLabel"
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
df['procLabel'] = 0

print("Dataset, Test, and Train Set Specifications: ")
print("Full Dataset size: " + str(df.shape[0]) + " rows")
print("Full Dataset size: " + str(round(df.shape[0] /5)) + " participants\n")

proc_values = df[BIN_DEPENDENCY]

#-----------------------------Split Dataframe into Train and Test Sets------------------------------#

def customStratifiedSampling(dataframe, y):
    y2 = y.to_frame()

    #------- Makes 4 dataframes containing target values classified into high medium low and none------------#
    none = y2.loc[y2[BIN_DEPENDENCY] <= 0]
    low = y2.loc[(0 < y2[BIN_DEPENDENCY]) & (y2[BIN_DEPENDENCY] < 60)] # value here is rough(although pretty close) estimate, can be choosen more intelligently via histogram analysis
    med = y2.loc[(60 <= y2[BIN_DEPENDENCY]) & (y2[BIN_DEPENDENCY] < 120)]
    high = y2.loc[y2[BIN_DEPENDENCY] >= 120]

    noneIdxs = none.index.values.tolist()
    lowIdxs = low.index.values.tolist()
    medIdxs = med.index.values.tolist()
    highIdxs = high.index.values.tolist()

    noneDF = dataframe.loc[dataframe.index.isin(noneIdxs)]
    lowDF = dataframe.loc[dataframe.index.isin(lowIdxs)] # partitions actual dataset by matching low target values with their respective listing
    medDF = dataframe.loc[dataframe.index.isin(medIdxs)]
    highDF = dataframe.loc[dataframe.index.isin(highIdxs)]

    noneDF['procLabel'] = 0
    lowDF['procLabel'] = 1
    medDF['procLabel'] = 2
    highDF['procLabel'] = 3

    labeledData = [noneDF, lowDF, medDF, highDF]

    dfWithLabels = pd.concat(labeledData)

    noneTrain = noneDF.sample(frac=0.6, random_state=1)
    lowTrain = lowDF.sample(frac=0.6, random_state=1) # select 80% of values from each partition of the dataset (still based on classified target values)
    medTrain = medDF.sample(frac=0.9, random_state=1)
    highTrain = highDF.sample(frac=0.9, random_state=1) # ...this should ensure frequency remains the same, the math checks out

    trainingFrames = [noneTrain, lowTrain, medTrain, highTrain]
    train = pd.concat(trainingFrames)

    trainingIndexes = train.index.values.tolist()
    test = dfWithLabels.loc[~dfWithLabels.index.isin(trainingIndexes)] # test set is built from remaining listings not included in training set

    weights = np.array((train.procLabel ** 10)+1 * 0.01)

    print("Training Set size: " + str(train.shape[0]) + " rows")
    print("Training Set size: " + str(round(train.shape[0] / 5)) + " participants\n")

    print("Test Set size: " + str(test.shape[0]) + " rows")
    print("Test Set size: " + str(round(test.shape[0] / 5)) + " participants\n")
    return train, test, weights


## ---------------------------------- CORE MODELING ROUTINE ------------------------------------##

train, test, weighting = customStratifiedSampling(df, proc_values)
df = pd.concat([train, test])
df = df.drop(BIN_DEPENDENCY, 1)
df = df.sort_index()
#print(df)


#-------------------------- L1 Feature Selection (Lasso) -------------------------#
targets = df[TARGET]
data = df.drop(TARGET, 1)
feature_names = list(data.columns) # used for coefficient mapping

lassoFS = SklearnWrapper(model=Lasso, seed=SEED, params=ls_params)
lassoFS.train(data, targets)
model = SelectFromModel(lassoFS, prefit=True)
feature_idx = model.get_support()
selectedFeatures = data.columns[feature_idx]
print("Viable Features for Modeling: ")
print(selectedFeatures.tolist())
data_new = data[selectedFeatures]
feature_names = list(data_new.columns) # update feature name list to selected features
data_new[TARGET] = targets
df = data_new

y_train = train[TARGET]
y_test = test[TARGET]

train.drop([TARGET], axis=1, inplace=True)
test.drop([TARGET], axis=1, inplace=True)

ntrain = train.shape[0]
ntest = test.shape[0]

x_train = np.array(train)
x_test = np.array(test)

kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)

#----------------- Cross-Validation Mdoel Selection -----------------------------#

def crossValidationModels(model): # get best setup out of folds of cross validation
    best_train = np.zeros((ntrain,))
    best_test = np.zeros((ntest,))

    for fold, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = targets[train_index]
        x_te = x_train[test_index]

        model.train(x_tr, y_tr, weighting)

        best_train[test_index] = model.predict(x_te)
        best_test[:] = model.predict(x_test)

    return model, best_train.reshape(-1, 1), best_test.reshape(-1, 1)

xg = XgbWrapper(seed=SEED, params=xgb_params)
xg_model, xg_train, xg_test = crossValidationModels(xg)


#------------------------ Results --------------------------#

resultTrainDf = pd.DataFrame()
resultTestDf = pd.DataFrame()

resultTrainDf['Actual_Train'] = y_train
resultTrainDf['XG_Pred_Train'] = xg_train

resultTestDf['Actual_Test'] = y_test
resultTestDf['XG_Pred_Test'] = xg_test

print(round(resultTrainDf, 6))
print(round(resultTestDf, 6))
print('\n')

print("XGBoost Results:")
print("XG-Train Accuracy: {}%".format(round((accuracy_score(y_train, xg_train)*100), 2)))
print("XG-Test Accuracy: {}%".format(round((accuracy_score(y_test, xg_test)*100), 2)))

noDF = resultTestDf.loc[(resultTestDf.Actual_Test == 0)]
lowDF = resultTestDf.loc[(resultTestDf.Actual_Test == 1)]
medDF = resultTestDf.loc[(resultTestDf.Actual_Test == 2)]
highDF = resultTestDf.loc[(resultTestDf.Actual_Test == 3)]

print("XG-Test No Accuracy: {}%".format(round((accuracy_score(noDF.Actual_Test, noDF.XG_Pred_Test)*100), 2)))
print("XG-Test Low Accuracy: {}%".format(round((accuracy_score(lowDF.Actual_Test, lowDF.XG_Pred_Test)*100), 2)))
print("XG-Test Med Accuracy: {}%".format(round((accuracy_score(medDF.Actual_Test, medDF.XG_Pred_Test)*100), 2)))
print("XG-Test High Accuracy: {}%".format(round((accuracy_score(highDF.Actual_Test, highDF.XG_Pred_Test)*100), 2)))