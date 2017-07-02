# encoding: utf-8

import numpy as np  # linear algebra library
import xgboost as xgb  # ensemble boosted tree model (don't have to import this yet!)
import pandas as pd  # data processing library, converts data into data frames and allows for manipulation of these frames
from sklearn.cross_validation import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import os
from math import sqrt

# ---------------------- Model Constructors -------------------------------#
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
    'learning_rate': 0.05,
    'gamma': 0,
    'objective': 'reg:linear',
    'max_depth': 7,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500,
    'booster': 'gblinear',
}

rd_params = {
    'alpha': 10
}

ls_params = {
    'alpha': 0.1,
    'max_iter': 100000
}

lr_params = {
    'normalize': True
}

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

print("Dataset, Test, and Train Set Specifications: ")
print("Full Dataset size: " + str(df.shape[0]) + " rows")
print("Full Dataset size: " + str(round(df.shape[0] /5)) + " participants\n")

#-------------------------- L1 Feature Selection (Lasso) -------------------------#
targets = df[TARGET]
data = df.drop(TARGET,1)
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

plt.rcParams["figure.figsize"] = (15, 15)

axs = targets.hist(bins=[-60, 0.001, 60, 120, 180])
vals = axs.get_yticks()
axs.set_yticklabels(['{:.0f}'.format((x)) for x in vals])
plt.xlabel("Procrastination (in Minutes)", size=18)
plt.ylabel("Number of Listings", size = 18)
plt.title("Histogram of Target Variable (4 bins)", size=24, verticalalignment ="bottom")
plt.show()

#-----------------------------Split Dataframe into Train and Test Sets------------------------------#
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
    for i in range(1, dataframe.shape[0]):
        if i % dayNumber == 0:
            test = test.append(dataframe.ix[i-1])
        else:
            train = train.append(dataframe.ix[i-1])

    print("Training Set size: " + str(train.shape[0]) + " rows")
    print("Training Set size: " + str(round(train.shape[0] / 5)) + " participants\n")

    print("Test Set size: " + str(test.shape[0]) + " rows")
    print("Test Set size: " + str(round(test.shape[0] / 5)) + " participants\n")
    return train, test

def customStratifiedSampling(dataframe, y):
    y2 = y.to_frame()

    #------- Makes 4 dataframes containing target values classified into high medium low and none------------#
    none = y2.loc[y2[TARGET] <= 0]
    low = y2.loc[(0 < y2[TARGET]) & (y2[TARGET] < 60)] # value here is rough(although pretty close) estimate, can be choosen more intelligently via histogram analysis
    med = y2.loc[(60 <= y2[TARGET]) & (y2[TARGET] < 120)]
    high = y2.loc[y2[TARGET] >= 120]

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

    noneTrain = noneDF.sample(frac=0.8, random_state=1)
    lowTrain = lowDF.sample(frac=0.8, random_state=1) # select 80% of values from each partition of the dataset (still based on classified target values)
    medTrain = medDF.sample(frac=0.8, random_state=1)
    highTrain = highDF.sample(frac=0.8, random_state=1) # ...this should ensure frequency remains the same, the math checks out

    trainingFrames = [noneTrain, lowTrain, medTrain, highTrain]
    train = pd.concat(trainingFrames)

    trainingIndexes = train.index.values.tolist()
    test = dfWithLabels.loc[~dfWithLabels.index.isin(trainingIndexes)] # test set is built from remaining listings not included in training set


    #------- for checking frequency of low, med, high remains during split -------#
    #freqNone = len(none) / y.shape[0]
    #freqLow = len(low) / y.shape[0]
    #freqMed = len(med) / y.shape[0]
    #freqHigh = len(high) / y.shape[0]

    #freNoneTr = noneTrain.shape[0] / train.shape[0]
    #freLowTr = lowTrain.shape[0] / train.shape[0]
    #freMedTr = medTrain.shape[0] / train.shape[0]
    #freHighTr = highTrain.shape[0] / train.shape[0]

    #print(str(freqNone) + " : "+ str(freNoneTr))
    #print(str(freqLow) + " : "+ str(freLowTr))
    #print(str(freqMed) + " : "+ str(freMedTr))
    #print(str(freqHigh) + " : "+ str(freHighTr))

    #print("Training Set size: " + str(train.shape[0]) + " rows")
    #print("Training Set size: " + str(round(train.shape[0] / 5)) + " participants\n")

    print("Training Set size: " + str(train.shape[0]) + " rows")
    print("Training Set size: " + str(round(train.shape[0] / 5)) + " participants\n")

    print("Test Set size: " + str(test.shape[0]) + " rows")
    print("Test Set size: " + str(round(test.shape[0] / 5)) + " participants\n")
    return train, test


## ---------------------------------- CORE MODELING ROUTINE ------------------------------------##
run = 1
while(run <= 3): # log all modelling runs

    print("\nModeling Run: "+str(run)+"\n")
    if run == 1:
        train, test = splitOnParticipants(df, 0.8)
    elif run == 2:
        train, test = splitOnDay(df, 5)
    elif run == 3:
        df['procLabel'] = 0
        train, test = customStratifiedSampling(df, targets)


    y_train = train[TARGET]
    y_test = test[TARGET]
    print("Value Range of dependent variable (Verschil_bedtijd_dag) in Training Set: " + str(y_train.min()) + " - " + str(y_train.max())+"\n")
    print("Standard Deviation of dependent variable (Verschil_bedtijd_dag) in Training Set: "+str(y_train.std())+'\n')
    print("Mean value of target variable: " + str(y_train.mean())+'\n')

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
        best_test_skf = np.empty((NFOLDS, ntest)) # best model containing mean of predicted values per fold on test data (stratified k-fold)

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

        best_test[:] = best_test_skf.mean(axis=0) # stratified cv model of test predictions
        return model, best_train.reshape(-1, 1), best_test.reshape(-1, 1)

    xg = XgbWrapper(seed=SEED, params=xgb_params)
    rd = SklearnWrapper(model=Ridge, seed=SEED, params=rd_params)
    ls = SklearnWrapper(model=Lasso, seed=SEED, params=ls_params)
    lr = SklearnWrapper(model=LinearRegression, seed=SEED, params=lr_params)

    lr_model, lr_train, lr_test = crossValidationModels(lr)
    xg_model, xg_train, xg_test = crossValidationModels(xg)
    rd_model, rd_train, rd_test = crossValidationModels(rd)
    ls_model, ls_train, ls_test = crossValidationModels(ls)

    lr_coef = pd.DataFrame(lr_model.coefs, index=["Linear Regression"])
    rd_coef = pd.DataFrame(rd_model.coefs, index=["Ridge Regression"])
    ls_coef = pd.DataFrame(ls_model.coefs, index=["Lasso Regression"])

    coef_frames = [lr_coef, rd_coef, ls_coef]
    coefDf = pd.concat(coef_frames)
    print("Predictor Coefficients of current run: (XGBoost Excluded for now...)\n")
    print(coefDf)
    print("\n")

    #------------------------- Plotting of Feature Importance -----------------------------------#
    ls_coef = ls_coef.sort(columns=ls_coef.index[:10].tolist(), axis=1, ascending=False)
    row = ls_coef.ix[0]
    print(row)
    ax = row.plot(kind='bar')
    plt.xlabel("Features", size=18)
    plt.ylabel("Feature Coefficient", size=18)
    plt.title("Feature Importance: Run " + str(run), size=24, verticalalignment="bottom")
    plt.xticks(horizontalalignment = "right", rotation=45)
    plt.show()


    #------------------------ Results --------------------------#

    resultTrainDf = pd.DataFrame()
    resultTestDf = pd.DataFrame()

    resultTrainDf['Actual_Train'] = y_train
    resultTrainDf['LR_Pred_Train'] = lr_train
    resultTrainDf['XG_Pred_Train'] = xg_train
    resultTrainDf['RD_Pred_Train'] = rd_train
    resultTrainDf['LS_Pred_Train'] = ls_train

    resultTestDf['Actual_Test'] = y_test
    resultTestDf['LR_Pred_Test'] = lr_test
    resultTestDf['XG_Pred_Test'] = xg_test
    resultTestDf['RD_Pred_Test'] = rd_test
    resultTestDf['LS_Pred_Test'] = ls_test

    print(round(resultTrainDf, 6))
    print(round(resultTestDf, 6))
    print('\n')

    def mape(actual, predicted): # Mean absolute percentage error (I'll write my own version of the function later or cite this one)
        mask = actual != 0
        return (np.fabs(actual[mask] - predicted[mask])/actual[mask]).mean() * 100 # returned value is a percentage

    print("Linear Regression Results:") # RMSE = Root mean squared error, MAE = Mean Absolute Error, MAPE = Mean Absolute Percent Error
    print("LR-Train RMSE: {}".format(sqrt(mean_squared_error(y_train, lr_train))))
    print("LR-Train MAE: {}".format(mean_absolute_error(y_train, lr_train)))
    print("LR-Train MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["LR_Pred_Train"]), 2))+"\n")

    print("LR-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, lr_test))))
    print("LR-Test MAE: {}".format(mean_absolute_error(y_test, lr_test)))
    print("LR-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["LR_Pred_Test"]), 2))+"\n")

    print("XGBoost Results:")
    print("XG-Train RMSE: {}".format(sqrt(mean_squared_error(y_train, xg_train))))
    print("XG-Train MAE: {}".format(mean_absolute_error(y_train, xg_train)))
    print("XG-Train MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["XG_Pred_Train"]), 2))+"\n")

    print("XG-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, xg_test))))
    print("XG-Test MAE: {}".format(mean_absolute_error(y_test, xg_test)))
    print("XG-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["XG_Pred_Test"]), 2))+"\n")

    print("Ridge Results:")
    print("RD-Train RMSE: {}".format(sqrt(mean_squared_error(y_train, rd_train))))
    print("RD-Train MAE: {}".format(mean_absolute_error(y_train, rd_train)))
    print("RD-Train MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["RD_Pred_Train"]), 2))+"\n")

    print("RD-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, rd_test))))
    print("RD-Test MAE: {}".format(mean_absolute_error(y_test, rd_test)))
    print("RD-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["RD_Pred_Test"]), 2))+"\n")

    print("Lasso Results:")
    print("LS-Train RMSE: {}".format(sqrt(mean_squared_error(y_train, ls_train))))
    print("LS-Train MAE: {}".format(mean_absolute_error(y_train, ls_train)))
    print("LS-Train MAPE: {}%".format(round(mape(resultTrainDf["Actual_Train"], resultTrainDf["LR_Pred_Train"]), 2))+"\n")

    print("LS-Test RMSE: {}".format(sqrt(mean_squared_error(y_test, ls_test))))
    print("LS-Test MAE: {}".format(mean_absolute_error(y_test, ls_test)))
    print("LS-Test MAPE: {}%".format(round(mape(resultTestDf["Actual_Test"], resultTestDf["LS_Pred_Test"]), 2))+"\n")
    run += 1