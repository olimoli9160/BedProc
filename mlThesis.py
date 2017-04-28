import numpy as np  # linear algebra library
import xgboost as xgb  # ensemble boosted tree model (don't have to import this yet!)
import pandas as pd  # data processing library, converts data into data frames and allows for manipulation of these frames
import sklearn as sk
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
import time
import datetime
import matplotlib as plt

#TARGET = "Verschil_bedtijd_dag1"
#NFOLDS = 5
#SEED = 0
#NROWS = None

pd.set_option('expand_frame_repr', False)  # shows dataframe without wrapping to next line
pd.set_option('chained_assignment', None)
pd.set_option('display.max_columns', 500)

data_path = "C:/Users/Michael/Desktop/BachelorThesis/"  # my working directory
data_file = data_path + "LISS_bedtime.csv"

df = pd.read_csv(data_file)
#print(df)

#----- Drop Unusable Data (already been transformed in another manner) -----#
df = df.drop("Reason4_nl", 1)
df = df.drop("Reason_overig", 1)
df = df.drop("brutoink", 1)
df = df.drop("brutoink_f", 1)
df = df.drop("nettoink", 1)
df = df.drop("nettoink_f", 1)
df = df.drop("netinc", 1)
df = df.drop("brutohh_f", 1)
df = df.drop("nettohh_f", 1)

#----- Drop Survey Specific Responses (Not Predictors) -----#
df = df.drop("im13a_m", 1)
df = df.drop("Vragenmoeilijk", 1)
df = df.drop("Vragenduidelijk", 1)
df = df.drop("Eval1", 1)
df = df.drop("Eval2", 1)
df = df.drop("Eval3", 1)
df = df.drop("opm", 1)
df = df.drop("evaopm", 1)
df = df.drop("Baseline_datumstart", 1)
df = df.drop("Baseline_tijdstart", 1)
df = df.drop("Baseline_datumeinde", 1)
df = df.drop("Baseline_tijdeinde", 1)
df = df.drop("Baseline_duur", 1)
df = df.drop("Dagboek_jaarmaand", 1)
df = df.drop("in13a031", 1)
df = df.drop("in13a032", 1)
df = df.drop("in13a033", 1)
df = df.drop("in13a034", 1)
df = df.drop("in13a035", 1)
df = df.drop("in13a057", 1)
df = df.drop("in13a058", 1)
df = df.drop("in13a059", 1)
df = df.drop("in13a060", 1)
df = df.drop("in13a061", 1)
df = df.drop("in13a083", 1)
df = df.drop("in13a084", 1)
df = df.drop("in13a085", 1)
df = df.drop("in13a086", 1)
df = df.drop("in13a087", 1)
df = df.drop("in13a109", 1)
df = df.drop("in13a110", 1)
df = df.drop("in13a111", 1)
df = df.drop("in13a112", 1)
df = df.drop("in13a113", 1)
df = df.drop("in13a135", 1)
df = df.drop("in13a136", 1)
df = df.drop("in13a137", 1)
df = df.drop("in13a138", 1)
df = df.drop("in13a139", 1)
df = df.drop("in13a161", 1)
df = df.drop("in13a162", 1)
df = df.drop("in13a163", 1)
df = df.drop("in13a164", 1)
df = df.drop("in13a165", 1)
df = df.drop("in13a187", 1)
df = df.drop("in13a188", 1)
df = df.drop("in13a189", 1)
df = df.drop("in13a190", 1)
df = df.drop("in13a191", 1)
df = df.drop("filter_$", 1)
df = df.drop("slaap6_r", 1)

#---------Drop Day Remarks and Unusable External Factor Specifics -----#
df = df.drop("Extern_nl_dag1", 1)
df = df.drop("Andersnl_dag1", 1)
df = df.drop("opm_dag1", 1)
df = df.drop("evaopm_dag1", 1)
df = df.drop("Extern_nl_dag2", 1)
df = df.drop("Andersnl_dag2", 1)
df = df.drop("opm_dag2", 1)
df = df.drop("evaopm_dag2", 1)
df = df.drop("Extern_nl_dag3", 1)
df = df.drop("Andersnl_dag3", 1)
df = df.drop("opm_dag3", 1)
df = df.drop("evaopm_dag3", 1)
df = df.drop("Extern_nl_dag4", 1)
df = df.drop("Andersnl_dag4", 1)
df = df.drop("opm_dag4", 1)
df = df.drop("evaopm_dag4", 1)
df = df.drop("Extern_nl_dag5", 1)
df = df.drop("Andersnl_dag5", 1)
df = df.drop("opm_dag5", 1)
df = df.drop("evaopm_dag5", 1)
df = df.drop("Extern_nl_dag6", 1)
df = df.drop("Andersnl_dag6", 1)
df = df.drop("opm_dag6", 1)
df = df.drop("evaopm_dag6", 1)
df = df.drop("Extern_nl_dag7", 1)
df = df.drop("Andersnl_dag7", 1)
df = df.drop("opm_dag7", 1)
df = df.drop("evaopm_dag7", 1)

#--------------- Drop Partner Data (Not Enough) and Bedtime Data per day (Not to be used as predictors) ------------#
df = df.drop("GeplandeBedtijd_dag1", 1)
df = df.drop("EchteBedtijd_dag1", 1)
df = df.drop("Slaaptijd_dag1", 1)
df = df.drop("Opstaan_dag1", 1)
df = df.drop("Bedtijd_partner_dag1", 1)
df = df.drop("Slaaptijd_partner_dag1", 1)
df = df.drop("Opstaan_partner", 1)
df = df.drop("GeplandeBedtijd_dag2", 1)
df = df.drop("EchteBedtijd_dag2", 1)
df = df.drop("Slaaptijd_dag2", 1)
df = df.drop("Opstaan_dag2", 1)
df = df.drop("Bedtijd_partner_dag2", 1)
df = df.drop("Slaaptijd_partner_dag2", 1)
df = df.drop("Opstaan_partner_dag2", 1)
df = df.drop("GeplandeBedtijd_dag3", 1)
df = df.drop("EchteBedtijd_dag3", 1)
df = df.drop("Slaaptijd_dag3", 1)
df = df.drop("Opstaan_dag3", 1)
df = df.drop("Bedtijd_partner_dag3", 1)
df = df.drop("Slaaptijd_partner_dag3", 1)
df = df.drop("Opstaan_partner_dag3", 1)
df = df.drop("GeplandeBedtijd_dag4", 1)
df = df.drop("EchteBedtijd_dag4", 1)
df = df.drop("Slaaptijd_dag4", 1)
df = df.drop("Opstaan_dag4", 1)
df = df.drop("Bedtijd_partner_dag4", 1)
df = df.drop("Slaaptijd_partner_dag4", 1)
df = df.drop("Opstaan_partner_dag4", 1)
df = df.drop("GeplandeBedtijd_dag5", 1)
df = df.drop("EchteBedtijd_dag5", 1)
df = df.drop("Slaaptijd_dag5", 1)
df = df.drop("Opstaan_dag5", 1)
df = df.drop("Bedtijd_partner_dag5", 1)
df = df.drop("Slaaptijd_partner_dag5", 1)
df = df.drop("Opstaan_partner_dag5", 1)
df = df.drop("GeplandeBedtijd_dag6", 1)
df = df.drop("EchteBedtijd_dag6", 1)
df = df.drop("Slaaptijd_dag6", 1)
df = df.drop("Opstaan_dag6", 1)
df = df.drop("Bedtijd_partner_dag6", 1)
df = df.drop("Slaaptijd_partner_dag6", 1)
df = df.drop("Opstaan_partner_dag6", 1)
df = df.drop("GeplandeBedtijd_dag7", 1)
df = df.drop("EchteBedtijd_dag7", 1)
df = df.drop("Slaaptijd_dag7", 1)
df = df.drop("Opstaan_dag7", 1)
df = df.drop("Bedtijd_partner_dag7", 1)
df = df.drop("Slaaptijd_partner_dag7", 1)
df = df.drop("Opstaan_partner_dag7", 1)
df = df.drop("in13a026", 1)
df = df.drop("in13a028", 1)
df = df.drop("in13a030", 1)
df = df.drop("in13a052", 1)
df = df.drop("in13a054", 1)
df = df.drop("in13a056", 1)
df = df.drop("in13a078", 1)
df = df.drop("in13a080", 1)
df = df.drop("in13a082", 1)
df = df.drop("in13a104", 1)
df = df.drop("in13a106", 1)
df = df.drop("in13a108", 1)
df = df.drop("in13a130", 1)
df = df.drop("in13a132", 1)
df = df.drop("in13a134", 1)
df = df.drop("in13a156", 1)
df = df.drop("in13a158", 1)
df = df.drop("in13a160", 1)
df = df.drop("in13a182", 1)
df = df.drop("in13a184", 1)
df = df.drop("in13a186", 1)
df = df.drop("Verschil_bedtijd_week", 1)
df = df.drop("Verschil_bedtijd_5dagen", 1)
df = df.drop("Verschil_bedtijd_weekend", 1)

#--------Drop Individual External Affects per Day (External_geldig suffices for now)--------#
df = df.drop("Slaapduur_dag1", 1)
df = df.drop("TV_dag1", 1)
df = df.drop("Computer_dag1", 1)
df = df.drop("Huishoud_dag1",1)
df = df.drop("Buitenshuis_dag1",1)
df = df.drop("Sociaal_dag1",1)
df = df.drop("Anders_dag1",1)
df = df.drop("Bezigheid_dag1",1)
df = df.drop("Extern_dag1",1)

df = df.drop("Slaapduur_dag2", 1)
df = df.drop("TV_dag2", 1)
df = df.drop("Computer_dag2", 1)
df = df.drop("Huishoud_dag2",1)
df = df.drop("Buitenshuis_dag2",1)
df = df.drop("Sociaal_dag2",1)
df = df.drop("Anders_dag2",1)
df = df.drop("Bezigheid_dag2",1)
df = df.drop("Extern_dag2",1)

df = df.drop("Slaapduur_dag3", 1)
df = df.drop("TV_dag3", 1)
df = df.drop("Computer_dag3", 1)
df = df.drop("Huishoud_dag3",1)
df = df.drop("Buitenshuis_dag3",1)
df = df.drop("Sociaal_dag3",1)
df = df.drop("Anders_dag3",1)
df = df.drop("Bezigheid_dag3",1)
df = df.drop("Extern_dag3",1)

df = df.drop("Slaapduur_dag4", 1)
df = df.drop("TV_dag4", 1)
df = df.drop("Computer_dag4", 1)
df = df.drop("Huishoud_dag4",1)
df = df.drop("Buitenshuis_dag4",1)
df = df.drop("Sociaal_dag4",1)
df = df.drop("Anders_dag4",1)
df = df.drop("Bezigheid_dag4",1)
df = df.drop("Extern_dag4",1)

df = df.drop("Slaapduur_dag5", 1)
df = df.drop("TV_dag5", 1)
df = df.drop("Computer_dag5", 1)
df = df.drop("Huishoud_dag5",1)
df = df.drop("Buitenshuis_dag5",1)
df = df.drop("Sociaal_dag5",1)
df = df.drop("Anders_dag5",1)
df = df.drop("Bezigheid_dag5",1)
df = df.drop("Extern_dag5",1)

df = df.drop("Slaapduur_dag6", 1)
df = df.drop("TV_dag6", 1)
df = df.drop("Computer_dag6", 1)
df = df.drop("Huishoud_dag6",1)
df = df.drop("Buitenshuis_dag6",1)
df = df.drop("Sociaal_dag6",1)
df = df.drop("Anders_dag6",1)
df = df.drop("Bezigheid_dag6",1)
df = df.drop("Extern_dag6",1)

df = df.drop("Slaapduur_dag7", 1)
df = df.drop("TV_dag7", 1)
df = df.drop("Computer_dag7", 1)
df = df.drop("Huishoud_dag7",1)
df = df.drop("Buitenshuis_dag7",1)
df = df.drop("Sociaal_dag7",1)
df = df.drop("Anders_dag7",1)
df = df.drop("Bezigheid_dag7",1)
df = df.drop("Extern_dag7",1)


binaryAttributesToEncode = [
    "Nachtdienst",
    "Reason_nvt",
    "Extern_dag1_geldig",
    "Extern_dag2_geldig",
    "Extern_dag3_geldig",
    "Extern_dag4_geldig",
    "Extern_dag5_geldig",
    "Extern_dag6_geldig",
    "Extern_dag7_geldig",
    ]

timeAttributesToEncode = [
    'Verschil_bedtijd_dag1',
    'Verschil_bedtijd_dag2',
    'Verschil_bedtijd_dag3',
    'Verschil_bedtijd_dag4',
    'Verschil_bedtijd_dag5',
    'Verschil_bedtijd_dag6',
    'Verschil_bedtijd_dag7']

slaapAttributesWithEffectAttribute = [
    ['Slaap1', 'Slaap1a'],
    ['Slaap2', 'Slaap2a'],
    ['Slaap3', 'Slaap3a'],
    ['Slaap4', 'Slaap4a'],
    ['Slaap5', 'Slaap5a']]


bedtimeProcV2Attributes = [
    "v2.bedproc1",
    "v2.bedproc2",
    "v2.bedproc3",
    "v2.bedproc4",
    "v2.bedproc5",
    "v2.bedproc6",
    "v2.bedproc7",
    "v2.bedproc8",
    "v2.bedproc9",
    "v2.bedproc2_r",
    "v2.bedproc3_r",
    "v2.bedproc7_r",
    "v2.bedproc9_r",
    "v2.BedProc_SCALE"]

actualProcrastinationPerDay = [
    "Verschil_bedtijd_dag1",
    "Verschil_bedtijd_dag2",
    "Verschil_bedtijd_dag3",
    "Verschil_bedtijd_dag4",
    "Verschil_bedtijd_dag5",
    "Verschil_bedtijd_dag6",
    "Verschil_bedtijd_dag7",]


#---------Encoding Data in Dataframe (when able)--------#

def applyBinary(attribute):
    try:
        return int(attribute)
    except:
        return 2

def timeToMinutes(hhmmss):
    if hhmmss.strip():
        [hours, minutes, seconds] = [int(x) for x in hhmmss.split(':')]
        t = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return t.total_seconds() / 60
    else:
        return np.NaN

for attribute in binaryAttributesToEncode:
    df[attribute] = df[attribute].apply(lambda x: applyBinary(x))

for attribute in timeAttributesToEncode:
    df[attribute] = df[attribute].apply(lambda x: timeToMinutes(x))


#-----Missing Data Manipulation ---------#
df = df.replace(r'\s+', np.nan, regex=True) #replace empty values (empty strings in this case) with NaN for easier identification

for [att1, att2] in slaapAttributesWithEffectAttribute: # Map NaN multipliers with survey response
    range = df[att1].values.tolist()
    numericRange = [int(s) for s in range if s >= 0]
    min = np.nanmin(numericRange)
    df[att2] = df[att2].replace(np.nan, min)

#for attribute in bedtimeProcV2Attributes:
#     df = df.drop(attribute, 1)
#print(df.shape)

df = df.apply(pd.to_numeric) #ensure all data in frame is numeric

for attribute in actualProcrastinationPerDay: # removal of listings which have missing bedtime proc values
    df = df[pd.notnull(df[attribute])]

# drop remaining listings with null data (Total listings go from 2637 to 1226, 46.5% of te original set retained)
df = df.dropna()


#----- New features and replicating individual listings for each of the 7 days------#
df = pd.concat([df]*7, ignore_index=True)
df = df.sort(columns="nomem_encr")
df = df.reset_index(drop=True)
#df.index = np.arange(1,len(df)+1) #Make index start from 1 instead of 0, easier (for me) to increment over dataframe

df["Proc_soFar"] = 0 # float feature calculating total amount procrastinated up until current day
df["Procrastinated_Prev_Day"] = 0 # binary feature indicating whether participant procrastinated previous day (1) or not (0)

copy_df = df.copy(deep=True)

def rebuild_with_Derived_Proc_Features(data):
    altered = 0 #refers to index of entry currently being altered
    participants = data.shape[0]
    dfs = [] #array which will hold splits of dataset

    while altered < participants: # while loop to split the dataset by every 7 entries and...
        current = data[:7]
        data = data[7:]
        proc = 0
        day = 0
        prevProc = 0.0

        while day < 7:
            current.ix[altered, "Proc_soFar"] = proc #....set value for current listing based on proc's value
            if proc != prevProc:
                current.ix[altered, "Procrastinated_Prev_Day"] = 1 #... set binary feature according to prev day's proc
            prevProc = proc
            proc += float(current.iloc[[day]][actualProcrastinationPerDay[day]].values) # increase proc by current day's actual proc value
            altered += 1 # current entry alteration finished, increment to next
            day += 1

        dfs.append(current)

    return pd.concat(dfs) #rebuild dataset from splits

#print(df)
df = rebuild_with_Derived_Proc_Features(copy_df)
print(df)

#-----------------------------Split Dataframe into Train and Test Sets------------------------------#

#train, test = train_test_split(df, train_size = 0.7)

#ntrain = train.shape[0]
#ntest = test.shape[0]

#x_train = np.array(df[:train.shape[0]])
#x_test = np.array(df[train.shape[0]:])

#y_train = np.log(train[TARGET]+1)
#train.drop([TARGET], axis=1, inplace=True)
#kf = KFold(ntrain, n_folds=NFOLDS, shuffle=True, random_state=SEED)



#------ Model Training -------#
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

xgb_params = {
    'seed': 0,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.7,
    'learning_rate': 0.075,
    'objective': 'reg:linear',
    'max_depth': 4,
    'num_parallel_tree': 1,
    'min_child_weight': 1,
    'eval_metric': 'rmse',
    'nrounds': 500
}

def get_oof(model): # get best set up out of folds of cross validation
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        model.train(x_tr, y_tr)

        oof_train[test_index] = model.predict(x_te)
        oof_test_skf[i, :] = model.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

#xg = XgbWrapper(seed=SEED, params=xgb_params)
#xg_oof_train, xg_oof_test = get_oof(xg)

