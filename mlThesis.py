# encoding: utf-8

import numpy as np  # linear algebra library
import pandas as pd  # data processing library, converts data into data frames and allows for manipulation of these frames
import datetime
import os

pd.set_option('expand_frame_repr', False)  # shows dataframe without wrapping to next line
pd.set_option('chained_assignment', None)
pd.set_option('display.max_columns', 500)

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
data_file = os.path.join(__location__, 'LISS_bedtime.csv')

df = pd.read_csv(data_file)

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
df = df.drop("im13a001", 1)
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
    "Verschil_bedtijd_dag6",  # weekend
    "Verschil_bedtijd_dag7"]  # weekend

mediaAttributes = [
    "TV_dag",
    "Computer_dag"]

dayAttributes = [
    "Verschil_bedtijd_dag",
    "Extern_dag",
    "Slaapduur_dag",
    "TV_dag",
    "Computer_dag",
    "Huishoud_dag",
    "Buitenshuis_dag",
    "Sociaal_dag",
    "Anders_dag",
    "Bezigheid_dag",
    "Extern_geldig_dag",
    "anticip_dag"]

weekendAttributes = [
    "Verschil_bedtijd_dag6",
    "Extern_dag6",
    "Extern_geldig_dag6",
    "Slaapduur_dag6",
    "TV_dag6",
    "Computer_dag6",
    "Huishoud_dag6",
    "Buitenshuis_dag6",
    "Sociaal_dag6",
    "Anders_dag6",
    "Bezigheid_dag6",
    "anticip_dag6",
    "Verschil_bedtijd_dag7",
    "Extern_dag7",
    "Extern_geldig_dag7",
    "Slaapduur_dag7",
    "TV_dag7",
    "Computer_dag7",
    "Huishoud_dag7",
    "Buitenshuis_dag7",
    "Sociaal_dag7",
    "Anders_dag7",
    "Bezigheid_dag7",
    "anticip_dag7",
    "slaapduur_weekend",
    "Slaapuren_weekend",
    "Slaapuren_weekend_voldoende"]

def dropWeekend(dataframe):
    for attribute in weekendAttributes:
        dataframe = dataframe.drop(attribute, 1)
    return dataframe

def dropProcSurveyV2(dataframe):
    for attribute in bedtimeProcV2Attributes:
        dataframe = dataframe.drop(attribute, 1)
    return dataframe

#---------Encoding Data in Dataframe (when able)--------#

def applyBinary(attribute):
    try:
        return int(attribute)
    except:
        return 2

def timeToMinutes(hhmmss):
    if hhmmss.strip():
        [hours, minutes, seconds] = [float(x) for x in hhmmss.split(':')]
        if hours < 0:
            minutes *= -1
            seconds *= -1
        t = datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        return t.total_seconds() / 60
    else:
        return np.NaN

df["anticip_dag1"] = 0
df["anticip_dag2"] = 0
df["anticip_dag3"] = 0
df["anticip_dag4"] = 0
df["anticip_dag5"] = 0
df["anticip_dag6"] = 0
df["anticip_dag7"] = 0

#anticipAttributes = [
#    "anticip_dag1",
#    "anticip_dag2",
#    "anticip_dag3",
#    "anticip_dag4",
#    "anticip_dag5",
#    "anticip_dag6",
#    "anticip_dag7"]

externGeldigAttributes = [
    "Extern_dag1_geldig",
    "Extern_dag2_geldig",
    "Extern_dag3_geldig",
    "Extern_dag4_geldig",
    "Extern_dag5_geldig",
    "Extern_dag6_geldig",
    "Extern_dag7_geldig"]

for attribute in binaryAttributesToEncode:
    df[attribute] = df[attribute].apply(lambda x: applyBinary(x))

for attribute in timeAttributesToEncode:
    df[attribute] = df[attribute].apply(lambda x: timeToMinutes(x))

dayNumber = 1
for attribute in externGeldigAttributes: # rename feature for consistency
    df["Extern_geldig_dag"+str(dayNumber)] = df[attribute]
    dayNumber += 1
    df = df.drop(attribute, 1)

for attribute in actualProcrastinationPerDay: # outlier removal based on unlikely procrastination / anti-procrastination levels
    while (1):
        x = df[attribute].median()
        ix = abs(df[attribute] - x) > (5 * df[attribute].std()) # removes outliers with a standard deviation 5 times the median difference (adjustable)
        if ix.sum() == 0:  # no more outliers -> stop
            break
        df.loc[ix, attribute] = np.nan  # exclude outliers

#-----Missing Data Manipulation ---------#
df = df.replace(r'\s+', np.nan, regex=True) #replace empty values (empty strings in this case) with NaN for easier identification

for [att1, att2] in slaapAttributesWithEffectAttribute: # Map NaN multipliers with survey response
    range = df[att1].values.tolist()
    numericRange = [float(s) for s in range if float(s) >= 0]
    min = np.nanmin(numericRange)
    df[att2] = df[att2].replace(np.nan, min)

df = df.apply(pd.to_numeric) #ensure all data in frame is numeric

for attribute in actualProcrastinationPerDay: # removal of listings which have missing bedtime proc values
    df = df[pd.notnull(df[attribute])]

# drop remaining listings with null data (Total listings go from 2637 to 1159, 43.95% of the original set retained)
df = df.dropna()
print(df.shape)

#------------------Optional Attribute Drops (used in iterative experimentation)--------------------#
df = dropProcSurveyV2(df)
df = dropWeekend(df)

#----- New features and replicating individual listings for each of the 5 days------#
df = pd.concat([df]*5, ignore_index=True) # number represents amount of days in week to check, change to 7 for full week
df = df.sort_values(by="nomem_encr")
df = df.reset_index(drop=True)

df["Proc_soFar_dag"] = 0 # float feature calculating total amount procrastinated up until current day
df["Procrastinated_Prev_dag"] = 0 # binary feature indicating whether participant procrastinated previous day (1) or not (0)
df["Early_To_Bed_Prev_dag"] = 0 # binary feature indicating whether participant went to sleep earlier than expected previous day
df["Media_Usage_dag"] = 0 # binary feature indicating whether participant used media before attempting to sleep
df["dag_Number"] = 0
df["Bezigheid_Work"] = 0
df["Bezigheid_Study"] = 0
df["Bezigheid_Household"] = 0
df["Bezigheid_Free_Time"] = 0

def rebuild_with_Derived_Proc_Features(data):
    altered = 0 # refers to index of entry currently being altered
    participants = data.shape[0]
    dfs = [] # array which will hold splits of dataset

    while altered < participants: # while loop to split the dataset by every 5 entries and...
        current = data[:5]
        data = data[5:]
        currentProc = 0
        day = 0
        prevProc = 0.0
        totalProc = 0

        while day < 5:
            dayString = str(day+1)
            current.ix[altered, "dag_Number"] = day+1
            current.ix[altered, "Proc_soFar_dag"] = totalProc #....set value for current listing based on proc's value

            if currentProc != prevProc and currentProc > prevProc:
                current.ix[altered, "Procrastinated_Prev_dag"] = 1 #... set binary feature according to prev day's proc

            elif currentProc != prevProc and currentProc < prevProc:
                current.ix[altered, "Early_To_Bed_Prev_dag"] = 1 # set only if proc decreases day to day

            if current.ix[altered, "TV_dag"+str(day+1)] > 0 or current.ix[altered, "Computer_dag"+str(day+1)] > 0:
                current.ix[altered, "Media_Usage_dag"] = 1

            prevProc = currentProc
            currentProc += float(current.iloc[[day]][actualProcrastinationPerDay[day]].values) # increase proc by current day's actual proc value

           # if current.ix[altered, actualProcrastinationPerDay[day]] < 0: # Separate days procrastinated from days not
           #     current.ix[altered, anticipAttributes[day]] = (current.ix[altered, actualProcrastinationPerDay[day]] * -1)
           #     current.ix[altered, actualProcrastinationPerDay[day]] = 0

            totalProc += float(current.iloc[[day]][actualProcrastinationPerDay[day]].values) # increase total proc so far based on days procrastinated

            for attribute in dayAttributes: # introduces general attribute in place of specific day attributes,
                                            # reducing dataset column size and improving readability
                actualAttributeString = attribute + dayString
                attributeValue = float(current.iloc[[day]][actualAttributeString].values)
                current.ix[altered, attribute] = attributeValue # creates the new generic attribute and gives value of specific day
                current = current.drop(actualAttributeString, 1) # removes now obsolete day attributes from each listing

            #------ Convert categorical feature bezigheid_dag into 4 binary features-------#
            if current.ix[altered, "Bezigheid_dag"] == 1.0:
                current.ix[altered, "Bezigheid_Work"] = 1
            elif current.ix[altered, "Bezigheid_dag"] == 2.0:
                current.ix[altered, "Bezigheid_Study"] = 1
            elif current.ix[altered, "Bezigheid_dag"] == 3.0:
                current.ix[altered, "Bezigheid_Household"] = 1
            elif current.ix[altered, "Bezigheid_dag"] == 4.0:
                current.ix[altered, "Bezigheid_Free_Time"] = 1

            altered += 1 # current entry alteration finished, increment to next full datset index
            day += 1 # increments to next day of week

        dfs.append(current)
    return pd.concat(dfs) #rebuild dataset from splits

copy_df = df.copy(deep=True)
df = rebuild_with_Derived_Proc_Features(copy_df)
df = df.drop("anticip_dag", 1) ## removes anticipation time from dataset (optional drop for now)
df = df.drop("Bezigheid_dag", 1)
#df.to_csv(os.path.join(__location__, 'LISS_bedtime_final.csv'))   ## save reformatted dataset