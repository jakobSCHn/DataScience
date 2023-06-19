import pandas as pd
import time as time

# ----------- Load the data ----------
t0 = time.time()
modelData_test = pd.read_csv("..\Data\ModelData_test.csv", index_col=0)
patientInfo = pd.read_csv("..\Data\Project_CCLE\sample_info.csv", index_col=1)
t1 = time.time()
print(f"Data loading time: {round(t1 - t0, 3)}s")
print(modelData_test.head())

# ---------- Retrieve some information about the data ----------
print(patientInfo.head())
print(f"There are {patientInfo.shape[1]} stored attributes about the samples.")
print(f"These attributes are stored for {patientInfo.shape[0]} samples.")
unique_primary_disease = patientInfo["primary_disease"].unique() #check if unknown is present
number_primary_disease = len(unique_primary_disease)
print(f"There are {number_primary_disease-1} different primary disease along all cell lines")
print(f"For some cell lines the primary disease is unknown")
missing_values_primary_disease = patientInfo["primary_disease"].isna().sum()
print(f"There are {missing_values_primary_disease} missing values in the primary disease column")
primary_disease_counts = patientInfo["primary_disease"].value_counts()
top_5_diseases = primary_disease_counts.head(5)
print("Top 5 primary diseases:")
print(top_5_diseases)

# ---------- Get the cancer types of the test set ----------
overlappingData = patientInfo.loc[patientInfo.index.isin(modelData_test.index)]
value_counts = overlappingData['primary_disease'].value_counts()
print(value_counts)

cellLines = value_counts.index
lungCancerLines = overlappingData.loc[overlappingData['primary_disease'] == cellLines[0]].index
liverCancerLines = overlappingData.loc[overlappingData['primary_disease'] == cellLines[1]].index
HandNCancerLines = overlappingData.loc[overlappingData['primary_disease'] == cellLines[2]].index
ovarianCancerLines = overlappingData.loc[overlappingData['primary_disease'] == cellLines[3]].index
skinCancerLines = overlappingData.loc[overlappingData['primary_disease'] == cellLines[4]].index

testLungCancer = modelData_test.loc[lungCancerLines]
testLiverCancer = modelData_test.loc[liverCancerLines]
testHandNCancer = modelData_test.loc[HandNCancerLines]
testOvarianCancer = modelData_test.loc[ovarianCancerLines]
testSkinCancer = modelData_test.loc[skinCancerLines]

# ---------- Check if these new data sets are roughly balanced ----------
print(testLungCancer['SF2'].mean())
print(testLiverCancer['SF2'].mean()) #poorly balanced
print(testHandNCancer['SF2'].mean())
print(testOvarianCancer['SF2'].mean())
print(testSkinCancer['SF2'].mean())

# ---------- Safe the new test datasets as csv files ----------
testLungCancer.to_csv("..\data\Testset_LungCancer.csv", index=True)
testLiverCancer.to_csv("..\data\Testset_LiverCancer.csv", index=True)
testHandNCancer.to_csv("..\data\Testset_HandNCancer.csv", index=True)
testOvarianCancer.to_csv("..\data\Testset_OvarianCancer.csv", index=True)
testSkinCancer.to_csv("..\data\Testset_SkinCancer.csv", index=True)
