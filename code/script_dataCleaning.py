import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as sts
import time as time
import seaborn as sns

from sklearn.preprocessing import StandardScaler


# ---------- Load the data ----------
t0 = time.time()
dataGeneExpression = pd.read_csv("..\data\Project_CCLE\expressionData.csv", index_col=0)
dataRadiosensitivity = pd.read_csv("..\data\Project_CCLE\sensitivity.csv", index_col=0)
t1 = time.time()
print(f"Data loading time: {round(t1 - t0, 3)}s")


# ---------- Get information about the data ----------
#get the shape of the data
print(f"There are {dataGeneExpression.shape[1]} genes in the dataset.")
print(f"These genes are from {dataGeneExpression.shape[0]} observations.")
print(f"There are {dataRadiosensitivity.shape[1]} observed outcomes in the dataset.")
print(f"These outcomes are from {dataRadiosensitivity.shape[0]} observations.")
if dataGeneExpression.index.all() == dataRadiosensitivity.index.all():
    print(f"The indices of the dataframes are identical")
else:
    print(f"Check indices")

#check for missing values
num_missing_dataGeneExpression= dataGeneExpression.isna().sum().sum()
print(f"There are {num_missing_dataGeneExpression} missing values in the Genes Dataset")
num_missing_dataRadiosensitivity= dataRadiosensitivity.isna().sum().sum()
print(f"There are {num_missing_dataRadiosensitivity} missing values in the Radiosensitivity Dataset")

#check for duplicates
print('checking for duplicates....')
duplicatesGeneExpression_Columns = dataGeneExpression.columns.duplicated()
print(f"There are {sum(duplicatesGeneExpression_Columns)} columns duplicated in the GeneExpression Dataset")
duplicatesGeneExpression_Rows= dataGeneExpression.duplicated()
print(f"There are {sum(duplicatesGeneExpression_Rows)} rows duplicated in the GeneExpression Dataset")
duplicatesRadiosensitivity_columns = dataRadiosensitivity.columns.duplicated()
print(f"There are {sum(duplicatesRadiosensitivity_columns)} columns duplicated in the Radiosensitivity Dataset")
duplicatesRadiosensitivity_rows = dataRadiosensitivity.duplicated()
print(f"There are {sum(duplicatesRadiosensitivity_rows)} rows duplicated in the Radiosensitivity Dataset")

#check the outcome values
print(f"Outcome values reach from {round(dataRadiosensitivity['SF2'].min(), 3)} to {round(dataRadiosensitivity['SF2'].max(), 3)}")

# ---------- Clean the data ----------
#check for meaningless values & delete them
shapebefore = dataGeneExpression.shape
count_unique_values = dataGeneExpression.nunique()
count_columns_fullwith0 = 0
for i in dataGeneExpression.columns:
    if count_unique_values.loc[i] == 1:
        count_columns_fullwith0 += 1
print(f"There are {count_columns_fullwith0} columns with 0 (meaningless) in the Genes Dataset")
for i in dataGeneExpression.columns:
    if count_unique_values.loc[i] == 1:
        dataGeneExpression = dataGeneExpression.drop(i, axis=1)
#check the dataset
shapeafter = dataGeneExpression.shape
print(f"{shapebefore[1]-shapeafter[1]} columns with 0 have been deleted")

#check for highly correlated genes
#Standardize the numerical features
sc = StandardScaler()
X_standardArray = sc.fit_transform(dataGeneExpression)
#create a dataframe from the standardized data
X_standard = pd.DataFrame(X_standardArray, columns=dataGeneExpression.columns, index=dataGeneExpression.index)
#prepare the correlation comparison
col = X_standard.columns
#corrData = pd.DataFrame(0, columns=col, index=col) #Datafrome to save the different correlation coefficients
# Iterate over every column in the standardized data
t0 = time.time()
corrAr = np.array(X_standard.corr(method='spearman'))
corrAr[np.triu_indices_from(corrAr, k=0)] = np.nan #set all values above the diagonal to NA (including diagonal)
corrData = pd.DataFrame(corrAr, index=X_standard.columns, columns=X_standard.columns)
print(corrData)

#plot the correlation matrix as a heatmap for the first few columns
plt.figure(figsize=(15, 15))
sns.heatmap(corrData.iloc[0:10, 0:10], cmap='RdYlBu', center=0, cbar=True, square=True, xticklabels=True, yticklabels=True, annot=True, annot_kws={'size': 10}, linecolor="Black", linewidths=0.5)
plt.title('Heatmap of Correlations (first 10 Genes only)', fontsize=22, fontweight='bold')
plt.tight_layout()
plt.savefig('..\plots\exampleHeatmap.png')
plt.show()

t1 = time.time()
print(f"Time to prepare the correlations: {t1 - t0}s")

# ---------- Convert correlation dataframe (matrix) for the histogram -----------
corrDataflatten = corrData.values.flatten()
corrDataflatten = corrDataflatten[~np.isnan(corrDataflatten)] #drop nan values, as we don't want them for the plot
print(f"{len(corrDataflatten)} correlation coefficients have been computed")

# ---------- Plot the correlation coefficients -----------
# Plot the histogram
plt.hist(corrDataflatten, bins=100, edgecolor='black')
plt.xlabel('Correlation Values')
plt.ylabel('Frequency')
plt.title('Distribution of Correlation Values')
plt.savefig('..\plots\DistributionCorrelationValues.png')
plt.show()
print(f"Summary of the Correlation Distribution of all Genes:")
mean = np.mean(corrDataflatten)
std = np.std(corrDataflatten)
min_value = np.min(corrDataflatten)
max_value = np.max(corrDataflatten)
percentiles = np.percentile(corrDataflatten, [1, 5, 25, 50, 75, 95, 99,])
_, p_valueNorm = sts.shapiro(corrDataflatten)
print("Mean: {:.7f}".format(mean))
print("Standard Deviation: {:.7f}".format(std))
print("Minimum Value: {:.7f}".format(min_value))
print("Maximum Value: {:.7f}".format(max_value))
print("Percentiles: ", percentiles)
print(f"p-value for normal distribution {p_valueNorm}")


# ----------- Drop Correlated Genes -------------------
t0=time.time()
threshold = mean + 2*std #keep it that way since correlations are slightly imbalanced and we want to eliminate this
print(f"The calculated threshold is: {threshold}")
corrColumns = corrData.columns[corrData.gt(threshold).any()]
corrColumns.append(corrData.columns[corrData.lt(- threshold).any()])
cleanedExpression = dataGeneExpression.drop(columns=corrColumns)
t1=time.time()
print(f'Time for dropping correlated genes {round(t1-t0,3)}s')
print(f"{len(corrColumns)} genes have been dropped due to cocorrelation (with threshold: {threshold})")

# ----------- New Dataframe appended with outcomes -------------------
cleaned_features_with_outcome = pd.concat([cleanedExpression, dataRadiosensitivity['SF2']], axis=1)
#adds the column (axis=1) from dataradiosensitivity to cleaned dataset ('cleanedExpression'),
#mind: should have the same index, here cell lines (check in the beginning was successful)

# ---------- Save the dataframe for feature selection ----------
t0 = time.time()
cleaned_features_with_outcome.to_csv("..\data\cleanedData.csv", index=True)
t1 = time.time()
print(f"Time to save csv files: {t1 - t0}s")
