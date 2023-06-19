import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.feature_selection import SelectKBest, f_classif

# ---------- Load the data ----------
t0 = time.time()
cleanedData = pd.read_csv("..\data\cleanedData.csv", index_col=0)
t1 = time.time()
print(f"Data loading time: {round(t1 - t0, 3)}s")

# ---------- Discretize the outcome ----------
#get a boxplot of the outcome distribution before discretizing
color = {"boxes": "Black", "whiskers": "Black", "medians": "DarkBlue", "caps": "Black"}
fig, axs = plt.subplots()
cleanedData[["SF2"]].plot.box(grid=True, ax=axs, color=color,  sym="r.").set(title='Outcome Distribution', ylabel='SF2 value')
plt.tight_layout()
plt.savefig('..\plots\BoxplotOutcome.png')
plt.show()


#get the rows with the least and greatest sensitivity for the Manhattan plots
max_cellLine = cleanedData["SF2"].sort_values(ascending=False)
min_cellLine = cleanedData["SF2"].sort_values(ascending=True)
print(max_cellLine)
print(min_cellLine)

#discretize
column_data = cleanedData['SF2']
binarizer = Binarizer(threshold=0.5)
binarized_data = binarizer.transform(column_data.values.reshape(-1, 1))
cleanedData['SF2'] = binarized_data.flatten()
print(cleanedData)

# ----------- Check if dataset is balanced -----------
balance = cleanedData["SF2"].mean()
print(f"The average outcome is {balance}")
#--> Dataset is not balanced: use stratified samples to counter this

# Create a bar plot to visualize the distribution
value_counts = cleanedData['SF2'].value_counts()

categories = ['responders', 'non-responders']
counts = [value_counts[0], value_counts[1]]

plt.bar(categories, counts)
plt.xlabel('Discretized Categories')
plt.ylabel('Count')
plt.title('Distribution of Outcome (SF2)')
#plt.text('Mean: ', round(balance,2))
plt.text(-0.3,300, f"Mean: {round(balance,3)}", fontsize=10, bbox=dict(facecolor="white", edgecolor= "black", boxstyle = "round"))
plt.savefig('..\plots\Distribution of Outcome (SF2).png')
plt.show()

# ---------- Prepare Data for the feature selection ----------
y = cleanedData['SF2']
x = cleanedData.drop('SF2', axis=1)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0) #random state for reproduceability
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# ---------- Perform univariate feature selection ----------
selector = SelectKBest(score_func=f_classif, k='all')
X_selected_train = pd.DataFrame(selector.fit_transform(X_train_sc, y_train), index=X_train.index, columns=X_train.columns)
X_selected_test = pd.DataFrame(selector.transform(X_test_sc), index=X_test.index, columns=X_test.columns)
print(X_selected_train)
print(X_selected_test)
p_values = selector.pvalues_
print(p_values)
selected_feature_mask = p_values < 0.1

keepColumns = X_selected_train.columns[selected_feature_mask]
#Check if indices of train and test are identical
#if keepColumns_test.all() == keepColumns_train.all():
#    print(f"The indices of the dataframes are identical")
#else:
#    print(f"Check indices")
modelData_train = X_selected_train[keepColumns]
print(modelData_train)
modelData_test = X_selected_test[keepColumns]
manPlotData = modelData_train.append(modelData_test, ignore_index=False)
modelData_train.loc[:, 'SF2'] = pd.Categorical(y_train, categories=[0, 1])
modelData_test.loc[:, 'SF2'] = pd.Categorical(y_test, categories=[0, 1])
print(modelData_train)
print(modelData_test)

# ---------- Extract CSV with cleaned data ----------
modelData_train.to_csv("..\data\ModelData_train.csv", index=True)
modelData_test.to_csv("..\data\ModelData_test.csv", index=True)

# ---------- Generate two example Manhatten plots ----------
for a in min_cellLine.index:
    if a in manPlotData.index:
        min_cellLine = a
        break
for b in max_cellLine.index:
    if b in manPlotData.index:
        max_cellLine = b
        break
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(28, 12))
ax[0].bar(manPlotData.columns, manPlotData.loc[min_cellLine])
ax[0].set_xticks(np.arange(len(manPlotData.columns)), labels=manPlotData.columns, rotation=90, fontsize='xx-small', linespacing=2)
ax[0].set_title("Least Resistant cell-line gene expression", fontsize=20)
ax[0].set_xlabel("Gene", fontsize=16)
ax[0].set_ylabel("Expression (scaled)", fontsize=16)

ax[1].bar(manPlotData.columns, manPlotData.loc[max_cellLine])
ax[1].set_xticks(np.arange(len(manPlotData.columns)), labels=manPlotData.columns, rotation=90, fontsize='xx-small', linespacing=2)
ax[1].set_title("Most Resistant cell-line gene expression", fontsize=20)
ax[1].set_xlabel("Gene", fontsize=16)
ax[1].set_ylabel("Expression (scaled)", fontsize=16)

for subplot in ax.flatten():
    subplot.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('..\plots\Manhatten_plots.png')

# ---------- Plot the p-values -----------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
ax.hist(p_values, bins=100, edgecolor='black')
ax.set_title("p-Value Distribution", fontsize=20)
ax.set_xlabel("p-Value", fontsize=16)
ax.set_ylabel("Frequency", fontsize=16)
plt.tight_layout()
plt.savefig('..\plots\FeatureSelectionPvalues.png')

# ------------- visualize top 20 genes representation -------------------
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
scores = selector.scores_
top_scoresValues = scores[0:20]
for i in range(19, len(scores)):
    top_scoresValues = np.sort(top_scoresValues)
    if scores[i] > top_scoresValues[0]:
        top_scoresValues[0] = scores[i]
top_scoresIndices = []
for j in range(0, len(scores)):
    for k in range(len(top_scoresValues)):
        if top_scoresValues[k] == scores[j]:
            top_scoresIndices.append(j)
top_scoresIndices = np.array(top_scoresIndices)
ax.bar(np.arange(20), top_scoresValues)
ax.set_xticks(np.arange(20), X_selected_train.iloc[:, top_scoresIndices].columns, rotation=90)
ax.set_title("20 most important genes", fontsize=20)
ax.set_xlabel("Gene", fontsize=16)
ax.set_ylabel("Importance (F-test score)", fontsize=16)
plt.tight_layout()
plt.savefig('..\plots\FeatureSelectionTop20genes.png')
