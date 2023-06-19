import pandas as pd
import time as time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import functions_project5

# ----------- Load the data ----------
t0 = time.time()
modelData_train = pd.read_csv("..\Data\ModelData_train.csv", index_col=0)
modelData_test = pd.read_csv("..\Data\ModelData_test.csv", index_col=0)
t1 = time.time()
print(f"Data loading time: {round(t1 - t0, 3)}s")
print(modelData_train.head())
print(modelData_test.head())

# ---------- Prepare the data for the fit ----------
y_train = modelData_train['SF2']
y_test = modelData_test['SF2']
X_train = modelData_train.drop('SF2', axis=1)
X_test = modelData_test.drop('SF2', axis=1)

# ---------- Fit the model once with default and once with optimized parameters ----------
#choose Parameters to optimize
n_estimators = [20, 50, 100, 200]  #higher numbers not possible due to computing time
criterion = ['gini', 'entropy', 'log_loss']
max_features = ['sqrt', 'log2', None]
max_depth = [10, 20, 50, 100, None]
#no more parameters could be optimized du to computing limits (max_depth)
optParameters = dict(n_estimators=n_estimators, criterion=criterion, max_features=max_features)

t0 = time.time()
RF_default = RandomForestClassifier(class_weight='balanced', random_state=0)
RF_optimized = GridSearchCV(RF_default, optParameters, cv=5, scoring='roc_auc')
RF_default.fit(X_train, y_train)
RF_optimized.fit(X_train, y_train)
bestModel = RF_optimized.best_estimator_
print('Default Parameters: ', )
print('Best Parameters:', RF_optimized.best_estimator_.get_params())
t1 = time.time()
print(f"Modell fitting time: {round(t1 - t0, 3)}s")

# ---------- Model evaluation ----------
#make predictions
y_predDefault = RF_default.predict(X_test)
y_predOpt = RF_optimized.best_estimator_.predict(X_test)
#predict probablities for ROC curve
y_preProbaDefault = RF_default.predict_proba(X_test)[:, 1]
y_preProbaOpt = RF_optimized.best_estimator_.predict_proba(X_test)[:, 1]
#get the metrics with a function
metricsDef, metricsOpt = functions_project5.evaluation(y_test, y_predDefault, y_preProbaDefault, y_predOpt, y_preProbaOpt, "RandomForrest")

# ---------- Evaluate the model with the specific cancer types ----------
testLungCancer = pd.read_csv("..\data\Testset_LungCancer.csv", index_col=0)
testLiverCancer = pd.read_csv("..\data\Testset_LiverCancer.csv", index_col=0)
testHandNCancer = pd.read_csv("..\data\Testset_HandNCancer.csv", index_col=0)
testOvarianCancer = pd.read_csv("..\data\Testset_OvarianCancer.csv", index_col=0)
testSkinCancer = pd.read_csv("..\data\Testset_SkinCancer.csv", index_col=0)

y_testLungCancer = testLungCancer['SF2']
y_testLiverCancer = testLiverCancer['SF2']
y_testHandNCancer = testHandNCancer['SF2']
y_testOvarianCancer = testOvarianCancer['SF2']
y_testSkinCancer = testSkinCancer['SF2']

imbalanceLungCancer = round(np.mean(y_testLungCancer), 3)
imbalanceLiverCancer = round(np.mean(y_testLiverCancer), 3)
imbalanceHandNCancer = round(np.mean(y_testHandNCancer), 3)
imbalanceOvarianCancer = round(np.mean(y_testOvarianCancer), 3)
imbalanceSkinCancer = round(np.mean(y_testSkinCancer), 3)

X_testLungCancer = testLungCancer.drop('SF2', axis=1)
X_testLiverCancer = testLiverCancer.drop('SF2', axis=1)
X_testHandNCancer = testHandNCancer.drop('SF2', axis=1)
X_testOvarianCancer = testOvarianCancer.drop('SF2', axis=1)
X_testSkinCancer = testSkinCancer.drop('SF2', axis=1)

y_predLungCancer = RF_optimized.best_estimator_.predict(X_testLungCancer)
y_predLiverCancer = RF_optimized.best_estimator_.predict(X_testLiverCancer)
y_predHandNCancer = RF_optimized.best_estimator_.predict(X_testHandNCancer)
y_predOvarianCancer = RF_optimized.best_estimator_.predict(X_testOvarianCancer)
y_predSkinCancer = RF_optimized.best_estimator_.predict(X_testSkinCancer)

balAcLungCancer = functions_project5.evaluationTypes(y_testLungCancer, y_predLungCancer, imbalanceLungCancer, "RandomForest", "LungCancer")
balAcLiverCancer = functions_project5.evaluationTypes(y_testLiverCancer, y_predLiverCancer, imbalanceLiverCancer, "RandomForest", "LiverCancer")
balAcHandNCancer = functions_project5.evaluationTypes(y_testHandNCancer, y_predHandNCancer, imbalanceHandNCancer, "RandomForest", "HandNCancer")
balAcOvarianCancer = functions_project5.evaluationTypes(y_testOvarianCancer, y_predOvarianCancer, imbalanceOvarianCancer, "RandomForest", "OvarianCancer")
balAcSkinCancer = functions_project5.evaluationTypes(y_testSkinCancer, y_predSkinCancer, imbalanceSkinCancer, "RandomForest", "SkinCancer")


# ---------- Plot the most important features from the models ----------
RF_coefs_default = pd.DataFrame(zip(X_train.columns, np.transpose(RF_default.feature_importances_)), columns=['features', 'coef'])
print(RF_coefs_default)
RF_coefs_default['coef'] = RF_coefs_default['coef']/RF_coefs_default['coef'].abs().sum() #normalisation
print(RF_coefs_default)
RF_coefs_default = RF_coefs_default.set_index('features')
RF_coefs_default['abs_coef'] = abs(RF_coefs_default['coef'])
RF_coefs_default.sort_values('abs_coef', inplace=True, ascending= False)


RF_coefs_optimized = pd.DataFrame(zip(X_train.columns, np.transpose(bestModel.feature_importances_)), columns=['features', 'coef'])
RF_coefs_optimized['coef'] = RF_coefs_optimized['coef']/RF_coefs_optimized['coef'].abs().sum() #normalisation
RF_coefs_optimized = RF_coefs_optimized.set_index('features')
RF_coefs_optimized['abs_coef'] = abs(RF_coefs_optimized['coef'])
RF_coefs_optimized.sort_values('abs_coef', inplace=True, ascending= False)

# Visualize the top10 most important features for default and optimized
fig, ax = plt.subplots(1, 2, figsize=(9, 6))
# Default Model
ax[0].bar(np.arange(10), RF_coefs_default['coef'][:10])
ax[0].set_xticks(np.arange(10), RF_coefs_default.index.tolist()[:10], rotation=90)
ax[0].set_title("Default Model", fontsize=14)
ax[0].set_xlabel('Gene', fontsize=12)
ax[0].set_ylabel("Normalized Feature Importance", fontsize=12)
# Optimized Model
ax[1].bar(np.arange(10), RF_coefs_optimized['coef'][:10])
ax[1].set_xticks(np.arange(10), RF_coefs_optimized.index.tolist()[:10], rotation=90)
ax[1].set_title("Optimized Model", fontsize=14)
ax[1].set_xlabel('Gene', fontsize=12)
ax[1].set_ylabel("Normalized Feature Importance", fontsize=12)
fig.suptitle('Top 10 most important genes - random forest', fontsize=16)
# Set the same y-axis limits for both subplots
min_value = min(np.min(RF_coefs_default['coef'][:10]), np.min(RF_coefs_optimized['coef'][:10]))
max_value = max(np.max(RF_coefs_default['coef'][:10]), np.max(RF_coefs_optimized['coef'][:10]))
ax[0].set_ylim(min_value - 0.001, max_value + 0.001)
ax[1].set_ylim(min_value - 0.001, max_value + 0.001) #for appearance reasons
# Adjust the spacing between subplots
plt.tight_layout()
plt.savefig(f"../plots/RandomForestImportance.png")
