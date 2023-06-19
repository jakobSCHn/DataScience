# Wichtige Punkte für KNN

# Genexpressionen skalieren (normalisieren oder standardisieren)
# Hyperparameter K wählen
# Check if data is balanced, Zb ob viel mehr 1 als 0 (using class weights can help address this issue)
# Auswahl einer geeigneten Distanzmetrik, Euclidean oder Manhattan oder Korrelationskoeffizienten
# precision, recall, F1-score, or area under the ROC curve (AUC-ROC))
# extras: noch eine feature selection to reduce dimensionality

import time
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline

import functions_project5

# ------------- Load Data -----------------
#load datasets
model_data_train = pd.read_csv("..\Data\ModelData_train.csv", index_col=0)
model_data_test =  pd.read_csv("..\Data\ModelData_test.csv", index_col=0)
# Divide Model Data into X and Y
# double brackets would give back a dataframe, one bracket only gives back a series
y_train = model_data_train['SF2']

print(y_train.head(5))
X_train = model_data_train.drop('SF2', axis= 1)
print(X_train.head(5))

y_test = model_data_test['SF2']
X_test = model_data_test.drop('SF2', axis= 1)

# ------------- Choose Hyperparameter -----------------

# Choose K
# first approach: square root of samples
# k= np.sqrt(len(model_data))


# Assuming you have your features stored in 'X' and the corresponding labels in 'y'

# Define the range of K values to evaluate
#k_values = [3, 5, 7, 9, 11, 15, 19, 21, 23, 25]




#check time for parameter research
t0=time.time()
# Define the resampling strategy
resampler = RandomOverSampler(random_state=0)

# Create the KNN classifier
knn_default = KNeighborsClassifier()

# Create the pipeline with resampling and KNN classifier
pipeline = Pipeline([('resampler', resampler), ('knn', knn_default)])

# Define the parameter grid to search over
param_grid = {
    'resampler__sampling_strategy': ['auto', 'minority'],
    'knn__n_neighbors': [3, 5, 7, 9, 11, 15, 19, 21, 23, 25],
    'knn__weights': ['uniform', 'distance'],
    'knn__metric': ['euclidean', 'manhattan', 'chebyshev']
}

# Create the GridSearchCV object
knn_optimized = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc')

# Fit knn with optimized parameters
knn_optimized.fit(X_train, y_train)

# Get the best parameters and best score
best_params = knn_optimized.best_params_
best_score = knn_optimized.best_score_

# Print the best parameters and best score
print("Best parameters:", best_params)
print("Best score:", best_score)

# Fit with default KNN
knn_default.fit(X_train,y_train)
t1=time.time()
print(f'Time for best Parameter search: {round(t1-t0, 3)}s')

# ---------- Model evaluation ----------
t0=time.time()
#make predictions
y_predDefault = knn_default.predict(X_test)
#print('done with default')
y_predOpt = knn_optimized.best_estimator_.predict(X_test)
#print('done with optimized')
#predict probablities for ROC curve
y_preProbaDefault = knn_default.predict_proba(X_test)[:, 1]
#print(f'default Probabibility:{y_preProbaDefault}')
y_preProbaOpt = knn_optimized.best_estimator_.predict_proba(X_test)[:, 1]
#print(f'Optimized Probabibility:{y_preProbaOpt}')
#print('done with prob for roc curve')
t1=time.time()
print(f'Time for fitting: {round(t1-t0, 3)}s')



t0=time.time()
metricsDef, metricsOpt = functions_project5.evaluation(y_test, y_predDefault, y_preProbaDefault, y_predOpt, y_preProbaOpt, "KNearestNeighbours")

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

y_predLungCancer = knn_optimized.best_estimator_.predict(X_testLungCancer)
y_predLiverCancer = knn_optimized.best_estimator_.predict(X_testLiverCancer)
y_predHandNCancer = knn_optimized.best_estimator_.predict(X_testHandNCancer)
y_predOvarianCancer = knn_optimized.best_estimator_.predict(X_testOvarianCancer)
y_predSkinCancer = knn_optimized.best_estimator_.predict(X_testSkinCancer)

balAcLungCancer = functions_project5.evaluationTypes(y_testLungCancer, y_predLungCancer, imbalanceLungCancer, "KNearestNeighbors", "LungCancer")
balAcLiverCancer = functions_project5.evaluationTypes(y_testLiverCancer, y_predLiverCancer, imbalanceLiverCancer, "KNearestNeighbors", "LiverCancer")
balAcHandNCancer = functions_project5.evaluationTypes(y_testHandNCancer, y_predHandNCancer, imbalanceHandNCancer, "KNearestNeighbors", "HandNCancer")
balAcOvarianCancer = functions_project5.evaluationTypes(y_testOvarianCancer, y_predOvarianCancer, imbalanceOvarianCancer, "KNearestNeighbors", "OvarianCancer")
balAcSkinCancer = functions_project5.evaluationTypes(y_testSkinCancer, y_predSkinCancer, imbalanceSkinCancer, "KNearestNeighbors", "SkinCancer")


print(f'Metrics for Default: {metricsDef}')
print(f'Metrics for Optimized: {metricsOpt}')
t1=time.time()
print(f'Time for Model Evaluation: {round(t1-t0, 3)}s')

