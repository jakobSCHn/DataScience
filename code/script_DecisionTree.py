import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn import tree

import functions_project5


#new datasets
model_data_train = pd.read_csv("..\data\ModelData_train.csv", index_col=0)
model_data_test =  pd.read_csv("..\data\ModelData_test.csv", index_col=0)
# Divide Model Data into X and Y
# double brackets would give back a dataframe, one bracket only gives back a series
y_train = model_data_train['SF2']

print(y_train.head(5))
X_train = model_data_train.drop('SF2', axis= 1)
print(X_train.head(5))

y_test = model_data_test['SF2']
X_test = model_data_test.drop('SF2', axis= 1)



#defining possible hyperparameters
criterion  = ['gini', 'entropy']
max_depth  = list(range(1,101))
max_features = ['sqrt', 'log2', None]
parameters = dict(criterion=criterion,
                  max_depth=max_depth,
                  max_features=max_features)

DT_default = tree.DecisionTreeClassifier(class_weight="balanced", random_state=0) #ensure reproducability
DT_optimized = GridSearchCV(DT_default, parameters, cv=5, scoring='roc_auc')
DT_default.fit(X_train, y_train)
DT_optimized.fit(X_train, y_train)
print('Default Parameters:')
print('Criterion:', DT_default.get_params()['criterion'])
print('Max depth:', DT_default.get_params()['max_depth'])
print('Max features:', DT_default.get_params()['max_features'])
print('Best Parameters:')
print('Best Criterion:', DT_optimized.best_estimator_.get_params()['criterion'])
print('Best max depth:', DT_optimized.best_estimator_.get_params()['max_depth'])
print('Best max features:', DT_optimized.best_estimator_.get_params()['max_features'])


# Make predictions
y_pred_default = DT_default.predict(X_test)
y_pred_opt = DT_optimized.best_estimator_.predict(X_test)


#predict probablities for ROC curve
y_preProbaDefault = DT_default.predict_proba(X_test)[:, 1]
y_preProbaOpt = DT_optimized.best_estimator_.predict_proba(X_test)[:, 1]

metricsDef, metricsOpt = functions_project5.evaluation(y_test, y_pred_default, y_preProbaDefault, y_pred_opt, y_preProbaOpt, "Decision Tree")

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

y_predLungCancer = DT_optimized.best_estimator_.predict(X_testLungCancer)
y_predLiverCancer = DT_optimized.best_estimator_.predict(X_testLiverCancer)
y_predHandNCancer = DT_optimized.best_estimator_.predict(X_testHandNCancer)
y_predOvarianCancer = DT_optimized.best_estimator_.predict(X_testOvarianCancer)
y_predSkinCancer = DT_optimized.best_estimator_.predict(X_testSkinCancer)

balAcLungCancer = functions_project5.evaluationTypes(y_testLungCancer, y_predLungCancer, imbalanceLungCancer, "DecisionTree", "LungCancer")
balAcLiverCancer = functions_project5.evaluationTypes(y_testLiverCancer, y_predLiverCancer, imbalanceLiverCancer, "DecisionTree", "LiverCancer")
balAcHandNCancer = functions_project5.evaluationTypes(y_testHandNCancer, y_predHandNCancer, imbalanceHandNCancer, "DecisionTree", "HandNCancer")
balAcOvarianCancer = functions_project5.evaluationTypes(y_testOvarianCancer, y_predOvarianCancer, imbalanceOvarianCancer, "DecisionTree", "OvarianCancer")
balAcSkinCancer = functions_project5.evaluationTypes(y_testSkinCancer, y_predSkinCancer, imbalanceSkinCancer, "DecisionTree", "SkinCancer")


#plots
fig = plt.figure(figsize=(50, 50))
_ = tree.plot_tree(DT_default,
                   feature_names=X_train.columns,
                   class_names=[str(c) for c in y_train.values],
                   filled=True)
plt.text(0.5, 1.05, "Decision Tree Default", ha='center', va='center', fontsize=100, transform=plt.gca().transAxes)
plt.savefig(f"../plots/DecisionTreeDefault.png")

fig = plt.figure(figsize=(50, 50))
_ = tree.plot_tree(DT_optimized.best_estimator_,
                   feature_names=X_train.columns,
                   class_names=[str(c) for c in y_train.values],
                   filled=True)
plt.text(0.5, 1.05, "Decision Tree Optimized", ha='center', va='center', fontsize=100, transform=plt.gca().transAxes)
plt.savefig(f"../plots/DecisionTreeOptimized.png")
