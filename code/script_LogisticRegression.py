import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import functions_project5


#------------------read the data------------------
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

# Define parameters for optimization
C_values = np.logspace(-4,4,50)
solver_options = ['lbfgs', 'newton-cg']
max_iterations = [100]
parameters = [
    {
        'penalty': [None],
        'solver': solver_options,
        'max_iter': max_iterations

    },
    {
        'penalty': ['l2'],
        'solver': solver_options,
        'max_iter': max_iterations,
        'C': C_values
    }
]
#needs to be done this way, do to differe between the penalty options (since penalty options None has no C-value)


# Create a logistic regression model
LR_default = LogisticRegression(class_weight= "balanced", random_state=0)
LR_optimized = GridSearchCV(LR_default, parameters, cv=5, scoring='roc_auc')
LR_default.fit(X_train, y_train)
LR_optimized.fit(X_train, y_train)
# Print default and best parameters
print('Default Parameters:')
print('C values:', LR_default.get_params()['C'])
print('Penalty:', LR_default.get_params()['penalty'])
print('Solver options:', LR_default.get_params()['solver'])
print('Maximal iterations:', LR_default.get_params()['max_iter'])
print('Best Parameters:')
print('C:', LR_optimized.best_estimator_.get_params()['C'])
print('Penalty:', LR_optimized.best_estimator_.get_params()['penalty'])
print('Solver options:', LR_optimized.best_estimator_.get_params()['solver'])
print('Maximal iterations:', LR_optimized.best_estimator_.get_params()['max_iter'])


# Make predictions
y_pred_default = LR_default.predict(X_test)
y_pred_opt = LR_optimized.best_estimator_.predict(X_test)

# Predict probabilities for ROC curve
y_preProbaDefault = LR_default.predict_proba(X_test)[:, 1]
y_preProbaOpt = LR_optimized.best_estimator_.predict_proba(X_test)[:, 1]

# Evaluate the classifiers
metricsDef, metricsOpt = functions_project5.evaluation(y_test, y_pred_default, y_preProbaDefault, y_pred_opt, y_preProbaOpt, "LogisticRegression")

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

y_predLungCancer = LR_optimized.best_estimator_.predict(X_testLungCancer)
y_predLiverCancer = LR_optimized.best_estimator_.predict(X_testLiverCancer)
y_predHandNCancer = LR_optimized.best_estimator_.predict(X_testHandNCancer)
y_predOvarianCancer = LR_optimized.best_estimator_.predict(X_testOvarianCancer)
y_predSkinCancer = LR_optimized.best_estimator_.predict(X_testSkinCancer)

balAcLungCancer = functions_project5.evaluationTypes(y_testLungCancer, y_predLungCancer, imbalanceLungCancer, "LogisticRegression", "LungCancer")
balAcLiverCancer = functions_project5.evaluationTypes(y_testLiverCancer, y_predLiverCancer, imbalanceLiverCancer, "LogisticRegression", "LiverCancer")
balAcHandNCancer = functions_project5.evaluationTypes(y_testHandNCancer, y_predHandNCancer, imbalanceHandNCancer, "LogisticRegression", "HandNCancer")
balAcOvarianCancer = functions_project5.evaluationTypes(y_testOvarianCancer, y_predOvarianCancer, imbalanceOvarianCancer, "LogisticRegression", "OvarianCancer")
balAcSkinCancer = functions_project5.evaluationTypes(y_testSkinCancer, y_predSkinCancer, imbalanceSkinCancer, "LogisticRegression", "SkinCancer")



#have a look at the most important features
#get the coefficents and sort them after absolute coefficents
LR_coefs_default = pd.DataFrame(zip(X_train.columns, np.transpose(LR_default.coef_[0])), columns=['features', 'coef'])
print(LR_coefs_default)
LR_coefs_default['coef'] = LR_coefs_default['coef']/LR_coefs_default['coef'].abs().sum() #normalisation
print(LR_coefs_default)
LR_coefs_default = LR_coefs_default.set_index('features')
LR_coefs_default['abs_coef'] = abs(LR_coefs_default['coef'])
LR_coefs_default.sort_values('abs_coef', inplace=True, ascending= False)


LR_coefs_optimized = pd.DataFrame(zip(X_train.columns, np.transpose(LR_optimized.best_estimator_.coef_[0])), columns=['features', 'coef'])
LR_coefs_optimized['coef'] = LR_coefs_optimized['coef']/LR_coefs_optimized['coef'].abs().sum() #normalisation
LR_coefs_optimized = LR_coefs_optimized.set_index('features')
LR_coefs_optimized['abs_coef'] = abs(LR_coefs_optimized['coef'])
LR_coefs_optimized.sort_values('abs_coef', inplace=True, ascending= False)

# Visualize the top10 most important features for default and optimized
fig, ax = plt.subplots(1, 2, figsize=(9, 6))
# Default Model
ax[0].bar(np.arange(10), LR_coefs_default['coef'][:10])
ax[0].set_xticks(np.arange(10), LR_coefs_default.index.tolist()[:10], rotation=90)
ax[0].set_title("Default Model", fontsize=14)
ax[0].set_xlabel('Gene', fontsize=12)
ax[0].set_ylabel("Normalized Feature Importance", fontsize=12)
# Optimized Model
ax[1].bar(np.arange(10), LR_coefs_optimized['coef'][:10])
ax[1].set_xticks(np.arange(10), LR_coefs_optimized.index.tolist()[:10], rotation=90)
ax[1].set_title("Optimized Model", fontsize=14)
ax[1].set_xlabel('Gene', fontsize=12)
ax[1].set_ylabel("Normalized Feature Importance", fontsize=12)
fig.suptitle('Top 10 most important genes - logistic regression', fontsize=16)
# Set the same y-axis limits for both subplots
min_value = min(np.min(LR_coefs_default['coef'][:10]), np.min(LR_coefs_optimized['coef'][:10]))
max_value = max(np.max(LR_coefs_default['coef'][:10]), np.max(LR_coefs_optimized['coef'][:10]))
ax[0].set_ylim(min_value - 0.001, max_value + 0.001)
ax[1].set_ylim(min_value - 0.001, max_value + 0.001) #for appearance reasons
# Adjust the spacing between subplots
plt.tight_layout()
plt.savefig(f"../plots/LogisticRegressionImportance.png")
