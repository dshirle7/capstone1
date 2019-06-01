# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 16:49:23 2019

@author: d
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier

np.random.seed(0)

# Loading data
print("Preparing the datasets...")
df_train = pd.read_csv('../../data/processed/train_users_2.csv')
df_test = pd.read_csv('../../data/processed/test_users.csv')
target = df_train['country_destination'].values
df_train = df_train.drop(['id', 'country_destination'], axis=1)
id_test = df_test['id']
df_test = df_test.drop(['id'], axis=1)

# Fill NaN; should already be filled but just to be sure
# XGBoost already handles missing values too
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.fillna(-2)
df_test = df_test.replace([np.inf, -np.inf], np.nan)
df_test = df_test.fillna(-2)

# Split train and test
X_train = df_train.to_numpy()
le = LabelEncoder()
y_train = le.fit_transform(target)
X_test = df_test.to_numpy()

# Create the Classifier model and the Hyperparameter Tuner
# We want multi:softprob objective to give us a table of the probabilities
# Then we can ake the top five for our submission
xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
                    objective='multi:softprob', subsample=0.5,
                    colsample_by_tree=0.5, seed=0)

# Select the parameters to tune
# Note by this grid, we are making 21 different models
param_grid = {'n_estimators': [25, 50, 100],
              'learning_rate': [0.3, 0.2, 0.1],
              'max_depth': [5, 6, 7]}

# With 3-fold CV, we will now be computing effectively 81 models
clf = GridSearchCV(xgb, param_grid, cv=3)
print("Running the parameter search...")

clf.fit(X_train, y_train)
print("Best parameters:")
print(clf.get_params())
y_pred = clf.predict_proba(X_test)

# Keep the 5 classes with the highest probabilities
ids = [] # List of ids
countries = [] # List of countries
for i in range(len(id_test)):
    this_id = id_test[i]
    ids += [this_id] * 5
    countries += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
    
# Generate submission
submission = pd.DataFrame(np.column_stack((ids, countries)),
                          columns=['id', 'country'])
submission.to_csv('submission.csv', index=False)