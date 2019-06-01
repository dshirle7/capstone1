# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 15:49:09 2019

@author: d
"""

print("Running 'wrangle.py'...")

import numpy as np
import pandas as pd

np.random.seed(0)

print('Beginning wrangling of training and test set')

# Loading data
df_train = pd.read_csv('../../data/raw/train_users_2.csv')
df_test = pd.read_csv('../../data/raw/test_users.csv')
target = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
trainsize = df_train.shape[0]

# Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# Removing date first booking because it does not appear in the test data
df_all = df_all.drop(['date_first_booking'], axis=1)
# Filling in NaN
df_all = df_all.fillna(-1)

### Feature Engineering ###
# date_account_created: three features, one day, one month, one year
dac = pd.to_datetime(df_all['date_account_created'])
df_all['dac_year'] = [entry.year for entry in dac]
df_all['dac_month'] = [entry.month for entry in dac]
df_all['dac_day'] = [entry.day for entry in dac]
df_all['dac_weekday'] = [entry.dayofweek for entry in dac]
df_all = df_all.drop(['date_account_created'], axis=1)

# timestamp_first_active, same ...
tfa = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
df_all['tfa_year'] = [entry.year for entry in tfa]
df_all['tfa_month'] = [entry.month for entry in tfa]
df_all['tfa_day'] = [entry.day for entry in tfa]
df_all['tfa_weekday'] = [entry.dayofweek for entry in tfa]
df_all = df_all.drop(['timestamp_first_active'], axis=1)

# Age
ages = df_all.age.values
# Convert birth years to ages:
ages = np.where(np.logical_and(ages < 2000, ages > 1900), 2014-ages, ages)
# Create a dummy number for all ages below 18:
ages = np.where(np.logical_and(ages < 18, ages > 0), 4, ages)
# Create a dummy number for all entries with current year instead of age:
ages = np.where(np.logical_and(ages < 2016, ages > 2010), 9, ages)
# Create a dummy number for all entries allegedly older than 100:
ages = np.where(ages > 99, 110, ages)
# Store to age
df_all['age'] = ages

# One Hot Encoding
dummiescols = ['gender', 'signup_method', 'signup_flow', 'language',
               'affiliate_channel', 'affiliate_provider',
               'first_affiliate_tracked', 'signup_app', 'first_device_type',
               'first_browser']
df_all = pd.get_dummies(df_all, prefix=dummiescols, columns=dummiescols)

print('Wrangling of training and test set complete')
print('Beginning wrangling of sessions data')

df_sess = pd.read_csv('../../data/raw/sessions.csv')
df_sess['id'] = df_sess['user_id']
df_sess = df_sess.drop(['user_id'], axis=1)

# Fill nan with 'NAN' in all columns except id and seconds
fillcols = ['action', 'action_type', 'action_detail', 'device_type']
df_sess[fillcols] = df_sess[fillcols].fillna('NAN')

grouped = df_sess.groupby(['id'])
# Initialize an empty list to append each id's feature sets
id_features = []
count = 0
last = len(grouped)
for key, table in grouped:
    if count % 10000 == 0:
        print(f"Processing {count} of {last}")
    # Initialize an empty list to append features
    list = []
    # Append the user id to the list
    list.append(key)
    # Append the number of actions taken
    list.append(len(table))
    # Append the number of unique actions taken
    list.append(table['action'].nunique())
    # Append the number of unique action details taken
    list.append(table['action_detail'].nunique())
    # Append the number of unique device types used
    list.append(table['device_type'].nunique())
    # Append the sum of seconds elapsed
    list.append(np.sum(table['secs_elapsed']))
    # Append the mean number of seconds elapsed
    list.append(np.mean(table['secs_elapsed']))
    # Append the standard deviation of seconds elapsed
    list.append(np.std(table['secs_elapsed']))
    
    # Append this data as a row of id_features.
    id_features.append(list)
    count += 1
   
# Creating the aggregate sessions table
new_features = ['id', 'action_count', 'unique_action_count',
                'unique_detail_count', 'device_type_count',
                'sum_secs', 'mean_secs', 'stdev_secs']

id_features = np.array(id_features)

df_sess_agg = pd.DataFrame(id_features, columns=new_features)
df_sess_agg = df_sess_agg.replace([np.inf, -np.inf], np.nan)
df_sess_agg = df_sess_agg.fillna(-2)

print('Wrangling of sessions data complete')
print('Wrapping up...')

df_all = pd.merge(df_all, df_sess_agg, on='id', how='left')
df_all = df_all.fillna(-2) # Summary statistics for no session data

df_train = df_all[:trainsize]
df_train['country_destination'] = target
df_train = df_train.replace([np.inf, -np.inf], np.nan)
df_train = df_train.fillna(-2)
df_test = df_all[trainsize:]
df_test = df_test.replace([np.inf, -np.inf], np.nan)
df_test = df_test.fillna(-2)

df_train.to_csv('../../data/processed/train_users_2.csv', index=False)
df_test.to_csv('../../data/processed/test_users.csv', index=False)