# -*- coding: utf-8 -*-

import os
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))

file = os.path.realpath(script_dir + '/../../data/raw/train_users_2.csv')

file2 = os.path.realpath(script_dir + '/../../data/raw/sessions.csv')

df = pd.read_csv(file)

df = df[df.first_browser != '-unknown-']

df = df[df.country_destination != '-unknown-']

df = df[df.age < 100]

df.dropna()

# This code converts all our time-based variables to datetime objects,
# and it creates a derivative field I used during my first EDA.

df['timestamp_first_active'] = pd.to_datetime(df['timestamp_first_active'], format='%Y%m%d%H%M%S')
df['date_first_booking'] = pd.to_datetime(df['date_first_booking'])
df['days_thinking'] = df['date_first_booking'] - df['timestamp_first_active']
df['days_thinking'] = df['days_thinking'].astype('timedelta64[D]')

df2 = pd.read_csv(file2)

# I want to go through both of these data frames and just delete
# any IDs that do not appear in both the user data and sessions data.
# Then we can see how many we have left.

df2 = df2[df2['user_id'].isin(df.id.unique())]

df = df[df['id'].isin(df2.user_id.unique())]

# This check demonstrates that there are no IDs present in either
# data frame that are absent in the other. I.e. the symmetric difference
# is empty. I also checked length of A to see how many users remained.

A = set(df.id.unique())
B = set(df2.user_id.unique())

print(A ^ B)
print(len(A))
print(len(B))

# This code derives a "number of actions" feature from the sessions
# DataFrame and attaches it to the first DataFrame. It also lays out
# the code for creating more features of this kind in future wrangles.

action_count = df2.groupby('user_id', as_index=False).count()

# Looks like for this table, some actions and some action types are blank.
# The most reliable column is device_type, which always has a value. So
# we can use its count.

action_count = action_count[['user_id', 'device_type']]
action_count.columns = ['id', 'number_of_actions']

df = pd.merge(df, action_count, on='id')

df.to_csv(os.path.realpath(script_dir + '/../../data/interim/train_users_2_2.csv'))

df2.to_csv(os.path.realpath(script_dir + '/../../data/interim/sessions_2.csv'))