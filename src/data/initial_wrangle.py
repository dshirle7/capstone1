# -*- coding: utf-8 -*-

'''

This code generates the interim data sets needed to run
the first Exploratory EDA notebook I created. Neither of
these data frames will be used by the Machine Learning
models, because the first EDA taught me some assumptions
I made here were wrong.

This is kept for archival purposes only.

'''

import os
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))

file = os.path.realpath(script_dir + '/../../data/raw/train_users_2.csv')

file2 = os.path.realpath(script_dir + '/../../data/raw/sessions.csv')

df = pd.read_csv(file)

df = df[df.gender != '-unknown-']

df = df[df.first_browser != '-unknown-']

df = df[df.country_destination != '-unknown-']

df = df[df.age < 100]

df.dropna()

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

df.to_csv(os.path.realpath(script_dir + '/../../data/interim/train_users_2.csv'))

df2.to_csv(os.path.realpath(script_dir + '/../../data/interim/sessions.csv'))