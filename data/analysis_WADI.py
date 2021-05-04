# -*- coding: utf-8 -*-
"""
Created on Sun May  2 21:08:22 2021

@author: gastong@fing.edu.uy
"""

from zipfile import ZipFile
import pandas as pd

#pd.set_option("display.max_rows", None, "display.max_columns", None)

# %%
zip_file = ZipFile('../../data/WADI_A2_19_Nov_2019.zip')
df_train = pd.read_csv(zip_file.open('WADI_A2_19_Nov_2019/WADI_14days_new.csv'))
df_test = pd.read_csv(zip_file.open('WADI_A2_19_Nov_2019/WADI_attackdataLABLE.csv'), header=1)

#Describe
tr_describe = df_train.describe()
te_describe = df_test.describe()

print('Características constantes:')
print('\t train:', tr_describe.columns[tr_describe.loc['std'] == 0])
print()
print('\t test:', te_describe.columns[te_describe.loc['std'] == 0])

# #Timestamp
# timestamp_tr = pd.to_datetime(df_train[' Timestamp'])
# hour = timestamp_tr.dt.hour