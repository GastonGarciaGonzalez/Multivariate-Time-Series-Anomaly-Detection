from zipfile import ZipFile
import pandas as pd

#pd.set_option("display.max_rows", None, "display.max_columns", None)

# %% xlsx to csv.
# df_train = pd.read_excel('SWaT/Physical/SWaT_Dataset_Normal_v1.xlsx'
#                           , header=1, engine='openpyxl'
#                           )
# df_train.to_csv (r'SWaT/Physical/SWaT_Dataset_Normal_v1.csv'
#                  , index = None, header=True
#                  )

# df_test = pd.read_excel('SWaT/Physical/SWaT_Dataset_Attack_v0.xlsx'
#                           , header=1, engine='openpyxl'
#                           )
# df_test.to_csv (r'SWaT/Physical/SWaT_Dataset_Attack_v0.csv'
#                 , index = None, header=True
#                 )
# %%
zip_file = ZipFile('SWaT.zip')
df_train = pd.read_csv(zip_file.open('SWaT/Physical/SWaT_Dataset_Normal_v1.csv'))
df_test = pd.read_csv(zip_file.open('SWaT/Physical/SWaT_Dataset_Attack_v0.csv'))

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
