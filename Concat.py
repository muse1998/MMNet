import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
path = "temp.csv"
path2 = "SMIC_all_cropped\concat.csv"
df = pd.read_csv(path)
df2 = pd.read_csv(path2)
# change the column name of df
df.columns = ['Subject', 'File_name',
              'Apex_predicted', 'ons', 'offs', 'Emotion']
# append new column TYPE with value casme to df
df['TYPE'] = 'casme'
# Let TYPE become column 0
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]
df_new = pd.concat([df, df2], axis=0)
print(df_new)
df_new.to_csv("temp2.csv", index=False)
