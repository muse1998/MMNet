import pandas as pd
import os
import numpy as np
import csv
path='result_file.csv'
# read ['TP_C'] and ['TP'] as float list
df=pd.read_csv(path)
all=df.iloc[89]
all=all.tolist()
#let all['acc'] be float
for i in range(3,5):
    all[i]=all[i].replace('[','')
    all[i]=all[i].replace(']','')
    all[i]=all[i].replace(' ','')
    all[i]=all[i].split(',')
    all[i]=[float(k) for k in all[i]]
all[3]=np.array(all[3])
all[4]=np.array(all[4]) 
dataframe=pd.DataFrame(columns = ['model_name','happy','fear','sad','disgust','anger','surprise','others','all_acc'])
# add model name to all[4]/all[3]'s result
temp=np.append(['ALL'],all[4]/all[3])
temp=np.append(temp,all[2])
dataframe.loc[(0,)]=temp
print(dataframe)
dataframe.to_csv('result_file_analyze.csv',index=False)
# for i in range(89):
#     all=df.iloc[i].tolist()
#     for j in range(3,5):
#         all[j]=all[j].replace('[','')
#         all[j]=all[j].replace(']','')
#         all[j]=all[j].replace(' ','')
#         all[j]=all[j].split(',')
#         all[j]=[float(k) for k in all[j]]
#     all[3]=np.array(all[3])
#     all[4]=np.array(all[4]) 
#     temp=np.append(all[0],all[4]/all[3])
#     temp=np.append(temp,all[2])
#     dataframe.loc[(i+1,)]=temp
# # set nan to 0 to every cell
# for i in range(1,6):
#     dataframe.iloc[:,i]=dataframe.iloc[:,i].fillna(0)
#     print(dataframe.iloc[:,i])
# print(dataframe)
# save to csv
# dataframe.to_csv('result_file_analyze.csv',index=False)

