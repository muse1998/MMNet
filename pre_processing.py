import csv
import os
import sys
import numpy as np
import pandas as pd
path='/home/drink36/Desktop/apex_frame_take/casme3_result_2'

df=pd.read_csv('result_2.csv')
# delete element which emotion equals to 'anger','surprise'
class_list=['happy','fear','sad','disgust','anger','surprise','others']
# show emotion number
for i in range(7):
    print(df[df['emotion']==class_list[i]].shape[0])
# delete anger and surprise


df=df.drop(df[(df['emotion']=='anger')].index)
df=df.drop(df[(df['emotion']=='sad')].index)
print(df)
print(len(df))
df.to_csv('result_3.csv',index=False)
# if subject list is not in list1, then delete it
# sorted only by  number