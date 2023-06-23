#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:

import os
import sys
import pickle
import pandas as pd



with open('lin_reg.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[9]:

year = int(sys.argv[1]) # 2021
month = int(sys.argv[2]) # 3

input_file = f"C:/Users/91981/mlops-zoomcamp/data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
df = read_data(input_file)


# In[10]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)




import numpy as np
print(np.mean(y_pred))


# In[17]:



