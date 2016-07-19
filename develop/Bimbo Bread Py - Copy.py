
# coding: utf-8

# <img src='../img/bimbo.png'>

# - Task: Demand = Sales - Returns
# - Create holdout test set

# ## Data exploration

# In[1]:

#jupyter nbconvert --to python "Bimbo Bread Py.ipynb"

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cross_validation import train_test_split
import xgboost as xgb
import operator

dir_sub = "C:/Users/568377/Cognizant/Data Science/kaggle/Bimbo Bread/submissions/"
dir_raw = "C:/Users/568377/Cognizant/Data Science/kaggle/Bimbo Bread/raw/"


# In[2]:

subm = pd.read_csv(os.path.join(dir_raw,"sample_submission.csv"))
prod = pd.read_csv(os.path.join(dir_raw,"producto_tabla.csv"))
town_state = pd.read_csv(os.path.join(dir_raw,"town_state.csv"))
client = pd.read_csv(os.path.join(dir_raw,"cliente_tabla.csv"))


# In[ ]:

print("Reading test data...")
test = pd.read_csv(os.path.join(dir_raw,"test.csv"))
print("Done")


# In[ ]:

print("Reading train data...")
train_1 = pd.read_csv(os.path.join(dir_raw,"train/xaa"))
print("first batch finished")
train_2 = pd.read_csv(os.path.join(dir_raw,"train/xab"))
print("second batch finished")
print("Done")


# In[ ]:

train = pd.concat([train_1,train_2])

train.head()
