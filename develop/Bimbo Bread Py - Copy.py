
# coding: utf-8

# <img src='../img/bimbo.png'>

# - Task: Demand = Sales - Returns
# - Create holdout test set

# ## Data exploration

# In[1]:

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cross_validation import train_test_split
#import xgboost as xgb
import operator

dir_sub = "C:/Users/568377/Cognizant/Data Science/kaggle/Bimbo Bread/submissions/"
dir_raw = "C:/Users/568377/Cognizant/Data Science/kaggle/Bimbo Bread/raw/"

