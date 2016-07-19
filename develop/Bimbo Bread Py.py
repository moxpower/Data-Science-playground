
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
train = pd.read_csv(os.path.join(dir_raw,"train.csv"))
print("Done")


# ### Features

# #### Product

# In[4]:

# prod.head()


# In[5]:

# extract weight and volume from product name
prod['unit_g'] = prod['NombreProducto'].str.extract("([0-9]+)(g )", expand=True)[1]
prod['amount_g'] = prod['NombreProducto'].str.extract("([0-9]+)(g )", expand=True)[0]
prod['unit_ml'] = prod['NombreProducto'].str.extract("([0-9]+)(ml )", expand=True)[1]
prod['amount_ml'] = prod['NombreProducto'].str.extract("([0-9]+)(ml )", expand=True)[0]

# make numeric
prod['amount_g'] = prod['amount_g'].convert_objects(convert_numeric=True)
prod['amount_ml'] = prod['amount_ml'].convert_objects(convert_numeric=True)


# In[6]:

# prod


# In[7]:

# show distribution of weight / volume
fig, axs = plt.subplots(1,2)
bins_ml=[i for i in range(0,1000,50)]
bins_g=[i for i in range(0,2000,100)]

prod['amount_ml'].hist(ax=axs[0],bins=bins_ml,alpha=0.5)
axs[0].set_xlabel("ml")
prod['amount_g'].hist(ax=axs[1],bins=bins_g,alpha=0.5)
axs[1].set_xlabel("g")


# # NEXT STEPS
# - Categorize food
# - Quota of large items
# - How new is item?
# 
# #### Town/State

# In[8]:

# town_state.head()


# In[9]:

# make lowercase
town_state['Town'] = town_state['Town'].str.lower()
town_state['State'] = town_state['State'].str.lower()
# separate zip, town_name
town_state['zip'] = town_state['Town'].str[:4]
town_state['town_name'] = town_state['Town'].str[5:]


# In[10]:

# town_state.head()


# In[11]:

# check uniques
# len(town_state['Town'].unique()), len(town_state['zip'].unique()), len(town_state['town_name'].unique())
##town_state = town_state.groupby('zip')
##town_state['Town'].str[4].unique() -> clean


# In[12]:

# twice, similar zips have different town_names
df = town_state.groupby('zip').Town.nunique()
df[df==2] #3 entries with different town names
double_town_names = np.array(df[df==2].index)
double_town_names # zips


# In[13]:

town_state['zip'].max() #max zip number


# In[14]:

town_state[town_state['zip'] == double_town_names[0]]


# In[15]:

town_state.loc[199,'zip']=9999
town_state.loc[199,'Town']="9999 cruce de anden noroeste"
town_state[town_state['zip'] == 9999]


# In[16]:

town_state[town_state['zip'] == double_town_names[1]]


# In[17]:

town_state.loc[311,'zip']=9998
town_state.loc[311,'Town']="9998 cruce de andén sureste"
town_state[town_state['zip'] == 9998]


# In[18]:

# town_state.head()


# In[19]:

# len(town_state['Town'].unique()), len(town_state['zip'].unique()), len(town_state['town_name'].unique())


# In[20]:

df1 = town_state.groupby('town_name').zip.nunique()


# In[21]:

df1[df1==2]


# In[22]:

df1[df1==2] #3 entries with different town names
double_zip_names = np.array(df1[df1==2].index)
double_zip_names # zips


# In[23]:

town_state[town_state['town_name'] == double_zip_names[0]] #towns can have multiple zip codes!!! :)


# In[24]:

town_state[town_state['town_name'] == double_zip_names[1]] #ID 387 interferes with town_name. If town_name big effect, look here. Test here for improvement.


# In[25]:

# town_state.head()


# In[26]:

# Create new category city type:
town_state['town_name_corpus'] = town_state['town_name']
town_state['town_name_type'] = ""


# In[27]:

# add town_name_type and add "ag"
town_state.loc[town_state['town_name_corpus'].str[:4] == "ag. ", 'town_name_type'] = "ag"
# remove ag from town_name_corpus
town_state.loc[town_state['town_name_corpus'].str[:4] == "ag. ", 'town_name_corpus'] = town_state.loc[town_state['town_name_corpus'].str[:4] == "ag. ", 'town_name_corpus'].str[4:]

town_state.loc[town_state['town_name_corpus'].str[:3] == "ag.", 'town_name_type'] = "ag"
town_state.loc[town_state['town_name_corpus'].str[:3] == "ag.", 'town_name_corpus'] = town_state.loc[town_state['town_name_corpus'].str[:3] == "ag.", 'town_name_corpus'].str[3:]

town_state.loc[town_state['town_name_corpus'].str[:4] == "cd. ", 'town_name_type'] = "cd"
town_state.loc[town_state['town_name_corpus'].str[:4] == "cd. ", 'town_name_corpus'] = town_state.loc[town_state['town_name_corpus'].str[:4] == "cd. ", 'town_name_corpus'].str[4:]


# In[28]:

#town_state.head()
#sorted(town_state['town_name'].str[:30].unique())


# #### Client

# In[29]:

#client = pd.read_csv(os.path.join(dir_raw,"cliente_tabla.csv"))
client.tail()
#len(client)


# In[30]:

"""client['NombreCliente'] = client['NombreCliente'].str.lower()
client = pd.read_csv(os.path.join(dir_raw,"cliente_tabla.csv"))
client = client.groupby(['Cliente_ID', 'NombreCliente'])
#client = client.drop_duplicates(subset=['Cliente_ID', 'NombreCliente'], keep=False)
client.tail()
len(client)

df1 = client.groupby('Cliente_ID').NombreCliente.nunique()
df1[df1!=1]

testclient = client[client['Cliente_ID'] == 4]
testclient

testclient.loc[4]['NombreCliente'] == testclient.loc[5]['NombreCliente']
"""


# In[31]:

test.head()


# In[32]:

train.head()


# ## Merge product ID, train

# In[33]:

train.tail()


# In[34]:

train = pd.read_csv(os.path.join(dir_raw,"train.csv"), nrows=500000)
test = pd.read_csv(os.path.join(dir_raw,"test.csv"), nrows=500000)


# In[35]:

train = pd.merge(train, prod, on='Producto_ID', how='left')
train = pd.merge(train, town_state, on='Agencia_ID', how='left')
#train = pd.merge(train, client, on='Cliente_ID', how='left') # no 

test = pd.merge(test, prod, on='Producto_ID', how='left')
test = pd.merge(test, town_state, on='Agencia_ID', how='left')
#test = pd.merge(test, client, on='Cliente_ID', how='left') # no 


# In[ ]:




# In[ ]:




# # XGBOOST

# https://www.kaggle.com/cast42/rossmann-store-sales/xgboost-in-python-with-rmspe-v2/code

# In[36]:

features = ['Semana',
 'Agencia_ID',
 'Canal_ID',
 'Ruta_SAK',
 'Cliente_ID',
 'Producto_ID',
 #'Venta_uni_hoy',
 #'Venta_hoy',
 #'Dev_uni_proxima',
 #'Dev_proxima',
 #'Demanda_uni_equil',
 'amount_g',
 'amount_ml']


# In[37]:

#replace nan
train=train.fillna(-999)
test=test.fillna(-999)


# In[38]:

params = {"objective": "reg:linear",
          "booster" : "gbtree",
          "eta": 0.3,
          "max_depth": 10,
          "subsample": 0.9,
          "colsample_bytree": 0.7,
          "silent": 1,
          "seed": 1301
          }
num_boost_round = 300


# In[39]:

print("Train a XGBoost model")
X_train, X_valid = train_test_split(train, test_size=0.012, random_state=10)
y_train = X_train.Demanda_uni_equil
y_valid = X_valid.Demanda_uni_equil
dtrain = xgb.DMatrix(X_train[features], y_train, missing=-999)
dvalid = xgb.DMatrix(X_valid[features], y_valid, missing=-999)


# In[42]:

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,   early_stopping_rounds=100,  verbose_eval=True) # add feval=rmspe_xg for scoring


# In[ ]:

#print("Validating")
#yhat = gbm.predict(xgb.DMatrix(X_valid[features]))
#error = rmspe(X_valid.Sales.values, np.expm1(yhat))
#print('RMSPE: {:.6f}'.format(error))

print("Make predictions on the test set")
dtest = xgb.DMatrix(test[features])
test_probs = gbm.predict(dtest)


# In[65]:

# Make Submission
result = pd.DataFrame({'id': test['id'], 'Demanda_uni_equil': test_probs})
result = result[['id','Demanda_uni_equil']]
result.to_csv(os.path.join(dir_sub, "submission_4.csv"), cols=["id","Demanda_uni_equil"], index=False)


# In[66]:

result.head()


# In[67]:

resultcsv = pd.read_csv(os.path.join(dir_sub,"submission_5.csv"))


# In[68]:

resultcsv.head()


# In[ ]:




# # NEXT STEPS

# In[ ]:

"""# onehotencode categorical features
from sklearn.preprocessing import OneHotEncoder
>>> enc = OneHotEncoder()

# run script in command line
"""


# In[ ]:




# In[ ]:




# http://nbviewer.jupyter.org/github/JohanManders/ROSSMANN-KAGGLE/blob/master/ROSSMANN%20STORE%20SALES%20COMPETITION%20KAGGLE.ipynb

# # Archive

# #### sub2

# In[85]:

train = pd.read_csv(os.path.join(dir_raw,"sample_submission.csv"))


# In[86]:

train["Demanda_uni_equil"]=6


# In[87]:

train.to_csv(os.path.join(dir_sub, "submission_2.csv"), index=False)


# In[ ]:

#jupyter nbconvert --to python "Bimbo Bread Py.ipynb"

