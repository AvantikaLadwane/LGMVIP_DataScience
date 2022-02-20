#!/usr/bin/env python
# coding: utf-8

# # LetsGrowMore Virtual Internship Program ( Feb 2022 - Mar 2021 )

# # BY : Avantika Ladwane

# Task 2 : Stock Market Prediction And Forecasting Using Stacked LSTM

# Level : Beginner

# # Importing Libraries

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# # Importing Data Set

# In[2]:


ds = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
ds


# # Describing the Dataset

# In[3]:


ds.head()


# In[4]:


ds.tail()


# In[5]:


ds.dtypes


# In[6]:


ds['Date'].value_counts()


# In[7]:


ds['High'].hist()


# In[8]:


plt.figure(figsize=(20,8))
ds.plot()


# In[9]:


data_set = ds.filter(['Close'])
Dataset = ds.values
training_data_len=math.ceil(len(ds) * 8)
training_data_len


# In[10]:


Dataset


# In[11]:


ds = ds.iloc[:, 0:5]
ds


# In[12]:


training_set = ds.iloc[:, 1:2].values
training_set


# # Scalling of Data Set

# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
data_training_scaled = scaler.fit_transform(training_set)


# In[14]:


features_set = []
labels = []
for i in range(60, 586):
  features_set.append(data_training_scaled[i - 60:i, 0])
  labels.append(data_training_scaled[i, 0])


# In[15]:


features_set, labels = np.array(features_set), np.array(labels)


# In[16]:


features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
features_set.shape


# # Building The LSTM

# In[17]:


import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM


# In[18]:


model = Sequential()


# In[19]:


model.compile(optimizer='adam', loss='mean_squared_error')


# In[20]:


model.fit(features_set, labels, epochs=100, batch_size=20)


# In[21]:


url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
data_test= pd.read_csv(url)
data_processed = data_test.iloc[:, 1:2]
data_processed


# # Prediction of the Data

# In[22]:


data_total = pd.concat((ds['Open'], ds['Open']), axis=0)


# In[23]:


test_inputs = data_total[len(data_total) - len(ds) - 60:].values
test_inputs.shape


# In[24]:


test_inputs = test_inputs.reshape(-1, 1)
test_inputs = scaler.transform(test_inputs)


# In[25]:


test_feature = []
for i in range(60, 80):
  test_feature.append(test_inputs[i-60:i, 0])


# In[26]:


test_feature = np.array(test_feature)
test_feature = np.reshape(test_feature, (test_feature.shape[0] - test_feature.shape[1], 1))
test_feature.shape


# In[27]:


predictions = model.predict(test_feature)


# In[28]:


predictions


# In[29]:


x_train = ds[0:1256]
y_train = ds[1:1257]
print(x_train.shape)
print(y_train.shape)


# In[30]:


x_train


# In[31]:


np.random.seed(1)
np.random.randn(3, 3)


# # Drawing a Single number from the Normal Distribution

# In[32]:


np.random.normal(1)


# # Drawing 5 numbers from Normal Distribution

# In[33]:


np.random.normal(5)


# In[34]:


np.random.seed(42)


# In[35]:


np.random.normal(size=1000, scale=100).std()


# # Ploting Results

# In[36]:


plt.figure(figsize=(18,6))
plt.title("Stock Market Price Prediction")
plt.plot(data_test['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()


# # Analyze the Closing price from the dataframe

# In[37]:


ds["Date"] = pd.to_datetime(ds.Date)
ds.index = ds['Date']

plt.figure(figsize=(20, 10))
plt.plot(ds["Open"], label='ClosePriceHist')


# In[38]:


plt.figure(figsize=(12,6))
plt.plot(ds['Date'])
plt.xlabel('Turnover (Lacs)', fontsize=18)
plt.ylabel('Total Trade Quantity', fontsize=18)
plt.show()


# # Analyze the Closing price from the dataframe

# In[39]:


ds["Turnover (Lacs)"] = pd.to_datetime(ds.Date)
ds.index = ds['Turnover (Lacs)']

plt.figure(figsize=(20, 10))
plt.plot(ds["Turnover (Lacs)"], label='ClosePriceHist')


# In[40]:


sns.set(rc = {'figure.figsize': (20, 5)})
ds['Open'].plot(linewidth = 1,color='blue')


# In[41]:


ds.columns


# In[42]:


df = pd.read_csv('https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv')
df


# In[43]:


cols_plot = ['Open','High','Low','Last','Close']
axes = df[cols_plot].plot(alpha = 1, figsize=(20, 30), subplots = True)

for ax in axes:
    ax.set_ylabel('Variation')

