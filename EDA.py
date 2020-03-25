#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as tp
import time
import tqdm
from calendar import monthrange
import calendar
from xgboost import XGBRegressor
from xgboost import plot_importance
import sklearn
from itertools import product
from sklearn.metrics import mean_squared_error
from math import sqrt
#from tqdm import tqdm_notebook as tqdm


# In[2]:


sales_train = pd.read_csv('sales_train.csv')
itemcat = pd.read_csv('items.csv')
print('Size of dataset before outliers: {}'.format(sales_train.shape))
sales_train = sales_train[sales_train['item_price']<100000]
sales_train = sales_train[sales_train['item_cnt_day']<1000]
sales_train = sales_train[sales_train['item_price']>0]
print('Size of dataset after removing outliers: {}'.format(sales_train.shape))
ts = time.time()
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(35):
    #sales = sales_train[sales_train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales_train.shop_id.unique(), itemcat.item_id.unique())), dtype='int16'))

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
time.time() - ts
exp = sales_train.groupby(['date_block_num','item_id','shop_id'])['item_cnt_day'].sum()
exp = pd.DataFrame(exp).reset_index()
exp.rename(columns={'item_cnt_day':'item_cnt_month'}, inplace = True)
#exp.sort_values(by=['date_block_num','item_id','shop_id'], inplace = True)
matrix = pd.merge(matrix, exp, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)#.astype(np.float16))
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))
train = matrix
lags = [1,2,3,6]
train = pd.merge(train,itemcat[['item_id','item_category_id']], on = ['item_id'], how = 'left')
train.rename(columns={'item_category_id':'item_cat'}, inplace= True)
train.info()


# In[3]:


#train = pd.read_csv('train.csv')
#train = train.drop('Unnamed: 0', axis = 1)
#train.sort_values(by=['date_block_num','item_id','shop_id'], inplace = True)
#train.info()
#train.item_cnt_month = train.item_cnt_month.clip(0,20)
#lags = [1,2,3,6]


# In[3]:


# prepare the test dataset to merge with the training dataset
'''
print('QUIT')
test = pd.read_csv('test.csv')
itemcat = pd.read_csv('items.csv')
#itemcat.head()
test = test.drop('ID',axis=1)
test['date_block_num'] = 34
test['item_cnt_month'] = 0
test.sort_values(by=['shop_id','item_id'], inplace = True)
itemcat.head()
train = pd.merge(train,itemcat[['item_id','item_category_id']], on = ['item_id'], how = 'left')
train.rename(columns={'item_category_id':'item_cat'}, inplace= True)
test = pd.merge(test,itemcat[['item_id','item_category_id']], on = ['item_id'], how = 'left')
test.rename(columns={'item_category_id':'item_cat'}, inplace= True)
'''
# In[4]:


#test.info()
#train.info()


# In[5]:


#test = test.drop('ID',axis=1)
#test.head()
#matrix = pd.concat([train,test], keys = ['shop_id','item_id','date_block_num'], ignore_index = True, sort= False)
#matrix.info()


# In[6]:


def downcast(df):
    #finding floating point columns
    float_cols = [c for c in df if df[c].dtype == 'float64']
    #finding integer columns
    int_cols = [c for c in df if df[c].dtype in ['int32','int64']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

train = downcast(train)
del matrix
train.info()


# In[7]:


train.date_block_num.unique()


# In[9]:


sr = {} # for sales record
srf = {} # for recording first sale
matrix =  train.copy()
default = -1
matrix['last_item_sale'] = default # guessing the sell happened long ago, needs to work on this value
for i in matrix.itertuples():
    idx = i.Index
    key = i.item_id
    if key not in sr:
        if i.item_cnt_month != 0:
            sr[key] = i.date_block_num
            srf[key] = i.date_block_num
    else:
        if i.date_block_num > sr[key]:
            matrix.at[idx,'last_item_sale'] = i.date_block_num - sr[key]
            sr[key] = i.date_block_num





# In[10]:


## Months after first item sale
#matrix['months_first_item_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id'])['date_block_num'].transform('min')
#matrix['months_first_item_shop_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

lel = pd.DataFrame(srf.items(),columns=['item_id','month_first_item_sale'])
matrix = pd.merge(matrix,lel,on='item_id',how='left')
matrix['month_first_item_sale'] = matrix['month_first_item_sale'].fillna(-1)
#matrix['months_since_first_item_sale'] = matrix['date_block_num'] - matrix['month_first_item_sale']
#matrix['months_since_first_item_sale'] = matrix.months_since_first_item_sale.apply(lambda x: x if x>0 else 35)
#matrix.drop(columns = 'month_first_item_sale',axis=1)
del lel
# In[14]:

#sr = {} # for shop-item sales record
#srf = {} # for shop-item first sale
#default = -1 # guessing the sell happened long ago, needs to work on this value
#matrix['last_shop_item_sale'] = default
#for i in matrix.itertuples():
#    idx = i.Index
#    key = str(i.item_id) + ' ' + str(i.shop_id)
#    if key not in sr:
#        if i.item_cnt_month != 0:
#            sr[key] = i.date_block_num
#            srf[key] = i.date_block_num
#    else:
#        matrix.at[idx,'last_item_sale'] = i.date_block_num - sr[key]
#        sr[key] = i.date_block_num

#lel = pd
train = matrix
del matrix

# In[9]:


# Function for introducing lag features in the dataset. "COL" is the intended feature
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]] #made a temporary dataframe
    for i in tqdm.tqdm(lags):
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

ts = time.time()
train = lag_feature(train, lags, 'item_cnt_month')
time.time() - ts


# In[ ]:


train.fillna(0, inplace = True)


# In[ ]:


train.info()


# In[ ]:


# Price analysis. In this section we will dome some experiment on price analysis will see if there are ups and
# downs in price


# In[ ]:


train.columns


# In[ ]:


# Now we shall do some investigation on item price over both shop and time


# In[ ]:


exp = sales_train.groupby(['shop_id','item_id'])['item_price'].max()
exp = pd.DataFrame(exp).reset_index()
exp.rename(columns={'item_price':'max'}, inplace = True)
exp.head()


# In[ ]:


exp1 = sales_train.groupby(['shop_id','item_id'])['item_price'].min()
exp1 = pd.DataFrame(exp1).reset_index()
exp1.rename(columns={'item_price':'min'}, inplace = True)
exp1.head()


# In[ ]:


matrix = pd.merge(exp,exp1, on=['shop_id','item_id'],how = 'left')


# In[ ]:


matrix.head()


# In[ ]:


matrix['change'] = np.abs(matrix['max'] -matrix['min'])


# In[ ]:


matrix


# In[ ]:


# This means there is a change of item price over TIME. Now we need to see if there is variation of item price
# over the SHOPS.


# In[ ]:


sales_train.head()


# In[ ]:


shop_matrix = sales_train.pivot_table(index=['date','item_id'], columns = 'shop_id', values = 'item_price',aggfunc= np.mean, fill_value=0)


# In[ ]:


shop_matrix = shop_matrix.droplevel('date',axis=0)


# In[ ]:


shop_matrix_2d = shop_matrix.values


# In[ ]:


for i in range (0,shop_matrix_2d.shape[0]):
    maxs = np.max(shop_matrix_2d[i])
    shop_matrix_2d[i] = shop_matrix_2d[i]/maxs


# In[ ]:


shop_matrix_2d.shape


# In[ ]:


shop_index = np.zeros((60,1)) # shop_index determines the usual practice of selling a product in a shopt
for i in range(0,shop_matrix_2d.shape[1]):
    temp = shop_matrix_2d[:][i]
    shop_index[i] = np.mean(temp[temp>0])
    #shop_index[i] = np.mean(np.unique(shop_matrix_2d[:][i]))


# In[ ]:


train['shop_index'] = ''


# In[ ]:


for i in range(0,60):
    train.loc[train.shop_id==i,'shop_index'] = shop_index[i]


# In[ ]:


np.min(train.shop_index)


# In[ ]:


def nday_month(x):
    if x<12:
        year = 2013
    if x>=12 and x<22:
        year = 2014
    if x>=22:
        year = 2015
    month = np.mod(x,12) + 1
    temp = monthrange(year,month)[1]
    return temp


# In[ ]:


nday_month(32)


# In[ ]:


train.head()


# In[ ]:


train['day_in_month'] = ''


# In[ ]:


train['day_in_month'] = train.date_block_num.apply(lambda x: nday_month(x))


# In[ ]:


train.head()


# In[ ]:


def wday_month(x):
    if x<12:
        year = 2013
    if x>=12 and x<22:
        year = 2014
    if x>=22:
        year = 2015
    month = np.mod(x,12) + 1
    temp = np.array(calendar.monthcalendar(year,month))[:,calendar.SATURDAY]
    return (temp>0).sum()


# In[ ]:


train['wday_in_month'] = ''


# In[ ]:


train['wday_in_month'] = train.date_block_num.apply(lambda x: wday_month(x))


# In[ ]:


train['month_index'] = ''


# In[ ]:


train['month_index'] = train.date_block_num.apply(lambda x: np.mod(x,12)+1)


# In[ ]:


def downcast(df):
    #finding floating point columns
    float_cols = [c for c in df if df[c].dtype == 'float64']
    #finding integer columns
    int_cols = [c for c in df if df[c].dtype in ['int32','int64']]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df

train = downcast(train)
train.info()


# In[ ]:


exp = train.groupby(['date_block_num','item_id'])['item_cnt_month'].mean()
exp = pd.DataFrame(exp).reset_index()
exp = exp.rename(columns={'item_cnt_month':'avg_item_sale'})
temp = pd.merge(train,exp,on=['item_id','date_block_num'],how = 'left')
temp.avg_item_sale = temp.avg_item_sale.astype(np.float32)
# adding lag feature
temp = lag_feature(temp,lags,'avg_item_sale')
temp.fillna(0,inplace=True)
temp = temp.drop(columns=['avg_item_sale'],axis=1)
train = temp
traincopy = train.copy()


# In[ ]:


train.info()


# In[ ]:


exp = train.groupby(['date_block_num','shop_id'])['item_cnt_month'].mean()
exp = pd.DataFrame(exp).reset_index()
exp = exp.rename(columns={'item_cnt_month':'avg_shop_sale'})
temp = pd.merge(train,exp,on=['shop_id','date_block_num'],how = 'left')
temp.avg_shop_sale = temp.avg_shop_sale.astype(np.float32)
# adding lag feature
temp = lag_feature(temp,lags,'avg_shop_sale')
temp.fillna(0,inplace=True)
temp = temp.drop(columns=['avg_shop_sale'],axis=1)


# In[ ]:


train = temp
traincopy = train.copy()


# In[ ]:


exp = train.groupby(['date_block_num','item_cat'])['item_cnt_month'].mean()
exp = pd.DataFrame(exp).reset_index()
exp = exp.rename(columns={'item_cnt_month':'avg_cat_sale'})
temp = pd.merge(train,exp,on=['item_cat','date_block_num'],how = 'left')
temp.avg_cat_sale = temp.avg_cat_sale.astype(np.float32)
# adding lag feature
temp = lag_feature(temp,lags,'avg_cat_sale')
temp.fillna(0,inplace=True)
temp = temp.drop(columns=['avg_cat_sale'],axis=1)
train = temp
traincopy = train.copy


# In[ ]:


exp = train.groupby(['date_block_num','item_id'])['item_cnt_month'].sum()
exp = pd.DataFrame(exp).reset_index()
exp = exp.rename(columns={'item_cnt_month':'tot_item_sale'})
temp = pd.merge(train,exp,on=['item_id','date_block_num'],how = 'left')
temp.tot_item_sale = temp.tot_item_sale.astype(np.float32)
# adding lag feature
temp = lag_feature(temp,lags,'tot_item_sale')
temp.fillna(0,inplace=True)
temp = temp.drop(columns=['tot_item_sale'],axis=1)
train = temp
traincopy = train.copy()


# In[ ]:


exp = train.groupby(['date_block_num','shop_id'])['item_cnt_month'].sum()
exp = pd.DataFrame(exp).reset_index()
exp = exp.rename(columns={'item_cnt_month':'tot_shop_sale'})
temp = pd.merge(train,exp,on=['shop_id','date_block_num'],how = 'left')
temp.tot_shop_sale = temp.tot_shop_sale.astype(np.float32)
# adding lag feature
temp = lag_feature(temp,lags,'tot_shop_sale')
temp.fillna(0,inplace=True)
temp = temp.drop(columns=['tot_shop_sale'],axis=1)
train = temp
traincopy = train.copy()


# In[ ]:


exp = train.groupby(['date_block_num','item_cat'])['item_cnt_month'].sum()
exp = pd.DataFrame(exp).reset_index()
exp = exp.rename(columns={'item_cnt_month':'tot_cat_sale'})
temp = pd.merge(train,exp,on=['item_cat','date_block_num'],how = 'left')
temp.tot_cat_sale = temp.tot_cat_sale.astype(np.float32)
# adding lag feature
temp = lag_feature(temp,lags,'tot_cat_sale')
temp.fillna(0,inplace=True)
temp = temp.drop(columns=['tot_cat_sale'],axis=1)
train = temp
traincopy = train.copy()


# In[ ]:


# Price Trend Feature
matrix = train.copy()
def strend(row):
    for i in lags:
        if row['pr_trend_lag_'+str(i)]:
            return row['pr_trend_lag_'+str(i)]
    return 0

exp = sales_train.groupby('item_id')['item_price'].mean()
exp = pd.DataFrame(exp)
exp = exp.reset_index()
exp = exp.rename(columns={'item_price':'avg_item_price'})
matrix = pd.merge(matrix,exp,on=['item_id'],how='left')
matrix = lag_feature(matrix,lags,'avg_item_price')
matrix.fillna(0,inplace=True)
for i in tqdm.tqdm(lags):
    matrix['pr_trend_lag_'+str(i)] = ''
    matrix['pr_trend_lag_'+str(i)] = (matrix['avg_item_price_lag_'+str(i)] - matrix['avg_item_price'])/matrix['avg_item_price']
matrix['slope_price'] = matrix.apply(strend, axis=1)
drop_item = ['avg_item_price']
for i in tqdm.tqdm(lags):
    drop_item += ['pr_trend_lag_'+str(i)]
matrix = matrix.drop(columns=drop_item,axis=1)
#train = matrix
#traincopy = train.copy()


# In[ ]:


train = matrix
traincopy = train.copy()
del matrix


# In[ ]:


############### MODEL PREPARATION ###################
train = traincopy
train.columns


# In[ ]:


#monthly sale
exp = sales_train.groupby('date_block_num')['item_cnt_day'].sum()
monthly_sale = pd.DataFrame(exp).reset_index()
temp = pd.merge(train,monthly_sale,on=['date_block_num'],how= 'left')
temp = temp.rename(columns = {'item_cnt_day':'monthly_sale'})
temp = lag_feature(temp,lags,'monthly_sale')
temp.fillna(0,inplace= True)
temp = temp.drop(columns=['monthly_sale'],axis=1)
train = temp
traincopy = train.copy()


# In[ ]:
# further downcasting
train.shop_index = train.shop_index.astype(np.float32)
train.date_block_num = train.date_block_num.astype(np.int8)
train.shop_id = train.shop_id.astype(np.int8)

traincopy = train.copy()
train = downcast(train)
#train = train.drop(columns=['item_price'],axis=1)
#train['item_cnt_month'] = train.item_cnt_month.clip(0,20)
train.fillna(0,inplace=True)
train['shop_index'] = train['shop_index'].astype(np.float32)
train = train[train['date_block_num']>= np.max(lags)]
valid_month = 32
test_month = 33
xtrain = train[train.date_block_num<valid_month].drop(columns='item_cnt_month', axis =1)
ytrain = train[train.date_block_num<valid_month]['item_cnt_month']
xvalid = train[train.date_block_num==valid_month].drop(columns='item_cnt_month', axis =1)
yvalid = train[train.date_block_num==valid_month]['item_cnt_month']
xtest = train[train.date_block_num==test_month].drop(columns='item_cnt_month', axis =1)
ytest = train[train.date_block_num==test_month]['item_cnt_month']

# In[ ]:
train.info()
print('XGBOOST IS STARTING....')

ts = time.time()
import sklearn
model = XGBRegressor(
    max_depth=12,
    n_estimators=1000,
    min_child_weight=0.5,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.1,
    #tree_method = 'exact',
    seed=42)

model.fit(
    xtrain,
    ytrain,
    eval_metric="rmse",
    eval_set=[(xtrain, ytrain), (xvalid, yvalid)],
    verbose=True,
    early_stopping_rounds = 10)

time.time() - ts


# In[ ]:


def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)
plot_features(model, (10,14))


# In[ ]:


ypred = model.predict(xtest)
rms = sqrt(mean_squared_error(ytest.values, ypred))
print(rms)
# In[ ]:


## Preparing submission file
'''
test = pd.read_csv('test.csv')


# In[ ]:


temp = xtest[['shop_id','item_id']]
temp['item_cnt_month'] = ypred


# In[ ]:


exp = pd.merge(test,temp,on=['shop_id','item_id'], how= 'inner')
exp = exp.drop(columns=['shop_id','item_id'], axis = 1)
exp.item_cnt_month = exp.item_cnt_month.clip(0,20)


# In[ ]:


exp.to_csv('submission_xgboost1.csv', index = False)
'''

# In[ ]:


#train.shape


# In[ ]:
