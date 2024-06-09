#!/usr/bin/env python
# coding: utf-8

# ### Problem statement:-
# To reduce customer churn, telecom companies need to predict which customers are at high risk of churn.
# In this project, we will analyse customer-level data of a leading telecom firm, build predictive models to identify customers at high risk of churn and identify the main indicators of churn.
# 
# Retaining high profitable customers is the main business goal here.
# 

# ## Steps to get the Churn of TELECOM:-
# 1. Reading, understanding and visualising the data
# 2. Preparing the data for modelling
# 3. Building the model
# 4. Evaluate the model
# 

# In[4]:


# Importing the libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[5]:


pd.set_option('display.max_columns', 500)


# # Reading and understanding the data

# In[6]:


# Reading the dataset
df = pd.read_csv('telecom_churn_data.csv')
df.head()


# In[7]:


df.shape


# In[8]:


df.info()


# In[9]:


df.describe()


# ## Handling missing values

# #### Handling missing values in columns

# In[10]:


# Cheking percent of missing values in columns
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# In[11]:


# List the columns having more than 30% missing values
col_list_missing_30 = list(df_missing_columns.index[df_missing_columns['null'] > 30])


# In[12]:


# Delete the columns having more than 30% missing values
df = df.drop(col_list_missing_30, axis=1)


# In[13]:


df.shape


# ##### Deleting the date columns as the date columns are not required in our analysis

# In[14]:


# List the date columns
date_cols = [k for k in df.columns.to_list() if 'date' in k]
print(date_cols) 


# In[15]:


# Dropping date columns
df = df.drop(date_cols, axis=1)


# Dropping circle_id column as this column has only one unique value. Hence there will be no impact of this column on the data analysis.

# In[16]:


# Drop circle_id column
df = df.drop('circle_id', axis=1)


# In[17]:


df.shape


# ### Filter high-value customers

# Creating column `avg_rech_amt_6_7` by summing up total recharge amount of month 6 and 7. Then taking the average of the sum.

# In[18]:


df['avg_rech_amt_6_7'] = (df['total_rech_amt_6'] + df['total_rech_amt_7'])/2


# Finding the 70th percentile of the avg_rech_amt_6_7

# In[19]:


X = df['avg_rech_amt_6_7'].quantile(0.7)
X


# Filter the customers, who have recharged more than or equal to X.

# In[20]:


df = df[df['avg_rech_amt_6_7'] >= X]
df.head()


# In[21]:


df.shape


# We can see that we have around ***~30K*** rows after filtering

# #### Handling missing values in rows

# In[22]:


# Count the rows having more than 50% missing values
df_missing_rows_50 = df[(df.isnull().sum(axis=1)) > (len(df.columns)//2)]
df_missing_rows_50.shape


# In[23]:


# Deleting the rows having more than 50% missing values
df = df.drop(df_missing_rows_50.index)
df.shape


# In[24]:


# Checking the missing values in columns again
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# Looks like MOU for all the types of calls for the month of September (9) have missing values together for any particular record.
# 
# Lets check the records for the MOU for Sep(9), in which these coulmns have missing values together.

# In[25]:


# Listing the columns of MOU Sep(9)
print(((df_missing_columns[df_missing_columns['null'] == 5.32]).index).to_list())


# In[26]:


# Creating a dataframe with the condition, in which MOU for Sep(9) are null
df_null_mou_9 = df[(df['loc_og_t2m_mou_9'].isnull()) & (df['loc_ic_t2f_mou_9'].isnull()) & (df['roam_og_mou_9'].isnull()) & (df['std_ic_t2m_mou_9'].isnull()) &
  (df['loc_og_t2t_mou_9'].isnull()) & (df['std_ic_t2t_mou_9'].isnull()) & (df['loc_og_t2f_mou_9'].isnull()) & (df['loc_ic_mou_9'].isnull()) &
  (df['loc_og_t2c_mou_9'].isnull()) & (df['loc_og_mou_9'].isnull()) & (df['std_og_t2t_mou_9'].isnull()) & (df['roam_ic_mou_9'].isnull()) &
  (df['loc_ic_t2m_mou_9'].isnull()) & (df['std_og_t2m_mou_9'].isnull()) & (df['loc_ic_t2t_mou_9'].isnull()) & (df['std_og_t2f_mou_9'].isnull()) & 
  (df['std_og_t2c_mou_9'].isnull()) & (df['og_others_9'].isnull()) & (df['std_og_mou_9'].isnull()) & (df['spl_og_mou_9'].isnull()) & 
  (df['std_ic_t2f_mou_9'].isnull()) & (df['isd_og_mou_9'].isnull()) & (df['std_ic_mou_9'].isnull()) & (df['offnet_mou_9'].isnull()) & 
  (df['isd_ic_mou_9'].isnull()) & (df['ic_others_9'].isnull()) & (df['std_ic_t2o_mou_9'].isnull()) & (df['onnet_mou_9'].isnull()) & 
  (df['spl_ic_mou_9'].isnull())]

df_null_mou_9.head()


# In[27]:


df_null_mou_9.shape


# In[28]:


# Deleting the records for which MOU for Sep(9) are null
df = df.drop(df_null_mou_9.index)


# In[29]:


# Again Cheking percent of missing values in columns
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# Looks like MOU for all the types of calls for the month of Aug (8) have missing values together for any particular record.
# 
# Lets check the records for the MOU for Aug(8), in which these coulmns have missing values together.

# In[30]:


# Listing the columns of MOU Aug(8)
print(((df_missing_columns[df_missing_columns['null'] == 0.55]).index).to_list())


# In[31]:


# Creating a dataframe with the condition, in which MOU for Aug(8) are null
df_null_mou_8 = df[(df['loc_og_t2m_mou_8'].isnull()) & (df['loc_ic_t2f_mou_8'].isnull()) & (df['roam_og_mou_8'].isnull()) & (df['std_ic_t2m_mou_8'].isnull()) &
  (df['loc_og_t2t_mou_8'].isnull()) & (df['std_ic_t2t_mou_8'].isnull()) & (df['loc_og_t2f_mou_8'].isnull()) & (df['loc_ic_mou_8'].isnull()) &
  (df['loc_og_t2c_mou_8'].isnull()) & (df['loc_og_mou_8'].isnull()) & (df['std_og_t2t_mou_8'].isnull()) & (df['roam_ic_mou_8'].isnull()) &
  (df['loc_ic_t2m_mou_8'].isnull()) & (df['std_og_t2m_mou_8'].isnull()) & (df['loc_ic_t2t_mou_8'].isnull()) & (df['std_og_t2f_mou_8'].isnull()) & 
  (df['std_og_t2c_mou_8'].isnull()) & (df['og_others_8'].isnull()) & (df['std_og_mou_8'].isnull()) & (df['spl_og_mou_8'].isnull()) & 
  (df['std_ic_t2f_mou_8'].isnull()) & (df['isd_og_mou_8'].isnull()) & (df['std_ic_mou_8'].isnull()) & (df['offnet_mou_8'].isnull()) & 
  (df['isd_ic_mou_8'].isnull()) & (df['ic_others_8'].isnull()) & (df['std_ic_t2o_mou_8'].isnull()) & (df['onnet_mou_8'].isnull()) & 
  (df['spl_ic_mou_8'].isnull())]

df_null_mou_8.head()


# In[32]:


# Deleting the records for which MOU for Aug(8) are null
df = df.drop(df_null_mou_8.index)


# In[33]:


# Again cheking percent of missing values in columns
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# Looks like MOU for all the types of calls for the month of Jun (6) have missing values together for any particular record.
# 
# Lets check the records for the MOU for Jun(6), in which these coulmns have missing values together.

# In[34]:


# Listing the columns of MOU Jun(6)
print(((df_missing_columns[df_missing_columns['null'] == 0.44]).index).to_list())


# In[35]:


# Creating a dataframe with the condition, in which MOU for Jun(6) are null
df_null_mou_6 = df[(df['loc_og_t2m_mou_6'].isnull()) & (df['loc_ic_t2f_mou_6'].isnull()) & (df['roam_og_mou_6'].isnull()) & (df['std_ic_t2m_mou_6'].isnull()) &
  (df['loc_og_t2t_mou_6'].isnull()) & (df['std_ic_t2t_mou_6'].isnull()) & (df['loc_og_t2f_mou_6'].isnull()) & (df['loc_ic_mou_6'].isnull()) &
  (df['loc_og_t2c_mou_6'].isnull()) & (df['loc_og_mou_6'].isnull()) & (df['std_og_t2t_mou_6'].isnull()) & (df['roam_ic_mou_6'].isnull()) &
  (df['loc_ic_t2m_mou_6'].isnull()) & (df['std_og_t2m_mou_6'].isnull()) & (df['loc_ic_t2t_mou_6'].isnull()) & (df['std_og_t2f_mou_6'].isnull()) & 
  (df['std_og_t2c_mou_6'].isnull()) & (df['og_others_6'].isnull()) & (df['std_og_mou_6'].isnull()) & (df['spl_og_mou_6'].isnull()) & 
  (df['std_ic_t2f_mou_6'].isnull()) & (df['isd_og_mou_6'].isnull()) & (df['std_ic_mou_6'].isnull()) & (df['offnet_mou_6'].isnull()) & 
  (df['isd_ic_mou_6'].isnull()) & (df['ic_others_6'].isnull()) & (df['std_ic_t2o_mou_6'].isnull()) & (df['onnet_mou_6'].isnull()) & 
  (df['spl_ic_mou_6'].isnull())]

df_null_mou_6.head()


# In[36]:


# Deleting the records for which MOU for Jun(6) are null
df = df.drop(df_null_mou_6.index)


# In[37]:


# Again cheking percent of missing values in columns
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# Looks like MOU for all the types of calls for the month of July (7) have missing values together for any particular record.
# 
# Lets check the records for the MOU for Jul(7), in which these coulmns have missing values together.

# In[38]:


# Listing the columns of MOU Jul(7)
print(((df_missing_columns[df_missing_columns['null'] == 0.12]).index).to_list())


# In[39]:


# Creating a dataframe with the condition, in which MOU for Jul(7) are null
df_null_mou_7 = df[(df['loc_og_t2m_mou_7'].isnull()) & (df['loc_ic_t2f_mou_7'].isnull()) & (df['roam_og_mou_7'].isnull()) & (df['std_ic_t2m_mou_7'].isnull()) &
  (df['loc_og_t2t_mou_7'].isnull()) & (df['std_ic_t2t_mou_7'].isnull()) & (df['loc_og_t2f_mou_7'].isnull()) & (df['loc_ic_mou_7'].isnull()) &
  (df['loc_og_t2c_mou_7'].isnull()) & (df['loc_og_mou_7'].isnull()) & (df['std_og_t2t_mou_7'].isnull()) & (df['roam_ic_mou_7'].isnull()) &
  (df['loc_ic_t2m_mou_7'].isnull()) & (df['std_og_t2m_mou_7'].isnull()) & (df['loc_ic_t2t_mou_7'].isnull()) & (df['std_og_t2f_mou_7'].isnull()) & 
  (df['std_og_t2c_mou_7'].isnull()) & (df['og_others_7'].isnull()) & (df['std_og_mou_7'].isnull()) & (df['spl_og_mou_7'].isnull()) & 
  (df['std_ic_t2f_mou_7'].isnull()) & (df['isd_og_mou_7'].isnull()) & (df['std_ic_mou_7'].isnull()) & (df['offnet_mou_7'].isnull()) & 
  (df['isd_ic_mou_7'].isnull()) & (df['ic_others_7'].isnull()) & (df['std_ic_t2o_mou_7'].isnull()) & (df['onnet_mou_7'].isnull()) & 
  (df['spl_ic_mou_7'].isnull())]

df_null_mou_7.head()


# In[40]:


# Deleting the records for which MOU for Jul(7) are null
df = df.drop(df_null_mou_7.index)


# In[41]:


# Again cheking percent of missing values in columns
df_missing_columns = (round(((df.isnull().sum()/len(df.index))*100),2).to_frame('null')).sort_values('null', ascending=False)
df_missing_columns


# We can see there are no more missing values in any columns.

# In[42]:


df.shape


# In[43]:


# Checking percentage of rows we have lost while handling the missing values
round((1- (len(df.index)/30011)),2)


# We can see that we have lost almost 7% records. But we have enough number of records to do our analysis.

# ### Tag churners

# Now tag the churned customers (churn=1, else 0) based on the fourth month as follows: Those who have not made any calls (either incoming or outgoing) AND have not used mobile internet even once in the churn phase. 

# In[44]:


df['churn'] = np.where((df['total_ic_mou_9']==0) & (df['total_og_mou_9']==0) & (df['vol_2g_mb_9']==0) & (df['vol_3g_mb_9']==0), 1, 0)


# In[45]:


df.head()


# #### Deleting all the attributes corresponding to the churn phase

# In[46]:


# List the columns for churn month(9)
col_9 = [col for col in df.columns.to_list() if '_9' in col]
print(col_9)


# In[47]:


# Deleting the churn month columns
df = df.drop(col_9, axis=1)


# In[48]:


# Dropping sep_vbc_3g column
df = df.drop('sep_vbc_3g', axis=1)


# #### Checking churn percentage

# In[49]:


round(100*(df['churn'].mean()),2)


# There is very little percentage of churn rate. We will take care of the class imbalance later.

# ## Outliers treatment

# In the filtered dataset except mobile_number and churn columns all the columns are numeric types. Hence, converting mobile_number and churn datatype to object.

# In[50]:


df['mobile_number'] = df['mobile_number'].astype(object)
df['churn'] = df['churn'].astype(object)


# In[51]:


df.info()


# In[52]:


# List only the numeric columns
numeric_cols = df.select_dtypes(exclude=['object']).columns
print(numeric_cols)


# In[53]:


# Removing outliers below 10th and above 90th percentile
for col in numeric_cols: 
    q1 = df[col].quantile(0.10)
    q3 = df[col].quantile(0.90)
    iqr = q3-q1
    range_low  = q1-1.5*iqr
    range_high = q3+1.5*iqr
    # Assigning the filtered dataset into data
    data = df.loc[(df[col] > range_low) & (df[col] < range_high)]

data.shape


# ### Derive new features

# In[54]:


# List the columns of total mou, rech_num and rech_amt
[total for total in data.columns.to_list() if 'total' in total]


# #### Deriving new column `decrease_mou_action`
# This column indicates whether the minutes of usage of the customer has decreased in the action phase than the good phase.

# In[55]:


# Total mou at good phase incoming and outgoing
data['total_mou_good'] = (data['total_og_mou_6'] + data['total_ic_mou_6'])


# In[56]:


# Avg. mou at action phase
# We are taking average because there are two months(7 and 8) in action phase
data['avg_mou_action'] = (data['total_og_mou_7'] + data['total_og_mou_8'] + data['total_ic_mou_7'] + data['total_ic_mou_8'])/2


# In[57]:


# Difference avg_mou_good and avg_mou_action
data['diff_mou'] = data['avg_mou_action'] - data['total_mou_good']


# In[58]:


# Checking whether the mou has decreased in action phase
data['decrease_mou_action'] = np.where((data['diff_mou'] < 0), 1, 0)


# In[59]:


data.head()


# #### Deriving new column `decrease_rech_num_action`
# This column indicates whether the number of recharge of the customer has decreased in the action phase than the good phase.

# In[60]:


# Avg rech number at action phase
data['avg_rech_num_action'] = (data['total_rech_num_7'] + data['total_rech_num_8'])/2


# In[61]:


# Difference total_rech_num_6 and avg_rech_action
data['diff_rech_num'] = data['avg_rech_num_action'] - data['total_rech_num_6']


# In[62]:


# Checking if rech_num has decreased in action phase
data['decrease_rech_num_action'] = np.where((data['diff_rech_num'] < 0), 1, 0)


# In[63]:


data.head()


# #### Deriving new column `decrease_rech_amt_action`
# This column indicates whether the amount of recharge of the customer has decreased in the action phase than the good phase.

# In[64]:


# Avg rech_amt in action phase
data['avg_rech_amt_action'] = (data['total_rech_amt_7'] + data['total_rech_amt_8'])/2


# In[65]:


# Difference of action phase rech amt and good phase rech amt
data['diff_rech_amt'] = data['avg_rech_amt_action'] - data['total_rech_amt_6']


# In[66]:


# Checking if rech_amt has decreased in action phase
data['decrease_rech_amt_action'] = np.where((data['diff_rech_amt'] < 0), 1, 0) 


# In[67]:


data.head()


# #### Deriving new column `decrease_arpu_action`
# This column indicates whether the average revenue per customer has decreased in the action phase than the good phase.

# In[68]:


# ARUP in action phase
data['avg_arpu_action'] = (data['arpu_7'] + data['arpu_8'])/2


# In[69]:


# Difference of good and action phase ARPU
data['diff_arpu'] = data['avg_arpu_action'] - data['arpu_6']


# In[70]:


# Checking whether the arpu has decreased on the action month
data['decrease_arpu_action'] = np.where(data['diff_arpu'] < 0, 1, 0)


# In[71]:


data.head()


# #### Deriving new column `decrease_vbc_action`
# This column indicates whether the volume based cost of the customer has decreased in the action phase than the good phase.

# In[72]:


# VBC in action phase
data['avg_vbc_3g_action'] = (data['jul_vbc_3g'] + data['aug_vbc_3g'])/2


# In[73]:


# Difference of good and action phase VBC
data['diff_vbc'] = data['avg_vbc_3g_action'] - data['jun_vbc_3g']


# In[74]:


# Checking whether the VBC has decreased on the action month
data['decrease_vbc_action'] = np.where(data['diff_vbc'] < 0 , 1, 0)


# In[75]:


data.head()


# ## EDA

# ### Univariate analysis

# ##### Churn rate on the basis whether the customer decreased her/his MOU in action month

# In[76]:


# Converting churn column to int in order to do aggfunc in the pivot table
data['churn'] = data['churn'].astype('int64')


# In[77]:


data.pivot_table(values='churn', index='decrease_mou_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# ***Analysis***
# 
# We can see that the churn rate is more for the customers, whose minutes of usage(mou) decreased in the action phase than the good phase. 

# ##### Churn rate on the basis whether the customer decreased her/his number of recharge in action month

# In[78]:


data.pivot_table(values='churn', index='decrease_rech_num_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# ***Analysis***
# 
# As expected, the churn rate is more for the customers, whose number of recharge in the action phase is lesser than the number in good phase.

# ##### Churn rate on the basis whether the customer decreased her/his amount of recharge in action month

# In[79]:


data.pivot_table(values='churn', index='decrease_rech_amt_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# ***Analysis***
# 
# Here also we see the same behaviour. The churn rate is more for the customers, whose amount of recharge in the action phase is lesser than the amount in good phase.

# ##### Churn rate on the basis whether the customer decreased her/his volume based cost in action month

# In[80]:


data.pivot_table(values='churn', index='decrease_vbc_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# ***Analysis***
# 
# Here we see the expected result. The churn rate is more for the customers, whose volume based cost in action month is increased. That means the customers do not do the monthly recharge more when they are in the action phase.

# ##### Analysis of the average revenue per customer (churn and not churn) in the action phase

# In[81]:


# Creating churn dataframe
data_churn = data[data['churn'] == 1]
# Creating not churn dataframe
data_non_churn = data[data['churn'] == 0]


# In[82]:


# Distribution plot
ax = sns.distplot(data_churn['avg_arpu_action'],label='churn',hist=False)
ax = sns.distplot(data_non_churn['avg_arpu_action'],label='not churn',hist=False)
ax.set(xlabel='Action phase ARPU')


# Average revenue per user (ARPU) for the churned customers is mostly densed on the 0 to 900. The higher ARPU customers are less likely to be churned.
# 
# ARPU for the not churned customers is mostly densed on the 0 to 1000. 

# ##### Analysis of the minutes of usage MOU (churn and not churn) in the action phase

# In[83]:


# Distribution plot
ax = sns.distplot(data_churn['total_mou_good'],label='churn',hist=False)
ax = sns.distplot(data_non_churn['total_mou_good'],label='non churn',hist=False)
ax.set(xlabel='Action phase MOU')


# Minutes of usage(MOU) of the churn customers is mostly populated on the 0 to 2500 range. Higher the MOU, lesser the churn probability.

# ### Bivariate analysis

# ##### Analysis of churn rate by the decreasing recharge amount and number of recharge in the action phase

# In[84]:


data.pivot_table(values='churn', index='decrease_rech_amt_action', columns='decrease_rech_num_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# ***Analysis***
# 
# We can see from the above plot, that the churn rate is more for the customers, whose recharge amount as well as number of recharge have decreased in the action phase than the good phase.

# ##### Analysis of churn rate by the decreasing recharge amount and volume based cost in the action phase

# In[85]:


data.pivot_table(values='churn', index='decrease_rech_amt_action', columns='decrease_vbc_action', aggfunc='mean').plot.bar()
plt.ylabel('churn rate')
plt.show()


# ***Analysis***
# 
# Here, also we can see that the churn rate is more for the customers, whose recharge amount is decreased along with the volume based cost is increased in the action month.

# ##### Analysis of recharge amount and number of recharge in action month

# In[86]:


plt.figure(figsize=(10,6))
ax = sns.scatterplot('avg_rech_num_action','avg_rech_amt_action', hue='churn', data=data)


# ***Analysis***
# 
# We can see from the above pattern that the recharge number and the recharge amount are mostly propotional. More the number of recharge, more the amount of the recharge.

# #### Dropping few derived columns, which are not required in further analysis

# In[87]:


data = data.drop(['total_mou_good','avg_mou_action','diff_mou','avg_rech_num_action','diff_rech_num','avg_rech_amt_action',
                 'diff_rech_amt','avg_arpu_action','diff_arpu','avg_vbc_3g_action','diff_vbc','avg_rech_amt_6_7'], axis=1)


# ## Train-Test Split

# In[88]:


# Import library
from sklearn.model_selection import train_test_split


# In[89]:


# Putting feature variables into X
X = data.drop(['mobile_number','churn'], axis=1)


# In[90]:


# Putting target variable to y
y = data['churn']


# In[91]:


# Splitting data into train and test set 80:20
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=100)


# ### Dealing with data imbalance

# We are creating synthetic samples by doing upsampling using SMOTE(Synthetic Minority Oversampling Technique).

# In[92]:


# Imporing SMOTE
from imblearn.over_sampling import SMOTE


# In[ ]:


# Instantiate SMOTE
sm = SMOTE(random_state=27)


# In[ ]:


# Fittign SMOTE to the train set
X_train, y_train = sm.fit_sample(X_train, y_train)


# ### Feature Scaling

# In[ ]:


# Standardization method
from sklearn.preprocessing import StandardScaler


# In[ ]:


# Instantiate the Scaler
scaler = StandardScaler()


# In[ ]:


# List of the numeric columns
cols_scale = X_train.columns.to_list()
# Removing the derived binary columns 
cols_scale.remove('decrease_mou_action')
cols_scale.remove('decrease_rech_num_action')
cols_scale.remove('decrease_rech_amt_action')
cols_scale.remove('decrease_arpu_action')
cols_scale.remove('decrease_vbc_action')


# In[ ]:


# Fit the data into scaler and transform
X_train[cols_scale] = scaler.fit_transform(X_train[cols_scale])


# In[ ]:


X_train.head()


# ##### Scaling the test set
# We don't fit scaler on the test set. We only transform the test set.

# In[93]:


# Transform the test set
X_test[cols_scale] = scaler.transform(X_test[cols_scale])
X_test.head()


# # Model with PCA

# In[94]:


#Import PCA
from sklearn.decomposition import PCA


# In[95]:


# Instantiate PCA
pca = PCA(random_state=42)


# In[96]:


# Fit train set on PCA
pca.fit(X_train)


# In[97]:


# Principal components
pca.components_


# In[98]:


# Cumuliative varinace of the PCs
variance_cumu = np.cumsum(pca.explained_variance_ratio_)
print(variance_cumu)


# In[99]:


# Plotting scree plot
fig = plt.figure(figsize = (10,6))
plt.plot(variance_cumu)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance')


# We can see that `60 components` explain amost more than 90% variance of the data. So, we will perform PCA with 60 components.

# ##### Performing PCA with 60 components

# In[100]:


# Importing incremental PCA
from sklearn.decomposition import IncrementalPCA


# In[101]:


# Instantiate PCA with 60 components
pca_final = IncrementalPCA(n_components=60)


# In[102]:


# Fit and transform the X_train
X_train_pca = pca_final.fit_transform(X_train)


# ##### Applying transformation on the test set
# We are only doing Transform in the test set not the Fit-Transform. Because the Fitting is already done on the train set. So, we just have to do the transformation with the already fitted data on the train set.

# In[103]:


X_test_pca = pca_final.transform(X_test)


# #### Emphasize Sensitivity/Recall than Accuracy
# 
# We are more focused on higher Sensitivity/Recall score than the accuracy.
# 
# Beacuse we need to care more about churn cases than the not churn cases. The main goal is to reatin the customers, who have the possiblity to churn. There should not be a problem, if we consider few not churn customers as churn customers and provide them some incentives for retaining them. Hence, the sensitivity score is more important here.

# ## Logistic regression with PCA

# In[104]:


# Importing scikit logistic regression module
from sklearn.linear_model import LogisticRegression


# In[105]:


# Impoting metrics
from sklearn import metrics
from sklearn.metrics import confusion_matrix


# #### Tuning hyperparameter  C
# C is the the inverse of regularization strength in Logistic Regression. Higher values of C correspond to less regularization.

# In[106]:


# Importing libraries for cross validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[107]:


# Creating KFold object with 5 splits
folds = KFold(n_splits=5, shuffle=True, random_state=4)

# Specify params
params = {"C": [0.01, 0.1, 1, 10, 100, 1000]}

# Specifing score as recall as we are more focused on acheiving the higher sensitivity than the accuracy
model_cv = GridSearchCV(estimator = LogisticRegression(),
                        param_grid = params, 
                        scoring= 'recall', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True) 

# Fit the model
model_cv.fit(X_train_pca, y_train)


# In[108]:


# results of grid search CV
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# In[109]:


# plot of C versus train and validation scores

plt.figure(figsize=(8, 6))
plt.plot(cv_results['param_C'], cv_results['mean_test_score'])
plt.plot(cv_results['param_C'], cv_results['mean_train_score'])
plt.xlabel('C')
plt.ylabel('sensitivity')
plt.legend(['test result', 'train result'], loc='upper left')
plt.xscale('log')


# In[110]:


# Best score with best C
best_score = model_cv.best_score_
best_C = model_cv.best_params_['C']

print(" The highest test sensitivity is {0} at C = {1}".format(best_score, best_C))


# #### Logistic regression with optimal C

# In[111]:


# Instantiate the model with best C
logistic_pca = LogisticRegression(C=best_C)


# In[112]:


# Fit the model on the train set
log_pca_model = logistic_pca.fit(X_train_pca, y_train)


# ##### Prediction on the train set

# In[113]:


# Predictions on the train set
y_train_pred = log_pca_model.predict(X_train_pca)


# In[114]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[115]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[116]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ##### Prediction on the test set

# In[117]:


# Prediction on the test set
y_test_pred = log_pca_model.predict(X_test_pca)


# In[118]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[119]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[120]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.86
#     - Sensitivity = 0.89
#     - Specificity = 0.83
# - Test set
#     - Accuracy = 0.83
#     - Sensitivity = 0.81
#     - Specificity = 0.83
#     
# Overall, the model is performing well in the test set, what it had learnt from the train set.

# ## Support Vector Machine(SVM) with PCA

# In[121]:


# Importing SVC
from sklearn.svm import SVC


# #### Hyperparameter tuning
# 
# C:- Regularization parameter.
# 
# gamma:- Handles non linear classifications.

# In[ ]:


# specify range of hyperparameters

hyper_params = [ {'gamma': [1e-2, 1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]}]


# specify model with RBF kernel
model = SVC(kernel="rbf")

# set up GridSearchCV()
model_cv = GridSearchCV(estimator = model, 
                        param_grid = hyper_params, 
                        scoring= 'accuracy', 
                        cv = 3, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train_pca, y_train)                  


# In[ ]:


# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results


# ##### Plotting the accuracy with various C and gamma values

# In[ ]:


# converting C to numeric type for plotting on x-axis
cv_results['param_C'] = cv_results['param_C'].astype('int')

# # plotting
plt.figure(figsize=(16,6))

# subplot 1/3
plt.subplot(131)
gamma_01 = cv_results[cv_results['param_gamma']==0.01]

plt.plot(gamma_01["param_C"], gamma_01["mean_test_score"])
plt.plot(gamma_01["param_C"], gamma_01["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.01")
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')

# subplot 2/3
plt.subplot(132)
gamma_001 = cv_results[cv_results['param_gamma']==0.001]

plt.plot(gamma_001["param_C"], gamma_001["mean_test_score"])
plt.plot(gamma_001["param_C"], gamma_001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.001")
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')


# subplot 3/3
plt.subplot(133)
gamma_0001 = cv_results[cv_results['param_gamma']==0.0001]

plt.plot(gamma_0001["param_C"], gamma_0001["mean_test_score"])
plt.plot(gamma_0001["param_C"], gamma_0001["mean_train_score"])
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title("Gamma=0.0001")
plt.ylim([0.80, 1])
plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
plt.xscale('log')



# In[ ]:


# Printing the best score 
best_score = model_cv.best_score_
best_hyperparams = model_cv.best_params_

print("The best test score is {0} corresponding to hyperparameters {1}".format(best_score, best_hyperparams))


# From the above plot, we can see that higher value of gamma leads to overfitting the model. With the lowest value of gamma (0.0001) we have train and test accuracy almost same.
# 
# Also, at C=100 we have a good accuracy and the train and test scores are comparable.
# 
# Though sklearn suggests the optimal scores mentioned above (gamma=0.01, C=1000), one could argue that it is better to choose a simpler, more non-linear model with gamma=0.0001. This is because the optimal values mentioned here are calculated based on the average test accuracy (but not considering subjective parameters such as model complexity).
# 
# We can achieve comparable average test accuracy (~90%) with gamma=0.0001 as well, though we'll have to increase the cost C for that. So to achieve high accuracy, there's a tradeoff between:
# - High gamma (i.e. high non-linearity) and average value of C
# - Low gamma (i.e. less non-linearity) and high value of C
# 
# We argue that the model will be simpler if it has as less non-linearity as possible, so we choose gamma=0.0001 and a high C=100.

# ##### Build the model with optimal hyperparameters

# In[ ]:


# Building the model with optimal hyperparameters
svm_pca_model = SVC(C=100, gamma=0.0001, kernel="rbf")

svm_pca_model.fit(X_train_pca, y_train)


# ##### Prediction on the train set

# In[ ]:


# Predictions on the train set
y_train_pred = svm_pca_model.predict(X_train_pca)


# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ##### Prediction on the test set

# In[ ]:


# Prediction on the test set
y_test_pred = svm_pca_model.predict(X_test_pca)


# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.89
#     - Sensitivity = 0.92
#     - Specificity = 0.85
# - Test set
#     - Accuracy = 0.85
#     - Sensitivity = 0.81
#     - Specificity = 0.85

# ## Decision tree with PCA

# In[ ]:


# Importing decision tree classifier
from sklearn.tree import DecisionTreeClassifier


# ##### Hyperparameter tuning

# In[ ]:


# Create the parameter grid 
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
}


# Instantiate the grid search model
dtree = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator = dtree, 
                           param_grid = param_grid, 
                           scoring= 'recall',
                           cv = 5, 
                           verbose = 1)

# Fit the grid search to the data
grid_search.fit(X_train_pca,y_train)


# In[ ]:


# cv results
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results


# In[ ]:


# Printing the optimal sensitivity score and hyperparameters
print("Best sensitivity:-", grid_search.best_score_)
print(grid_search.best_estimator_)


# ##### Model with optimal hyperparameters

# In[ ]:


# Model with optimal hyperparameters
dt_pca_model = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=10, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)

dt_pca_model.fit(X_train_pca, y_train)


# ##### Prediction on the train set

# In[ ]:


# Predictions on the train set
y_train_pred = dt_pca_model.predict(X_train_pca)


# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ##### Prediction on the test set

# In[ ]:


# Prediction on the test set
y_test_pred = dt_pca_model.predict(X_test_pca)


# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.90
#     - Sensitivity = 0.91
#     - Specificity = 0.88
# - Test set
#     - Accuracy = 0.86
#     - Sensitivity = 0.70
#     - Specificity = 0.87
#     
#     
# We can see from the model performance that the Sesitivity has been decreased while evaluating the model on the test set. However, the accuracy and specificity is quite good in the test set.

# ## Random forest with PCA

# In[ ]:


# Importing random forest classifier
from sklearn.ensemble import RandomForestClassifier


# ##### Hyperparameter tuning

# In[ ]:


param_grid = {
    'max_depth': range(5,10,5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'n_estimators': [100,200,300], 
    'max_features': [10, 20]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, 
                           param_grid = param_grid, 
                           cv = 3,
                           n_jobs = -1,
                           verbose = 1, 
                           return_train_score=True)

# Fit the model
grid_search.fit(X_train_pca, y_train)


# In[ ]:


# printing the optimal accuracy score and hyperparameters
print('We can get accuracy of',grid_search.best_score_,'using',grid_search.best_params_)


# ##### Model with optimal hyperparameters

# In[ ]:


# model with the best hyperparameters

rfc_model = RandomForestClassifier(bootstrap=True,
                             max_depth=5,
                             min_samples_leaf=50, 
                             min_samples_split=100,
                             max_features=20,
                             n_estimators=300)


# In[ ]:


# Fit the model
rfc_model.fit(X_train_pca, y_train)


# ##### Prediction on the train set

# In[ ]:


# Predictions on the train set
y_train_pred = rfc_model.predict(X_train_pca)


# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_train, y_train_pred)
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train, y_train_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ##### Prediction on the test set

# In[ ]:


# Prediction on the test set
y_test_pred = rfc_model.predict(X_test_pca)


# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test, y_test_pred)
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test, y_test_pred))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.84
#     - Sensitivity = 0.88
#     - Specificity = 0.80
# - Test set
#     - Accuracy = 0.80
#     - Sensitivity = 0.75
#     - Specificity = 0.80
#     
#     
# We can see from the model performance that the Sesitivity has been decreased while evaluating the model on the test set. However, the accuracy and specificity is quite good in the test set.

# ### Final conclusion with PCA
# After trying several models we can see that for acheiving the best sensitivity, which was our ultimate goal, the classic Logistic regression or the SVM models preforms well. For both the models the sensitivity was approx 81%. Also we have good accuracy of apporx 85%.

# # Without PCA

# ## Logistic regression with No PCA

# In[ ]:


##### Importing stats model
import statsmodels.api as sm


# In[ ]:


# Instantiate the model
# Adding the constant to X_train
log_no_pca = sm.GLM(y_train,(sm.add_constant(X_train)), family=sm.families.Binomial())


# In[ ]:


# Fit the model
log_no_pca = log_no_pca.fit().summary()


# In[ ]:


# Summary
log_no_pca


# ***Model analysis***
# 1. We can see that there are few features have positive coefficients and few have negative.
# 2. Many features have higher p-values and hence became insignificant in the model.
# 
# ***Coarse tuning (Auto+Manual)***
# 
# We'll first eliminate a few features using Recursive Feature Elimination (RFE), and once we have reached a small set of variables to work with, we can then use manual feature elimination (i.e. manually eliminating features based on observing the p-values and VIFs).

# ### Feature Selection Using RFE

# In[ ]:


# Importing logistic regression from sklearn
from sklearn.linear_model import LogisticRegression
# Intantiate the logistic regression
logreg = LogisticRegression()


# #### RFE with 15 columns

# In[ ]:


# Importing RFE
from sklearn.feature_selection import RFE

# Intantiate RFE with 15 columns
rfe = RFE(logreg, 15)

# Fit the rfe model with train set
rfe = rfe.fit(X_train, y_train)


# In[ ]:


# RFE selected columns
rfe_cols = X_train.columns[rfe.support_]
print(rfe_cols)


# ### Model-1 with RFE selected columns

# In[ ]:


# Adding constant to X_train
X_train_sm_1 = sm.add_constant(X_train[rfe_cols])

#Instantiate the model
log_no_pca_1 = sm.GLM(y_train, X_train_sm_1, family=sm.families.Binomial())

# Fit the model
log_no_pca_1 = log_no_pca_1.fit()

log_no_pca_1.summary()


# #### Checking VIFs

# In[ ]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[rfe_cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[rfe_cols].values, i) for i in range(X_train[rfe_cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ##### Removing column og_others_8, which is insignificatnt as it has the highest p-value 0.99

# In[ ]:


# Removing og_others_8 column 
log_cols = rfe_cols.to_list()
log_cols.remove('og_others_8')
print(log_cols)


# ### Model-2
# Building the model after removing og_others_8 variable.

# In[ ]:


# Adding constant to X_train
X_train_sm_2 = sm.add_constant(X_train[log_cols])

#Instantiate the model
log_no_pca_2 = sm.GLM(y_train, X_train_sm_2, family=sm.families.Binomial())

# Fit the model
log_no_pca_2 = log_no_pca_2.fit()

log_no_pca_2.summary()


# #### Checking VIF for Model-2

# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[log_cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[log_cols].values, i) for i in range(X_train[log_cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# As we can see from the model summary that all the variables p-values are significant and offnet_mou_8 column has the highest VIF 7.45. Hence, deleting offnet_mou_8 column.

# In[ ]:


# Removing offnet_mou_8 column
log_cols.remove('offnet_mou_8')


# ### Model-3
# Model after removing offnet_mou_8 column.

# In[ ]:


# Adding constant to X_train
X_train_sm_3 = sm.add_constant(X_train[log_cols])

#Instantiate the model
log_no_pca_3 = sm.GLM(y_train, X_train_sm_3, family=sm.families.Binomial())

# Fit the model
log_no_pca_3 = log_no_pca_3.fit()

log_no_pca_3.summary()


# #### VIF Model-3

# In[ ]:


vif = pd.DataFrame()
vif['Features'] = X_train[log_cols].columns
vif['VIF'] = [variance_inflation_factor(X_train[log_cols].values, i) for i in range(X_train[log_cols].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Now from the model summary and the VIF list we can see that all the variables are significant and there is no multicollinearity among the variables.
# 
# Hence, we can conclused that ***Model-3 log_no_pca_3 will be the final model***.

# ###  Model performance on the train set

# In[ ]:


# Getting the predicted value on the train set
y_train_pred_no_pca = log_no_pca_3.predict(X_train_sm_3)
y_train_pred_no_pca.head()


# ##### Creating a dataframe with the actual churn and the predicted probabilities

# In[ ]:


y_train_pred_final = pd.DataFrame({'churn':y_train.values, 'churn_prob':y_train_pred_no_pca.values})

#Assigning Customer ID for each record for better readblity
#CustID is the index of each record.
y_train_pred_final['CustID'] = y_train_pred_final.index

y_train_pred_final.head()


# ##### Finding Optimal Probablity Cutoff Point

# In[ ]:


# Creating columns for different probablity cutoffs
prob_cutoff = [float(p/10) for p in range(10)]

for i in prob_cutoff:
    y_train_pred_final[i] = y_train_pred_final['churn_prob'].map(lambda x : 1 if x > i else 0)
    
y_train_pred_final.head()


# ##### Now let's calculate the accuracy sensitivity and specificity for various probability cutoffs.

# In[ ]:


# Creating a dataframe
cutoff_df = pd.DataFrame(columns=['probability', 'accuracy', 'sensitivity', 'specificity'])

for i in prob_cutoff:
    cm1 = metrics.confusion_matrix(y_train_pred_final['churn'], y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)
    


# In[ ]:


# Plotting accuracy, sensitivity and specificity for different probabilities.
cutoff_df.plot('probability', ['accuracy','sensitivity','specificity'])
plt.show()


# ##### Analysis of the above curve
# Accuracy - Becomes stable around 0.6
# 
# Sensitivity - Decreases with the increased probablity.
# 
# Specificity - Increases with the increasing probablity.
# 
# `At point 0.6` where the three parameters cut each other, we can see that there is a balance bethween sensitivity and specificity with a good accuracy.
# 
# Here we are intended to acheive better sensitivity than accuracy and specificity. Though as per the above curve, we should take 0.6 as the optimum probability cutoff, we are taking ***0.5*** for acheiving higher sensitivity, which is our main goal.

# In[ ]:


# Creating a column with name "predicted", which is the predicted value for 0.5 cutoff 
y_train_pred_final['predicted'] = y_train_pred_final['churn_prob'].map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# ##### Metrics

# In[ ]:


# Confusion metrics
confusion = metrics.confusion_matrix(y_train_pred_final['churn'], y_train_pred_final['predicted'])
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_train_pred_final['churn'], y_train_pred_final['predicted']))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# We have got good accuracy, sensitivity and specificity on the train set prediction.

# ##### Plotting the ROC Curve (Trade off between sensitivity & specificity)

# In[ ]:


# ROC Curve function

def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[ ]:


draw_roc(y_train_pred_final['churn'], y_train_pred_final['churn_prob'])


# We can see the area of the ROC curve is closer to 1, whic is the Gini of the model.

# ### Testing the model on the test set

# In[ ]:


# Taking a copy of the test set
X_test_log = X_test.copy()


# In[ ]:


# Taking only the columns, which are selected in the train set after removing insignificant and multicollinear variables
X_test_log = X_test_log[log_cols]


# In[ ]:


# Adding constant on the test set
X_test_sm = sm.add_constant(X_test_log)


# ##### Predictions on the test set with final model

# In[ ]:


# Predict on the test set
y_test_pred = log_no_pca_3.predict(X_test_sm)


# In[ ]:


y_test_pred.head()


# In[ ]:


# Converting y_test_pred to a dataframe because y_test_pred is an array
y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1.head()


# In[ ]:


# Convetting y_test to a dataframe
y_test_df = pd.DataFrame(y_test)
y_test_df.head()


# In[ ]:


# Putting index to Customer ID 
y_test_df['CustID'] = y_test_df.index


# In[ ]:


# Removing index form the both dataframes for merging them side by side
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[ ]:


# Appending y_pred_1 and y_test_df
y_test_pred_final = pd.concat([y_test_df, y_pred_1], axis=1)


# In[ ]:


y_test_pred_final.head()


# In[ ]:


# Renaming the '0' column as churn probablity
y_test_pred_final = y_test_pred_final.rename(columns={0:'churn_prob'})


# In[ ]:


# Rearranging the columns
y_test_pred_final = y_test_pred_final.reindex_axis(['CustID','churn','churn_prob'], axis=1)


# In[ ]:


y_test_pred_final.head()


# In[ ]:


# In the test set using probablity cutoff 0.5, what we got in the train set 
y_test_pred_final['test_predicted'] = y_test_pred_final['churn_prob'].map(lambda x: 1 if x > 0.5 else 0)


# In[ ]:


y_test_pred_final.head()


# ##### Metrics

# In[ ]:


# Confusion matrix
confusion = metrics.confusion_matrix(y_test_pred_final['churn'], y_test_pred_final['test_predicted'])
print(confusion)


# In[ ]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[ ]:


# Accuracy
print("Accuracy:-",metrics.accuracy_score(y_test_pred_final['churn'], y_test_pred_final['test_predicted']))

# Sensitivity
print("Sensitivity:-",TP / float(TP+FN))

# Specificity
print("Specificity:-", TN / float(TN+FP))


# ***Model summary***
# 
# - Train set
#     - Accuracy = 0.84
#     - Sensitivity = 0.81
#     - Specificity = 0.83
# - Test set
#     - Accuracy = 0.78
#     - Sensitivity = 0.82
#     - Specificity = 0.78
#     
# Overall, the model is performing well in the test set, what it had learnt from the train set.

# #### Final conclusion with no PCA
# 
# We can see that the logistic model with no PCA has good sensitivity and accuracy, which are comparable to the models with PCA. So, we can go for the more simplistic model such as logistic regression with PCA as it expliains the important predictor variables as well as the significance of each variable. The model also hels us to identify the variables which should be act upon for making the decision of the to be churned customers. Hence, the model is more relevant in terms of explaining to the business.

# ## Business recomendation

# #### Top predictors
# 
# Below are few top variables selected in the logistic regression model.
# 
# | Variables   | Coefficients |
# |---------------------|--------------|
# |loc_ic_mou_8|-3.3287|
# |og_others_7|-2.4711|
# |ic_others_8|-1.5131|
# |isd_og_mou_8|-1.3811|
# |decrease_vbc_action|-1.3293|
# |monthly_3g_8|-1.0943|
# |std_ic_t2f_mou_8|-0.9503|
# |monthly_2g_8|-0.9279|
# |loc_ic_t2f_mou_8|-0.7102|
# |roam_og_mou_8|0.7135|
# 
# We can see most of the top variables have negative coefficients. That means, the variables are inversely correlated with the churn probablity.
# 
# E.g.:- 
# 
# If the local incoming minutes of usage (loc_ic_mou_8) is lesser in the month of August than any other month, then there is a higher chance that the customer is likely to churn.
# 
# ***Recomendations***
# 
# 1. Target the customers, whose minutes of usage of the incoming local calls and outgoing ISD calls are less in the action phase (mostly in the month of August).
# 2. Target the customers, whose outgoing others charge in July and incoming others on August are less.
# 3. Also, the customers having value based cost in the action phase increased are more likely to churn than the other customers. Hence, these customers may be a good target to provide offer.
# 4. Cutomers, whose monthly 3G recharge in August is more, are likely to be churned. 
# 5. Customers having decreasing STD incoming minutes of usage for operators T to fixed lines of T for the month of August are more likely to churn.
# 6. Cutomers decreasing monthly 2g usage for August are most probable to churn.
# 7. Customers having decreasing incoming minutes of usage for operators T to fixed lines of T for August are more likely to churn.
# 8. roam_og_mou_8 variables have positive coefficients (0.7135). That means for the customers, whose roaming outgoing minutes of usage is increasing are more likely to churn.
# 

# #### Plots of important predictors for churn and non churn customers

# In[ ]:


# Plotting loc_ic_mou_8 predictor for churn and not churn customers
fig = plt.figure(figsize=(10,6))
sns.distplot(data_churn['loc_ic_mou_8'],label='churn',hist=False)
sns.distplot(data_non_churn['loc_ic_mou_8'],label='not churn',hist=False)
plt.show()


# We can see that for the churn customers the minutes of usage for the month of August is mostly populated on the lower side than the non churn customers.

# In[ ]:


# Plotting isd_og_mou_8 predictor for churn and not churn customers
fig = plt.figure(figsize=(10,6))
sns.distplot(data_churn['isd_og_mou_8'],label='churn',hist=False)
sns.distplot(data_non_churn['isd_og_mou_8'],label='not churn',hist=False)
plt.show()


# We can see that the ISD outgoing minutes of usage for the month of August for churn customers is densed approximately to zero. On the onther hand for the non churn customers it is little more than the churn customers.

# In[ ]:


# Plotting monthly_3g_8 predictor for churn and not churn customers
fig = plt.figure(figsize=(10,6))
sns.distplot(data_churn['monthly_3g_8'],label='churn',hist=False)
sns.distplot(data_non_churn['monthly_3g_8'],label='not churn',hist=False)
plt.show()


# The number of mothly 3g data for August for the churn customers are very much populated aroud 1, whereas of non churn customers it spreaded accross various numbers.
# 
# Similarly we can plot each variables, which have higher coefficients, churn distribution.
