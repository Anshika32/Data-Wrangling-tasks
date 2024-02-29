#!/usr/bin/env python
# coding: utf-8

# In[55]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[63]:


df = pd.read_csv("C:\\Users\\Anshika Chauhan\\Downloads\\laptops.csv")


# In[64]:


df.head()


# In[65]:


print(df.info())


# In[66]:


df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)
df.head()


# # Evaluate the dataset for missing data

# In[67]:


missing_data = df.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# # Replace with mean

# In[68]:


weight_avg = df['Weight_kg'].astype('float').mean(axis=0)
df['Weight_kg'].replace(np.nan, weight_avg, inplace=True)


# In[69]:


common_screen_size = df['Screen_Size_cm'].value_counts().idxmax()
df["Screen_Size_cm"].replace(np.nan, common_screen_size, inplace=True)


# # Fixing the data types

# In[70]:


df[['Weight_kg','Screen_Size_cm']] = df[['Weight_kg','Screen_Size_cm']].astype('float')


# # Data Standardization
# 
# 1 inch = 2.54 cm
# 1 kg   = 2.205 pounds

# In[71]:


df["Weight_kg"] = df["Weight_kg"]*2.205
df.rename(columns={'Weight_kg':'Weight_pounds'}, inplace=True)

df["Screen_Size_cm"] = df["Screen_Size_cm"]/2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'}, inplace=True)


# # Data Normalization

# In[72]:


df['CPU_frequency'] = df['CPU_frequency']/df['CPU_frequency'].max()


# # Binning

# In[73]:


bins = np.linspace(min(df["Price"]), max(df["Price"]), 4)
group_names = ['Low', 'Medium', 'High']
df['Price-binned'] = pd.cut(df['Price'], bins, labels=group_names, include_lowest=True )


# In[74]:


plt.bar(group_names, df["Price-binned"].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("Price bins")


# # Indicator variables

# In[75]:


dummy_variable_1 = pd.get_dummies(df["Screen"])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-IPS_panel', 'Full HD':'Screen-Full_HD'}, inplace=True)
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "Screen" from "df"
df.drop("Screen", axis = 1, inplace=True)


# In[76]:


print(df.head())


# In[ ]:




