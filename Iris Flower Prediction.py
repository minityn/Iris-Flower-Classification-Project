#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df =pd.read_csv("/Users/nitinarora/Downloads/archive-3/iris.csv")


# In[4]:


df.head(10)


# In[5]:


from dataprep.eda import plot, plot_correlation, plot_missing


# In[8]:


from dataprep.eda import create_report


# In[9]:


create_report(df)


# In[10]:


df.describe()


# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[12]:


X = df.drop(columns=['Id','Species'])
y = df['Species']


# In[13]:


y = le.fit_transform(y)


# In[14]:


import seaborn as sns
sns.distplot(df['SepalLengthCm'])


# In[15]:


sns.distplot(df['SepalWidthCm'])


# In[16]:


sns.distplot(df['PetalLengthCm'])


# In[17]:


sns.distplot(df['PetalWidthCm'])


# In[18]:


from sklearn.model_selection import train_test_split, cross_val_score
# classify column
def classify(model):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model.fit(x_train, y_train)
    print('Accuracy:', model.score(x_test, y_test))
    
    score = cross_val_score(model, X, y, cv=5)
    print('CV Score:', np.mean(score))


# In[19]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
classify(model)


# In[20]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
classify(model)


# In[21]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
classify(model)


# In[22]:


from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
classify(model)


# In[23]:


from xgboost import XGBClassifier
model = XGBClassifier()
classify(model)


# In[ ]:




