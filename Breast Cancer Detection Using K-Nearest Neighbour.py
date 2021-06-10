#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("BCData.csv")
data.head()


# In[3]:


data.drop(['id','Unnamed: 32'],axis=1,inplace=True)
data.head()


# In[4]:


data.isnull().sum()


# In[5]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['diagnosis']=le.fit_transform(data['diagnosis'])
data.head(22)


# In[6]:


plt.figure(figsize=(32,32))
sns.heatmap(data.corr(),cmap='Blues',annot=True)
plt.title("Correlation Map",fontsize=16)


# In[8]:


x=data.drop(['diagnosis','texture_mean','smoothness_mean','symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se','texture_worst','smoothness_worst','compactness_worst','symmetry_worst', 'fractal_dimension_worst'],axis=1)

y=data['diagnosis']
print("x is ",x)
print("y is ",y)


# In[24]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print('x_train',x_train)
print('----------------------------------')
print('x_test',x_test)


# In[25]:


from sklearn.preprocessing import MinMaxScaler
mm=MinMaxScaler()
x_train=mm.fit_transform(x_train)
x_test=mm.fit_transform(x_test)
print('x_train',x_train)
print('----------------------------------')
print('x_test',x_test)


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 6, metric = 'manhattan')
knn.fit(x_train,y_train)


# In[36]:


#getting confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('confusion matrix:\n',cm)


# In[37]:


#checking accuracy
from sklearn.metrics import accuracy_score
knna = accuracy_score(y_test,y_pred)
print('accuracy score = ',accuracy_score(y_test,y_pred))

