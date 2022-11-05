#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline


# In[2]:


# importing the data from the given csv file

df = pd.read_csv('car_data.csv')
del df['User ID']


# In[3]:


df.head()


# In[4]:


df.Gender.unique()


# In[5]:


sns.violinplot(x="Gender", y="Age", data=df)


# In[6]:


df.loc[df["Gender"] == "Male","Gender"]= 1
df.loc[df["Gender"] == "Female","Gender"]= -1


# In[7]:


sns.displot(df,x="AnnualSalary",kde=True)


# In[8]:


sns.displot(df,x="Age",kde=True, color='skyblue')


# In[9]:


X = df.values[:,0:3]
y = df.values[:,3]


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 10)


# In[11]:


sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# In[ ]:



pca = PCA()
X_train = pca.fit_transform(x_train)
X_test = pca.transform(x_test)


# In[ ]:


explained_variance = pca.explained_variance_ratio_


# In[ ]:


explained_variance


# In[ ]:



pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')


# In[14]:


reg = LogisticRegression(random_state=0).fit(X_train, y_train)
reg.score(X_train, y_train)

# Predicting the Test set results
y_pred = reg.predict(X_test)
cm = confusion_matrix(y_test, y_test)
print(accuracy_score(y_test, y_test))
sns.heatmap(cm, annot=True, fmt='d',linewidths=1.5, cmap="YlGnBu")


# In[15]:



model_name = 'kernel SVM Classifier'
svmClassifier = SVC(kernel='rbf', gamma='auto')
svm_model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
svm_model.fit(X_train,y_train)

# Predicting the Test set results
y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_test)
print(accuracy_score(y_test, y_test))
sns.heatmap(cm, annot=True, fmt='d',linewidths=1.5, cmap="YlGnBu")


# In[ ]:




classifier = RandomForestClassifier(max_depth=1, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_test)
print(accuracy_score(y_test, y_test))
sns.heatmap(cm, annot=True, fmt='d',linewidths=1.5, cmap="YlGnBu")

