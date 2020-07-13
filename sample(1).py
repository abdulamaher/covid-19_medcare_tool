#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import datasets
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


wine = pd.read_csv("winequality-red.csv")


# In[3]:


wine.head()


# In[4]:


wine.info()


# In[5]:


wine.isnull().sum()


# In[6]:


bins=(2,6.5,8)
group_names=['bad','good']
wine['quality'] = pd.cut(wine['quality'], bins=bins, labels=group_names)
wine['quality'].unique()


# In[7]:


label_quality = LabelEncoder()


# In[8]:


#basically 0 is first group name and 1 is second 
wine['quality'] = label_quality.fit_transform(wine['quality'])


# In[9]:


wine['quality'].value_counts()


# In[10]:


wine.head(10)


# In[11]:


#sample plot
sns.countplot(wine['quality'])


# In[12]:


#data measurements seperate
X = wine.drop('quality',axis=1)
y = wine['quality']


# In[13]:


#train and test splitting data where 80%train 20%test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)


# In[14]:


#scalers
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# RandfCLassifier

# In[15]:


rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
pred_rfc = rfc.predict(X_test)


# In[16]:


#performance & confusion matrix[[bad-pred-right bad-pred-wrong][good-pred-wrong good-pred-right]]
print(classification_report(y_test,pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


# SVM Classifier

# In[17]:


clf=svm.SVC()
clf.fit(X_train,y_train)
pred_clf = clf.predict(X_test)


# In[18]:


print(classification_report(y_test,pred_clf))
print(confusion_matrix(y_test, pred_clf))


# Nerual-Nets

# In[19]:


mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500) #11cols compared against quant
mlpc.fit(X_train,y_train)
pred_mlpc = mlpc.predict(X_test)


# In[20]:


print(classification_report(y_test,pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))


# In[21]:


from sklearn.metrics import accuracy_score
cm = accuracy_score(y_test, pred_rfc)
cm


# In[22]:


wine.head(10)


# In[23]:


X_new=[[7.3,0.058,0,2,0.065,15,21,0.9946,3.36,0.47,10]]
X_new = sc.transform(X_new)
y_new = rfc.predict(X_new)
y_new


# Visualization

# In[28]:


#plt.scatter(x=wine.'citric acid',y=wine.quality)


# In[ ]:




