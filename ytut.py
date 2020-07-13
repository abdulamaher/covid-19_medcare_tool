#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn import datasets


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = 7, 4


# In[3]:


iris=pd.read_csv("Desktop/virtual-medicine/iris.csv",names=["sepal_length","sepal_width","petal_length","petal_width","class"])


# In[4]:


iris.head(15)


# In[5]:


plt.scatter(iris['sepal_length'],iris['sepal_width'])
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Scatter plot on Iris dataset')


# In[6]:


type(iris)


# In[7]:


sns.set_style("whitegrid")
sns.FacetGrid(iris, hue="class", size=4)    .map(plt.scatter, "sepal_length", "sepal_width")    .add_legend()
plt.show()


# In[8]:


sns.set_style("whitegrid");
sns.pairplot(iris, hue="class", size=3);
plt.show()


# In[22]:


sns.boxplot(y='sepal_length',data=iris)


# In[32]:


sns.violinplot(y='petal_width',data=iris)


# In[35]:


sns.FacetGrid(iris,hue='class').map(sns.distplot,'sepal_width').add_legend()


# In[ ]:




