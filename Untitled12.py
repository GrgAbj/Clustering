#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


# In[4]:


data = pd.read_csv('3.01.+Country+clusters.csv')
data


# In[5]:


data_mapped = data.copy()
data_mapped['Language']=data_mapped['Language'].map({'English':0,'French':1,'German':2})
data_mapped


# In[6]:


x=data_mapped.iloc[:,1:4]
x


# In[9]:


kmeans = KMeans(3)
kmeans.fit(x)


# In[10]:


identified_clusters = kmeans.fit_predict(x)
identified_clusters


# In[11]:


data_with_clusters = data_mapped.copy()
data_with_clusters['Cluster'] = identified_clusters
data_with_clusters


# In[12]:


plt.scatter(data_with_clusters['Longitude'],data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()


# ## WCSS

# In[13]:


kmeans.inertia_


# In[14]:


wcss=[]

for i in range(1,7):
    kmeans=KMeans(i)
    kmeans.fit(x)
    wcss_iter=kmeans.inertia_
    wcss.append(wcss_iter)


# In[15]:


wcss


# In[16]:


number_clusters=range(1,7)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number Of Clusters')
plt.ylabel('Within-cluster Sum Of Squares')


# In[ ]:




