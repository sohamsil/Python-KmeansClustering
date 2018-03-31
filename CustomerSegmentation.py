
# coding: utf-8

# In[2]:


import pandas as pd


# In[4]:


df_offers = pd.read_excel("/Users/HighOnOctave/Downloads/WineKMC.xlsx", sheet_name=0)


# In[5]:


df_offers.head()


# In[6]:


df_offers.columns = ["offer_id","campaign","varietal","min_qty","discount","origin","past_peak"]


# In[7]:


df_transactions = pd.read_excel("/Users/HighOnOctave/Downloads/WineKMC.xlsx", sheet_name=1)


# In[8]:


df_transactions.head()


# In[9]:


df_transactions.columns = ["customer_name", "offer_id"]


# In[10]:


df_transactions['n'] = 1


# In[11]:


df_transactions.head()


# In[12]:


# Join offers and transactions table
df = pd.merge(df_offers, df_transactions)


# In[13]:


#create pivot table which will give us the number of times each customer responded to the offer
matrix = df.pivot_table(index=['customer_name'],columns=['offer_id'], values='n')


# In[16]:


matrix.head()


# In[15]:


#Fill NaN values and make index a column
matrix = matrix.fillna(0).reset_index()


# In[17]:


#save a list of 0/1 column. It will be used later
x_cols = matrix.columns[1:]


# In[53]:


from sklearn.cluster import KMeans

cluster = KMeans(n_clusters = 5)

matrix['cluster'] =  cluster.fit_predict(matrix[matrix.columns[2:]])

matrix.cluster.value_counts()



# In[39]:


# Visualize as bar plot
import matplotlib.pyplot as plt
cp = matrix.cluster.value_counts().plot(kind="bar", title="Cluster Bar Plot")
cp.set_xlabel("Clusters")
cp.set_ylabel("Counts")
plt.show()


# In[64]:


# Implementing PCA to reduce the dimensionality to 2 columns

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
matrix = matrix.reset_index()

customer_clusters =  matrix[['customer_name','cluster','x','y']]
customer_clusters.head()


# In[90]:


df = pd.merge(df_transactions, customer_clusters)
df = pd.merge(df_offers, df)


# In[110]:


from ggplot import *

ggplot(df, aes(x='x', y='y', color='cluster')) +     geom_point(size=75) +     ggtitle("Customers Grouped by Cluster")


# In[117]:


cluster_centers =  pca.transform(cluster.cluster_centers_)
cluster_centers = pd.DataFrame(cluster_centers, columns=['x','y'])
cluster_centers['cluster'] = range(0,len(cluster_centers))


# In[126]:


# Plot the clusters along with their centroids
ggplot(df, aes(x='x',y='y', color='cluster')) +     geom_point(size=75) +     geom_point(cluster_centers, size=500) +     ggtitle("Customers grouped by Cluster")


# In[128]:


df['is_4'] = df.cluster==4


# In[130]:


df.groupby("is_4").varietal.value_counts()


# In[131]:


df.groupby("is_4")[['min_qty','discount']].mean()


# In[132]:


df['is_3'] = df.cluster==3


# In[133]:


df.groupby("is_3").varietal.value_counts()


# In[134]:


df.groupby("is_3")[['min_qty','discount']].mean()

