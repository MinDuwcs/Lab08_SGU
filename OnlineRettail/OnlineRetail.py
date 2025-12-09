import pandas as pd
import numpy as np
import sklearn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings
df = pd.read_csv("OnlineRetail.csv",encoding='ISO-8859-1')
df
df.isnull().sum()
df['Description'] = df['Description'].fillna(0)
df['CustomerID'] = df['CustomerID'].fillna(0)
df.isnull().sum()
df.value_counts

df.groupby('CustomerID')['Quantity'].sum()

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
x = df[['Quantity','UnitPrice']]

x = x.dropna()

from sklearn.preprocessing import StandardScaler
x_scaled = StandardScaler().fit_transform(x)

nertia = []
k_range = range(1, 11)  

for k in k_range:
    kmeans = KMeans(n_clusters=k,init='k-means++',random_state=42)
    kmeans.fit(x_scaled)  # Fit K-Means model on the scaled data
    nertia.append(kmeans.inertia_)  # Inertia (sum of squared distances to centroids)

# Plot the inertia for each k
plt.plot(k_range, nertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

k = 4
kmeans = KMeans(n_clusters=k, random_state=42,init='k-means++')
kmeans.fit(x_scaled)
df['cluster'] = kmeans.labels_
df[['Quantity', 'UnitPrice', 'cluster']].head()

plt.figure(figsize=(8, 6))
plt.scatter(df['Quantity'], df['UnitPrice'], c=df['cluster'], cmap='viridis', alpha=0.6)
plt.title('K-Means Clusters: Quantity vs UnitPrice')
plt.xlabel('Quantity')
plt.ylabel('UnitPrice')
plt.colorbar(label='Cluster')  