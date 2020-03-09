import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

path = '/Users/elenitziaferi/PycharmProjects/udemy_ml_logistic_lda_knn/Files/House-Price.csv'
data = pd.read_csv(path)
data = data.drop(['airport','waterbody', 'bus_ter', 'n_hos_beds'], axis=1)
# X = data.drop('Sold', axis=1)
# y = data.Sold

X = StandardScaler().fit_transform(data.drop('price', axis=1))
spectral = cluster.SpectralClustering(n_clusters=4, eigen_solver= 'arpack', affinity= 'nearest_neighbors')
spectral.fit(X)
data['category'] = spectral.labels_
data_clusters = data.groupby('category').mean().sort_values('price')
data_clusters.index = ['low', 'mid_low', 'mid_high', 'high']
print(data_clusters[['price', 'Sold', 'room_num', 'age', 'resid_area', 'poor_prop', 'dist1', 'dist2', 'dist3', 'dist4' ]])
print(data.columns)

clusters = data.groupby('category'). price.agg(['mean', 'median', 'std']).sort_values('mean')
clusters.index = ['low', 'mid_low', 'mid_high', 'high']
print(clusters)

fig,ax = plt.subplots(figsize=(14,6))
sns.distplot(data[data.category ==0].price, hist=False, rug=True, color='purple')
sns.distplot(data[data.category ==1].price, hist=False, rug=True, color='blue')
sns.distplot(data[data.category ==2].price, hist=False, rug=True, color='red')
sns.distplot(data[data.category ==3].price, hist=False, rug=True, color='green')
ax.set_title("Distribution of clusters on Price")
plt.show()
