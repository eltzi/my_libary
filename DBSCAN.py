import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn import cluster

path = '/Users/elenitziaferi/PycharmProjects/udemy_ml_logistic_lda_knn/Files/House-Price.csv'
data = pd.read_csv(path)
data = data.drop(['airport','waterbody', 'bus_ter', 'n_hos_beds'], axis=1)
# X = data.drop('Sold', axis=1)
# y = data.Sold

X = StandardScaler().fit_transform(data)

for epsilon in np.arange(0.1, 1.9, 0.1):
    db = cluster.DBSCAN(eps=epsilon)
    db.fit(X)
    labels = np.unique(db.labels_)
    print("Epsilon = %.1f; labels=[%d .. %d]" % (epsilon, labels[0], labels[-1]))


data['category'] = cluster.DBSCAN(eps=1.3).fit(X).labels_
print(data.groupby('category').price.agg(['mean', 'median', 'std', 'count']).sort_values('count', ascending=False).head())

data['category'] = cluster.DBSCAN(eps=1.5, min_samples=3).fit(X).labels_
print(data.groupby('category').price.agg(['mean', 'median', 'std', 'count']).sort_values('count', ascending=False).head())

fig, ax = plt.subplots(figsize=(14,6))

for cat in range(data.category.max()+1):
    sns.distplot(data[data.category == cat].price,
                 label=str(cat), hist=False, rug=True)
ax.set_title("Distribution of clusters on PRICE")
plt.show()