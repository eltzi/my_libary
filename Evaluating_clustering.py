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
from sklearn import cluster, metrics
from itertools import combinations


path = '/Users/elenitziaferi/PycharmProjects/udemy_ml_logistic_lda_knn/Files/House-Price.csv'
data = pd.read_csv(path)
data = data.drop(['airport','waterbody', 'bus_ter', 'n_hos_beds'], axis=1)
# X = data.drop('Sold', axis=1)
# y = data.Sold

X = StandardScaler().fit_transform(data)

models = [cluster.SpectralClustering(n_clusters=4),
          cluster.KMeans(n_clusters=4),
          cluster.MiniBatchKMeans(n_clusters=4),
          cluster.AgglomerativeClustering(n_clusters=4)]


for model in models:
    model.fit(X)
    print(model.__class__.__name__)
    print("\tFirst 5 labels", model.labels_[:5])
    print("\t", len(model.labels_))

for clust1, clust2 in combinations(models, 2):
    print(clust1.__class__.__name__, "versus", clust2.__class__.__name__,)
    print("\tRand score: ", metrics.adjusted_rand_score(clust1.labels_, clust2.labels_))
    print("\tMutual info: ", metrics.adjusted_mutual_info_score(clust1.labels_, clust2.labels_))

















