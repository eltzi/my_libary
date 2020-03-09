import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes

def elbow_method(clustering_algorithm, clusters_range_lower_bound, clusters_range_upper_bound, dataset,
                 init_Cao_or_Huang_for_kprototypes = None,
                 list_categorical_features_indeces_for_kprototypes = None):

    wcss =[]

    for num_clusters in range(clusters_range_lower_bound, clusters_range_upper_bound):

        if clustering_algorithm == 'kmeans':
            kmeans = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, verbose=0, random_state=42)
            kmeans.fit(dataset)
            wcss.append(kmeans.inertia_)

        if clustering_algorithm == 'kprototypes':
            kprototypes = KPrototypes(n_clusters=num_clusters, init = init_Cao_or_Huang_for_kprototypes, n_init=20, verbose=0)
            kprototypes.fit(dataset, categorical= list_categorical_features_indeces_for_kprototypes)
            wcss.append(kprototypes.cost_)

    plt.plot(range(clusters_range_lower_bound, clusters_range_upper_bound), wcss)
    plt.title('Elbow Method for '+ str(clustering_algorithm) +' Clustering')
    plt.xlabel('No. of Cluster')
    plt.ylabel('wcss: sum of distances or clustering cost of sample to their closest cluster center')
    plt.show()





