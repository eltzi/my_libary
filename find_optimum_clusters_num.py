from sklearn.cluster import KMeans
from kmodes.kprototypes import  KPrototypes
from sklearn.metrics import *
import pandas as pd
from elbow_method import elbow_method


def getting_clusters_optimum_num_silhouette_or_error(clusters_range_lower_bound, clusters_range_upper_bound, dataset,
                                                     clustering_algorithm, distance_algorithm,
                                                     determine_optimum_num_clusters_based_on_silhouette_or_error ='silhouette',
                                                     print_optimum_metrics = True, init_Cao_or_Huang_for_kprototypes=None,
                                                     list_categorical_features_indeces_for_kprototypes = None):
    list_clustering_metrics =[]
    if clustering_algorithm == 'kmeans':
        list_clustering_metrics = kmeans_compute_metrics_for_every_cluster_number(clusters_range_lower_bound, clusters_range_upper_bound,
                                                                                  dataset, distance_algorithm, print_optimum_metrics)

    if clustering_algorithm == 'kprototypes':
        list_clustering_metrics = kprototypes_compute_metrics_for_every_cluster_number(clusters_range_lower_bound,
                                                                                       clusters_range_upper_bound,
                                                                                       dataset, distance_algorithm,
                                                                                       init_Cao_or_Huang_for_kprototypes,
                                                                                       list_categorical_features_indeces_for_kprototypes,
                                                                                       print_optimum_metrics)

    scores = pd.DataFrame(list_clustering_metrics)
    scores['error_diff'] = abs(scores['error'].pct_change())

    if determine_optimum_num_clusters_based_on_silhouette_or_error == 'silhouette':
        scores = scores[(scores['silhouette'] == scores['silhouette'].max())]


    if determine_optimum_num_clusters_based_on_silhouette_or_error == 'error':
        scores = scores[(scores['error_diff'] >= float(0.1)) & (scores['silhouette'] >= 0.54)]
        scores = scores[(scores['error_diff'] == scores['error_diff'].max())]

        if clustering_algorithm =='kmeans':
            elbow_method(str(clustering_algorithm), clusters_range_lower_bound, clusters_range_upper_bound, dataset)


        if clustering_algorithm =='kprototypes':
            elbow_method(str(clustering_algorithm), clusters_range_lower_bound, clusters_range_upper_bound, dataset,
                         init_Cao_or_Huang_for_kprototypes = init_Cao_or_Huang_for_kprototypes,
                         list_categorical_features_indeces_for_kprototypes = list_categorical_features_indeces_for_kprototypes)


    return int(scores['clusters'])


def kmeans_compute_metrics_for_every_cluster_number(clusters_range_lower_bound, clusters_range_upper_bound, dataset,
                                                    distance_algorithm, print_optimum_metrics = True ):

    kmeans_list_metrics = []

    for num_of_clusters in range(clusters_range_lower_bound, clusters_range_upper_bound):

        kmeans = KMeans(n_clusters= int(num_of_clusters), init='k-means++', n_init=10,  random_state=42, verbose=0)
        predictions = kmeans.fit_predict(dataset)
        centers = kmeans.cluster_centers_
        wcss = kmeans.inertia_
        num_jobs = kmeans.n_iter_
        silhouette = silhouette_score(dataset, predictions, distance_algorithm)

        kmeans_list_metrics.append({'clusters': num_of_clusters, 'silhouette': silhouette,
                                    'error': wcss, 'num_jobs': num_jobs})

        if print_optimum_metrics is True:
            print("For n_clusters = {}, silhouette score is {}, cluster_errors is {}, "
                  "n_jobs {})".format(num_of_clusters, silhouette, wcss, num_jobs))



    return kmeans_list_metrics

def kprototypes_compute_metrics_for_every_cluster_number(clusters_range_lower_bound, clusters_range_upper_bound, dataset,
                                                         distance_algorithm, init_Cao_or_Huang_for_kprototypes,
                                                         list_categorical_features_indeces_for_kprototypes,
                                                         print_optimum_metrics = True):

    kprototypes_list_metrics = []

    for num_of_clusters in range(clusters_range_lower_bound, clusters_range_upper_bound):

        kprototypes = KPrototypes(n_clusters=int(num_of_clusters), init=str(init_Cao_or_Huang_for_kprototypes),
                                  n_init=50, verbose=0)
        predictions = kprototypes.fit_predict(dataset,
                                              categorical=list_categorical_features_indeces_for_kprototypes)
        centers = kprototypes.cluster_centroids_
        cost_function = kprototypes.cost_
        num_jobs = kprototypes.n_iter_
        error_metric = cost_function
        silhouette = silhouette_score(dataset, predictions, distance_algorithm)

        kprototypes_list_metrics.append({'clusters': num_of_clusters, 'silhouette': silhouette,
                                                'error': error_metric, 'num_jobs': num_jobs})
        if print_optimum_metrics is True:
            print("For n_clusters = {}, silhouette score is {}, cluster_errors is {}, "
                  "n_jobs {})".format(num_of_clusters, silhouette, error_metric, num_jobs))

    return kprototypes_list_metrics





























