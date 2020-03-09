from time import time
from sklearn.metrics import *

class clusteringEvaluationMetrics(object):

    def __init__(self, clustering_algorithm, clusterer, dataset, num_of_clusters,
                 distance_algorithm, column_index_num_to_compute_metrics = 1,
                 categorical_features_index_nums = None):

        self.clustering_algorithm = clustering_algorithm
        self.clusterer = clusterer
        self.dataset = dataset
        self.num_of_clusters = num_of_clusters
        self.distance_algorithm = distance_algorithm
        self.column_index_num_to_compute_metrics = column_index_num_to_compute_metrics
        self.categorical_features_index_nums = categorical_features_index_nums

        if self.clustering_algorithm == 'kmeans' and self.categorical_features_index_nums is None:
            self.get_kmeans_evaluation_metrics_for_optimum_params()
        if self.clustering_algorithm == 'kprototypes' and self.categorical_features_index_nums is not None:
            self.get_kprototypes_evaluation_metrics_for_optimum_params()


    def get_kmeans_evaluation_metrics_for_optimum_params(self):
        t0 = time()
        self.clusterer.fit_predict(self.dataset)
        print('distance metric\t\ttime\tnclusters\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tnorm_mutual\tsilhouette')
        print('%-9s\t\t\t%.2fs\t%i\t\t\t%.1f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f'
              % (self.distance_algorithm, (time() - t0), self.num_of_clusters, self.clusterer.inertia_,
                 homogeneity_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                   self.clusterer.labels_),
                 completeness_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                    self.clusterer.labels_),
                 v_measure_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                 self.clusterer.labels_),
                 adjusted_rand_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                     self.clusterer.labels_),
                 adjusted_mutual_info_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                            self.clusterer.labels_),
                 normalized_mutual_info_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                              self.clusterer.labels_),
                 silhouette_score(self.dataset, self.clusterer.labels_, metric=self.distance_algorithm)))

    def get_kprototypes_evaluation_metrics_for_optimum_params(self):
        t0 = time()
        self.clusterer.fit_predict(self.dataset, categorical=self.categorical_features_index_nums)
        print('distance metric\t\ttime\tnclusters\tcost_function\thomo\tcompl\tv-meas\tARI\tAMI\tnorm_mutual\tsilhouette')
        print('%-9s\t\t\t%.2fs\t%i\t\t\t%.1f\t\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.4f'
              % (self.distance_algorithm, (time() - t0), self.num_of_clusters, self.clusterer.cost_,
                 homogeneity_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                   self.clusterer.labels_),
                 completeness_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                    self.clusterer.labels_),
                 v_measure_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                 self.clusterer.labels_),
                 adjusted_rand_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                     self.clusterer.labels_),
                 adjusted_mutual_info_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                            self.clusterer.labels_),
                 normalized_mutual_info_score(self.dataset[:, self.column_index_num_to_compute_metrics],
                                            self.clusterer.labels_),
                 silhouette_score(self.dataset, self.clusterer.labels_, metric=self.distance_algorithm)))






