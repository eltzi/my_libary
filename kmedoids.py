import numpy as np
from scipy.spatial.distance import cityblock
from time import time

class kmedoids(object):
    def __init__(self, num_clusters, max_iterations = 100):
        self.max_num_clusters = num_clusters
        self.max_iterations = max_iterations

    def fit(self, data):
        self.medoids = {}
        self.clusters = {}

        for i in range(self.max_num_clusters):
            self.medoids[i] = data[i]

        for iteration in range(self.max_iterations):
            for num_cluster in range(self.max_num_clusters):
                self.clusters[num_cluster] = []

            for point in data:
                distances = [cityblock(point, self.medoids[i]) for i in self.medoids]
                min_distances = np.min(distances)
                index = distances.index(min_distances)
                self.clusters[index].append(point)


            for i in self.clusters:
                distances = [[cityblock(point, every_point) for every_point in self.clusters[i]]  for point in self.clusters[i]]
                costs = list(np.sum(distances, axis=1))
                index = costs.index(np.min(costs))
                self.medoids[i] = self.clusters[i][index]

    def predict(self, data):
        self.predictions = []
        for point in data:
            distances = [cityblock(point, self.medoids[i]) for i in self.medoids]
            min_distance = np.min(distances)
            index = distances.index(min_distance)
            self.predictions.append(index)

        return  self.predictions

