import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn
import matplotlib.pyplot as plt

class pca(object):

    def __init__(self, dataset, final_variance_ratio, pca_plot_show=False):

        self.dataset = dataset
        self.final_variance_ratio = final_variance_ratio

        if pca_plot_show is True:
            self.pca_plot()
        else:
            pass

    def pca_analysis(self):


        self.standard_scaler = StandardScaler()
        self.array_of_dataset = self.standard_scaler.fit_transform(self.dataset)

        self.pca = PCA()
        self.pca.fit(self.array_of_dataset)


        self.sum_of_variance_ratio = []

        for feature_ratio_variance in self.pca.explained_variance_ratio_:
            self.sum_of_variance_ratio.append(feature_ratio_variance)
            if sum(self.sum_of_variance_ratio) >= float(self.final_variance_ratio):
                break

        self.optimum_num_pca_components = len(self.sum_of_variance_ratio)

        self.pca_optimum = PCA(n_components=self.optimum_num_pca_components)
        self.pca_optimum_num_features_array = self.pca_optimum.fit_transform(
            self.array_of_dataset[:, : self.optimum_num_pca_components])

        return self.pca_optimum_num_features_array

    def pca_plot(self):

        self.independent_variables = self.dataset
        mean_vec = np.mean(self.independent_variables, axis=0)
        self.covariance_matrix = np.cov(self.independent_variables.T)
        self.eigenvalue_tuples, self.eigenvector_tuples = np.linalg.eig(self.covariance_matrix)
        self.eig_pairs = [(np.abs(self.eigenvalue_tuples[index_number]), self.eigenvector_tuples[:, index_number]) for
                          index_number in range(len(self.eigenvalue_tuples))]

        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)

        self.sum_eigenvalues = sum(self.eigenvalue_tuples)
        self.individual_explained_variance = [(eigenvalue / self.sum_eigenvalues) * 100 for eigenvalue in
                                              sorted(self.eigenvalue_tuples, reverse=True)]
        self.cumulative_explained_variance = np.cumsum(self.individual_explained_variance)

        plt.figure(figsize=(7, 5))
        plt.bar(range(len(self.individual_explained_variance)), self.individual_explained_variance, alpha=0.3333,
                align='center', label='individual explained variance', color='g')
        plt.step(range(len(self.cumulative_explained_variance)), self.cumulative_explained_variance, where='mid',
                 label='cumulative explained variance')
        plt.ylabel('Explained variance ratio')
        plt.xlabel('Principal components')
        plt.legend(loc='best')
        plt.show()