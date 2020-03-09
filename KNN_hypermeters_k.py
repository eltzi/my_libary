import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import numpy as np
from seaborn import heatmap
from sklearn.model_selection import GridSearchCV


def find_optimum_k_for_knn(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    scores = []

    for k in range(1,40):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        scores.append(score)

    scores = pd.Series(scores, index=range(1,40), name="Score")
    print(scores)

    plt.plot(range(1, 40), scores,marker='o', markerfacecolor='red', markersize=5)
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.title('Model response to number of neighbors')
    plt.savefig(os.path.join(os.path.dirname(__file__) + '/plots/knn_optimum_k.png'))
    plt.show()


def find_optimum_distance_metric(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    distance_metrics = ['minkowski', 'manhattan', 'euclidean', 'chebyshev']
    K = range(1, 40, 2)
    scores = np.empty((len(distance_metrics), len(K)))

    for x, k in enumerate(K):
        for y, metric in enumerate(distance_metrics):
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)
            score = knn.score(X_test, y_test)
            scores[y, x] = score

    heatmap(scores,
            xticklabels=list(K), yticklabels=distance_metrics)
    plt.xlabel('n_neighbors')
    plt.ylabel('Distance Metric')
    plt.title('Model response to 2 hyperparameters')
    plt.savefig(os.path.join(os.path.dirname(__file__) + '/plots/knn_distance_metric_vs_k_neighbors.png'))
    plt.show()

def knn_classifier_gridSearchCV(X, y):
    parameters = {'n_neighbors': range(1, 18, 2),
                  'weights': ['uniform', 'distance'],
                  'metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev']
                  }

    grid = GridSearchCV(KNeighborsClassifier(), parameters)
    model = grid.fit(X, y)
    print(model.best_params_, '\n')
    print(model.best_estimator_, '\n')
    print(model.best_score_)
    print((pd.DataFrame(grid.cv_results_).set_index('rank_test_score').sort_index()).T)


