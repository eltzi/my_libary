from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from time import time
nfast = 11


def compare_classiifiers(X, y):
    classifiers = {
        "KNN(3)"       : KNeighborsClassifier(5),
        "RBF SVM"      : SVC(gamma=2, C=1),
        "Decision Tree": DecisionTreeClassifier(max_depth=7),
        "Random Forest": RandomForestClassifier(max_depth=7, n_estimators=10, max_features=4),
        "AdaBoost"     : AdaBoostClassifier(),
        "Naive Bayes"  : GaussianNB(),
        "QDA"          : QuadraticDiscriminantAnalysis(),
        "Linear SVC"   : LinearSVC(),
        "Linear SVM"   : SVC(kernel="linear"),
        "Gaussian Proc": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "XGB"          : XGBClassifier()
    }

    algos = list(classifiers.items())[:nfast]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    for algo_name, classifier in algos:
        start = time()
        classifier.fit(X_train, y_train)
        train_duration = time() - start
        score = classifier.score(X_test, y_test)
        score_duration = time() - start
        print("{:<15} | score = {:.3f} | time = {:,.3f}s/{:,.3f}s".format(algo_name, score, train_duration, score_duration))



