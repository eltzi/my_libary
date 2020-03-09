from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

def dummy_classifiers_scores(dataset, target_column):
    X = dataset.drop(target_column, axis=1)
    y = dataset[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    for strategy in ['most_frequent', 'stratified', 'prior', 'uniform']:
        dummy = DummyClassifier(strategy=strategy)
        dummy.fit(X_train, y_train)
        dummy.predict(X_test)
        score = dummy.score(X_test, y_test)
        print("{:<15} |  score = {:.3f}".format(strategy, score))

