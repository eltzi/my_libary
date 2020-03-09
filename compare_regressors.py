import pandas as pd
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from time import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from xgboost import XGBRegressor
from sklearn.linear_model import Lasso, Ridge


def compare_regressors(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    regressors = [
        LinearRegression(),
        Ridge(),
        Lasso(),
        KNeighborsRegressor(),
        KNeighborsRegressor(n_neighbors=5, metric='manhattan'),
        SVR(),
        LinearSVR(),
        DecisionTreeRegressor(random_state=42, max_depth=5),
        DecisionTreeRegressor(random_state=42, max_depth=10),
        RandomForestRegressor(random_state=42, max_depth=10),
        GradientBoostingRegressor(n_estimators=200),
        GaussianProcessRegressor(),
        SVR(kernel='linear'),
        XGBRegressor()
    ]

    algorithms=14
    for model in regressors[:algorithms]:
        train_start = time()
        model.fit(X_train, y_train)
        train_duration = time() - train_start
        prediction_start= time()
        predictions = model.predict(X_test)
        prediction_duration = time() - prediction_start
        print(model)
        print("\tTraining time: %0.3fs" % train_duration)
        print("\tPrediction time: %0.3fs" % prediction_duration)
        print("\tExplained variance:", explained_variance_score(y_test, predictions))
        print("\tMean Absolute Error:", mean_absolute_error(y_test, predictions))
        print("\tR2 score:", r2_score(y_test, predictions))
        print("--------------------------------------------")