import pandas as pd
import matplotlib.pyplot as plt
import os


def feature_importance(model, X):

    pd.Series(model.feature_importances_, index=X.columns).plot.barh(figsize=(18,7))
    plt.savefig(os.path.join(os.path.dirname(__file__) + '/plots/feature_importance.png'))
    plt.show()

