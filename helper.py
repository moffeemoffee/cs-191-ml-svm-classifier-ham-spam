from sklearn.base import BaseEstimator
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd


# From https://stackoverflow.com/a/53926103/3256255
class ClfSwitcher(BaseEstimator):

    def __init__(self, estimator=SGDClassifier()):
        """
        A Custom BaseEstimator that can switch between classifiers.
        :param estimator: sklearn object - The classifier
        """
        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)


# Modified from http://www.davidsbatista.net/blog/2018/02/23/model_optimization/
def score_summary(gs, sort_by='mean_score'):
    def row(scores, params):
        d = {
            'min_score': min(scores),
            'max_score': max(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
        }
        return pd.Series({**params, **d})

    rows = []
    params = gs.cv_results_['params']
    scores = []
    for i in range(gs.cv):
        key = "split{}_test_score".format(i)
        r = gs.cv_results_[key]
        scores.append(r.reshape(len(params), 1))

    all_scores = np.hstack(scores)
    for p, s in zip(params, all_scores):
        rows.append((row(s, p)))

    df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

    columns = ['min_score', 'mean_score', 'max_score', 'std_score']
    columns = columns + [c for c in df.columns if c not in columns]

    return df[columns]
