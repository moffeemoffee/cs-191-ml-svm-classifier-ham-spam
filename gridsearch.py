from helper import ClfSwitcher, score_summary
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

csv_path = 'processed.csv'
scores_path = 'scores.csv'

if __name__ == '__main__':
    # Read file
    df = pd.read_csv(os.path.join(csv_path), header=0, index_col=0).head(2000)

    # Split data
    targets = np.int64([0, 1])
    target_names = ['ham', 'spam']
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['is_spam'], test_size=0.5, random_state=191)

    print('Data set:')
    print('{} total'.format(df.shape[0]))
    for t, t_name in zip(targets, target_names):
        print('{} {}'.format(len(df[df['is_spam'] == t]), t_name))

    print('\nTraining set:')
    print('{} total'.format(len(X_train)))
    for t, t_name in zip(targets, target_names):
        print('{} {}'.format(sum([y == t for y in y_train]), t_name))

    print('\nTest set:')
    print('{} total'.format(len(X_test)))
    for t, t_name in zip(targets, target_names):
        print('{} {}'.format(sum([y == t for y in y_test]), t_name))
    print('')

    # Pipeline
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', ClfSwitcher()),
    ])

    params = [
        {
            'clf__estimator': [SVC()],
            'clf__estimator__C': [0.1, 1, 10, 100, 1000],
            'clf__estimator__kernel': ['rbf', 'linear'],
            'clf__estimator__gamma': ['scale', 0.1, 1, 10, 100]
        },
        {
            'clf__estimator': [LinearSVC()],
            'clf__estimator__C': [0.1, 1, 10, 100, 1000],
        },
        {
            'clf__estimator': [SVC()],
            'clf__estimator__C': [0.1, 1, 10, 100, 1000],
            'clf__estimator__kernel': ['poly'],
            'clf__estimator__degree': [2, 3, 4, 5, 6],
            'clf__estimator__gamma': ['scale', 0.1, 1, 10, 100]
        }
    ]

    gs = GridSearchCV(pipe, params, cv=3, n_jobs=-1, return_train_score=True)
    gs.fit(X_train, y_train)

    scores = score_summary(gs)
    scores.to_csv(scores_path)

    print(scores)
