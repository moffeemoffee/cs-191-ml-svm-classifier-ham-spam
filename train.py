from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from tqdm import tqdm
import os
import numpy as np
import pandas as pd

csv_path = 'processed.csv'

if __name__ == '__main__':
    # Read file
    df = pd.read_csv(os.path.join(csv_path), header=0, index_col=0)

    # Split data
    targets = np.int64([0, 1])
    target_names = ['ham', 'spam']
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['is_spam'], test_size=0.2, random_state=191)

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
        ('clf', SVC(kernel='sigmoid', C=10, gamma=1))
        # ('clf', LinearSVC(C=100))
        # ('clf', LinearSVC(C=10))
        # ('clf', SVC(kernel='rbf', C=100, gamma='auto'))
        # ('clf', SVC(kernel='linear', C=10, gamma=10))
    ])
    pipe.fit(X_train, y_train)

    predicted = pipe.predict(X_test)
    print('Accuracy: {:.05%}'.format(np.mean(predicted == y_test)))
    print(metrics.classification_report(y_test,
                                        predicted,
                                        target_names=target_names))
