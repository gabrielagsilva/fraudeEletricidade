import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, train_test_split


def avaliacao_teste(X_test, y_test, y_pred, clf):
    print(f"Test Precision {precision_score(y_test, y_pred, zero_division=0)}")
    print(f"Test Recall {recall_score(y_test, y_pred)}")
    print(f"Test Accuracy {accuracy_score(y_test, y_pred)}")
    print(f"Test F1-measure {f1_score(y_test, y_pred)}")
    plot_confusion_matrix(clf, X_test, y_test, values_format='d')  
    plt.show()

dataset = pd.read_csv(open("data/dataset_subestacao_t.csv"), header=0)
X, y = dataset.iloc[:, 1:-1], dataset['classe_cliente']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)

# sgd = SGDClassifier(random_state=9, loss='hinge')
# param_grid = [
#     {"alpha": [0.0001, 0.001, 0.01], "average": [True, False], "penalty": ["l2", "l1"], "class_weight": [None, "balanced"]},
#     {"alpha": [0.0001, 0.001, 0.01], "average": [True, False], "penalty": ["elasticnet"], "l1_ratio": np.linspace(0, 1, num=10), "class_weight": [None, "balanced"]}
# ]
# clf = GridSearchCV(
#     estimator=sgd,
#     param_grid=param_grid,
#     scoring=('accuracy', 'precision', 'recall', 'f1'),
#     cv=10,              # validação cruzada (cross validation) com 10 folds
#     refit="precision",  # qual métrica determinará o melhor conjunto de parâmetros
#     verbose=1
# )
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# avaliacao_teste(y_test, y_pred)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
param_grid = {
    'max_features': ['auto', 'sqrt', 'log2'],
    'criterion' :['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}
clf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=('accuracy', 'precision', 'recall', 'f1'),
    cv=10,              # validação cruzada (cross validation) com 10 folds
    refit="precision",  # qual métrica determinará o melhor conjunto de parâmetros
    verbose=2
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
avaliacao_teste(X_test, y_test, y_pred, clf)

# salvar modelo
with open('data/model.classifier', 'wb') as f:
    pickle.dump(clf, f)
