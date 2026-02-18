import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.evaluate import sweep_thresholds, best_threshold_by_f1, print_threshold_table, best_threshold_with_min_recall

# Iris binario: versicolor vs virginica (drop setosa)
iris = load_iris()
X = iris.data
y = iris.target

mask = (y != 0)
X = X[mask]
y = y[mask]

# versicolor -> 0, virginica -> 1
y = (y == 2).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = LogisticRegression(max_iter=200, random_state=42)
clf.fit(X_train, y_train)

y_proba_pos = clf.predict_proba(X_test)[:, 1]

thresholds = np.round(np.arange(0.1, 1.0, 0.1), 1)
results = sweep_thresholds(y_test, y_proba_pos, thresholds=thresholds)

print_threshold_table(results)

best = best_threshold_by_f1(results)
print("\nBest threshold by F1:", best)

best_r09 = best_threshold_with_min_recall(results, min_recall=0.9)
print("\nBest threshold with recall>=0.90:", best_r09)