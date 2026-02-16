# Day 6 — Confusion Matrix + Precision/Recall/F1 + Threshold demo
# Dataset: Iris (multi-class) + demo binaria con soglia

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

# -----------------------------
# 1) MULTI-CLASS: Iris
# -----------------------------
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf_mc = LogisticRegression(max_iter=1000)
clf_mc.fit(X_train, y_train)

y_pred = clf_mc.predict(X_test)

print("=== MULTI-CLASS (Iris) ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nConfusion Matrix (rows=true, cols=pred):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Optional: guarda anche le probabilità (per multi-class è argmax)
proba_mc = clf_mc.predict_proba(X_test)
print("\nEsempio prime 3 righe predict_proba (ogni colonna è una classe):")
print(np.round(proba_mc[:3], 3))
print("Predizioni prime 3:", y_pred[:3])


# -----------------------------
# 2) BINARY: Iris (versicolor vs virginica) per demo threshold
# -----------------------------
# Teniamo solo le classi 1 e 2 (versicolor, virginica). Togliamo la classe 0 (Setosa)
mask = (y != 0)
X2 = X[mask]
y2 = y[mask]

# Rimappo: versicolor(1)->0, virginica(2)->1
y_bin = (y2 == 2).astype(int)  # 1 = virginica, 0 = versicolor

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X2, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

clf_bin = LogisticRegression(max_iter=1000)
clf_bin.fit(X_train_b, y_train_b)

p1 = clf_bin.predict_proba(X_test_b)[:, 1]  # prob di virginica

def eval_threshold(thr: float):
    y_hat = (p1 >= thr).astype(int)
    cm = confusion_matrix(y_test_b, y_hat)  # [[TN, FP],[FN, TP]]
    print(f"\n=== BINARY (virginica vs versicolor) @ threshold={thr:.2f} ===")
    print("Confusion Matrix [[TN, FP],[FN, TP]]:")
    print(cm)
    print("Report:")
    print(classification_report(y_test_b, y_hat, digits=4))

eval_threshold(0.20)
eval_threshold(0.50)
eval_threshold(0.80)