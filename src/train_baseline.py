from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# -----------------------------
# Config
# -----------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "target"
MAX_ITER = 500


def main() -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("Shapes:", X_train.shape, X_test.shape, "| Train class counts:", y_train.value_counts().to_dict())

    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy (test): {acc:.4f}")

    # Confusion matrix (raw)
    cm = confusion_matrix(y_test, y_pred)
    class_names = list(map(str, iris.target_names))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print("Classes:", class_names)
    print(cm)

    # Confusion matrix normalized per riga (row-wise)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    print("\nConfusion Matrix normalized (row-wise):")
    print(np.round(cm_norm, 3))

    # Most confused pair (solo se esistono errori)
    cm_offdiag = cm.copy()
    np.fill_diagonal(cm_offdiag, 0)

    most_confused = cm_offdiag.max()
    if most_confused > 0:
        i, j = np.unravel_index(np.argmax(cm_offdiag), cm_offdiag.shape)
        print("\nMost confused pair:")
        print(f"True '{class_names[i]}' predicted as '{class_names[j]}' -> {most_confused} times")
    else:
        print("\nMost confused pair: none (no off-diagonal errors)")

    # Classification report
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=3,
        )
    )


if __name__ == "__main__":
    main()
