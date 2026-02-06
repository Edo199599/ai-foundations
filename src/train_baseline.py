from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 500

def main() -> None:
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = LogisticRegression(max_iter=MAX_ITER)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"X shape: {X.shape} | y shape {y.shape}")
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"Accuracy (test): {acc:.4f}")

if __name__ == "__main__":
    main()
