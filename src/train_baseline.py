from rich.traceback import install
install()

from sklearn.datasets import load_iris

def main() -> None:
    data = load_iris()
    X = data.data
    y = data.target

    print("Dataset: Iris")
    print(f"Rows: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {set(y)}")
    print("Feature names:", list(data.feature_names))
    print("Target names:", list(data.target_names))
    print(type(X), type(y))

if __name__ == "__main__":
    main()
