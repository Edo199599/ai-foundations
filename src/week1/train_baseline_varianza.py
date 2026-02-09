from __future__ import annotations

from statistics import mean, pstdev

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Config
# -----------------------------
TARGET_COL = "target"
MAX_ITER = 500

TEST_SIZES = [0.2, 0.4, 0.6, 0.9]
SEEDS = list(range(20))


def eval_for_test_size(test_size: float) -> tuple[float, float, float, float]:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    accs: list[float] = []

    for seed in SEEDS:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
            #stratify=y,  # commenta questa riga per vedere cosa succede senza
        )

        model = LogisticRegression(max_iter=MAX_ITER)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accs.append(accuracy_score(y_test, y_pred))

    return (mean(accs), pstdev(accs), min(accs), max(accs))


def main() -> None:
    print("test_size | mean_acc | std_acc | mean-2std | mean+2std | min | max")
    for ts in TEST_SIZES:
        media, dev_st, mn, mx = eval_for_test_size(ts)
        range_lo = media - 2 * dev_st
        range_hi = media + 2 * dev_st
        print(f"{ts:7.1f} | {media:8.4f} | {dev_st:7.4f} | {range_lo:9.4f} | {range_hi:9.4f} | {mn:4.2f} | {mx:4.2f}")


if __name__ == "__main__":
    main()
