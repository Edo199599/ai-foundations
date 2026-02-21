import numpy as np

# importa dal tuo modulo evaluate
from src.evaluate import evaluate_standard, print_standard_eval


def sanity_check_mcc_examples() -> None:
    # 1) Perfetto -> MCC ~ 1
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    res = evaluate_standard(y_true, y_pred)
    print_standard_eval(res, title="SANITY 1: perfect (MCC≈1)")

    # 2) Inverso (bilanciato) -> MCC ~ -1
    y_true = np.array([0, 0, 1, 1])
    y_pred = 1 - y_true
    res = evaluate_standard(y_true, y_pred)
    print_standard_eval(res, title="SANITY 2: inverse (MCC≈-1)")

    # 3) Imbalance + always-0 -> MCC ~ 0
    y_true = np.array([0] * 990 + [1] * 10)
    y_pred = np.zeros_like(y_true)
    res = evaluate_standard(y_true, y_pred)
    print_standard_eval(res, title="SANITY 3: imbalance always-0 (MCC≈0)")


if __name__ == "__main__":
    sanity_check_mcc_examples()
