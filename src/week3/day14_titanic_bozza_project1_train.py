from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

CATEGORICAL_COLS = ["sex", "pclass"]
NUMERICAL_COLS = ["age"]
TARGET_COL = "survived"
FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS

def load_data():
    titanic = fetch_openml(name="titanic", version=1, as_frame=True)
    df = titanic.frame.copy()
    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    df = df.dropna()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test


def build_training_pipeline():

    preprocessing = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERICAL_COLS)
        ]
    )
    pipe = Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )
    return pipe


def main():
    X_train, _, y_train, _ = load_data()
    pipe = build_training_pipeline()
    pipe.fit(X_train, y_train)
    return pipe


if __name__ == "__main__":
    main()
