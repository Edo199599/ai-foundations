import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def make_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([
        rng.normal(loc=0.0, scale=1.0, size=(80, 2)),
        rng.normal(loc=5.0, scale=1.0, size=(20, 2)),
    ])
    y = np.array([0]*80 + [1]*20)
    return X, y

def leaky_flow(X, y):
    # ❌ Errore: fit dello scaler su TUTTO X prima dello split
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X) # esegue fit(X) e transorm(X) assieme

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    return acc, scaler.mean_, scaler.scale_

def correct_flow(X, y):
    # ✅ Split prima, poi fit scaler SOLO su train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    return acc, scaler.mean_, scaler.scale_

def pipeline_flow(X, y):
    # ✅ Stessa logica del correct_flow, ma incapsulata
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    scaler = pipe.named_steps["scaler"]
    return acc, scaler.mean_, scaler.scale_

def main():
    X, y = make_data(seed=0)

    acc_leaky, mean_leaky, scale_leaky = leaky_flow(X, y)
    acc_corr,  mean_corr,  scale_corr  = correct_flow(X, y)
    acc_pipe,  mean_pipe,  scale_pipe  = pipeline_flow(X, y)

    print("\n=== ACCURACY ===")
    print(f"LEAKY   acc: {acc_leaky:.4f}")
    print(f"CORRECT acc: {acc_corr:.4f}")
    print(f"PIPE    acc: {acc_pipe:.4f}")

    print("\n=== SCALER PARAMS (first 2 features) ===")
    print("LEAKY   mean_ :", mean_leaky)
    print("CORRECT mean_ :", mean_corr)
    print("PIPE    mean_ :", mean_pipe)

    print("\nLEAKY   scale_:", scale_leaky)
    print("CORRECT scale_:", scale_corr)
    print("PIPE    scale_:", scale_pipe)

if __name__ == "__main__":
    main()