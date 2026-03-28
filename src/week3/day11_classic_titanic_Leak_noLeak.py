from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic = fetch_openml(name="titanic", version=1, as_frame=True)

df = titanic.frame.copy()

df = df[["sex", "age", "pclass", "survived"]].copy()
df = df.dropna()
X = df[["sex", "age", "pclass"]]
y = df["survived"]

categorical_col = ["sex"]
numerical_col = ["age", "pclass"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25, stratify=y)

preprocessing = ColumnTransformer(
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_col),
        ("num", StandardScaler(), numerical_col)
    ])

pipe = Pipeline([
    ("preprocessing", preprocessing),
    ("model", LogisticRegression())
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(accuracy_score(y_test, y_pred))


# Ora provo con il modello leaky

preprocessing.fit(X)

X_train_leak = preprocessing.transform(X_train)
X_test_leak = preprocessing.transform(X_test)

model = LogisticRegression()

model.fit(X_train_leak, y_train)
y_pred_leak = model.predict(X_test_leak)

print(accuracy_score(y_test, y_pred_leak))

# Anche in questo caso la distribuzione di X-train e X_test sono molto simili e il modello è ugualmente
# bravo anche in caso di leak a trovare la soluzione 

print(confusion_matrix(y_test, y_pred))
print(confusion_matrix(y_test, y_pred_leak))