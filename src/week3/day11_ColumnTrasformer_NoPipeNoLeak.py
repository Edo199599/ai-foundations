import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Crea un mini dataset con:
# - una colonna categorica: city
# - una colonna numerica: age
# - una target binaria: bought

df = pd.DataFrame({
    "city": ["Venezia", "Milano", "Roma", "Roma", "Milano", "Venezia", "Venezia", "Milano", "Roma", "Venezia", "Milano", "Roma"],
    "age": [23, 67, 45, 20, 54, 47, 86, 21, 34, 52, 76, 18],
    "bought": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
})

# 2) Separa feature e target
X = df[["city", "age"]]
y = df["bought"]

# 3) Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# 4) Definisci le liste di colonne
categorical_cols = ["city"]
numeric_cols = ["age"]


# 5) Preprocessing

preprocessor = ColumnTransformer(
    transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ]
)

# FIT del preprocessor fuori dalla pipeline
X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)

# modello allenato sui dati preprocessati
fragile_model = LogisticRegression()

# fit
fragile_model.fit(X_train_prepared, y_train)

# predict
y_pred_fragile = fragile_model.predict(X_test_prepared)

print(y_pred_fragile)
print(accuracy_score(y_test, y_pred_fragile))

# Ottengo lo stesso risultato perché è come se avessi esploso la Pipeline in tutti i suoi passaggi. 
# Comunque correttamente utilizzati