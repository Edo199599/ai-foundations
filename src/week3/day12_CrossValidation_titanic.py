from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

titanic = fetch_openml(name="titanic", version=1, as_frame=True)

df = titanic.frame.copy()

df = df[["sex", "pclass", "age", "survived"]].copy()
df = df.dropna()
X = df[["sex", "pclass", "age"]]
y = df["survived"].astype(int)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
#print(X_test.shape)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


numerical_col = ["age", "pclass"]
categorical_col = ["sex"]

preprocessing = ColumnTransformer( transformers = [
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_col),
        ("num", StandardScaler(), numerical_col)
    ]
)

pipe = Pipeline (
    [
        ("preprocessing", preprocessing),
        ("model", LogisticRegression(max_iter=1000))
    ]
)

scores = cross_val_score(pipe, X_train, y_train, cv = cv, scoring = "accuracy")

print("CV accuracy scores:", scores.round(3))
print(f"Mean CV accuracy: {scores.mean():.3f}")
print(f"Std CV accuracy: {scores.std():.3f}")

# ne risulta questo array delle accuracy [0.75  0.679 0.714 0.821 0.774 0.798 0.807 0.771 0.819 0.771]
# con queste medie e std 0.770, 0.044

# il modello ha una buona accuracy totale e una std non necessariamente troppo alta. Val minimo 0.679 e max 0.821

print("-------------------------------------------------")

# Confrontiamo questa accuracy media con quella che troveremmo agendo con la pipe sullo split iniziale come solito

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
acc_test = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc_test:.3f}")
print(f"Mean CV accuracy: {scores.mean():.3f}")
print(f"Std CV accuracy: {scores.std():.3f}")

# Test Accuracy: 0.833 suggerisce che lo split è fortunato e avrebbe potuto far sovrastimare la bontà del nostro modello
# considerando che la media delle accuracy in cross validation è 0,77 con una moderata ma non alta std pari a 0.044

print("-------------------------------------------------\n")

# aggiungiamo altri valori importanti di riferimento come f1, per l'analisi della bontà del modello

f1_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
precision_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="precision")
recall_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall")

print("f1 CV:", f1_scores.round(3))
print(f"Mean CV f1: {f1_scores.mean():.3f}")
print(f"std CV f1: {f1_scores.std():.3f}\n")

print("Precision CV:", precision_scores.round(3))
print(f"Mean CV precision: {precision_scores.mean():.3f}")
print(f"Std CV precision: {precision_scores.std():.3f}\n")

print("Recall CV:", recall_scores.round(3))
print(f"Mean CV recall: {recall_scores.mean():.3f}")
print(f"Std CV recall: {recall_scores.std():.3f}")

#Con la cross-validation su pipeline abbiamo ottenuto una stima più robusta della performance rispetto 
# a uno split singolo. Il test esterno sembrava favorevole, mentre la CV ha mostrato una performance media più prudente. 
# Inoltre, guardando più metriche, abbiamo visto che il modello ha una precision discreta ma una recall più debole, 
# quindi l’accuracy da sola sarebbe stata troppo ottimistica.