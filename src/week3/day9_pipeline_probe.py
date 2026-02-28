import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# 1) Dataset volutamente "sbilanciato":
#    le prime righe hanno valori piccoli, le ultime valori grandi (così il test può finire più "alto")
rng = np.random.default_rng(0)
X = np.concatenate([
    rng.normal(loc=0.0, scale=1.0, size=(80, 2)),   # "passato"
    rng.normal(loc=5.0, scale=1.0, size=(20, 2)),   # "futuro" (diverso)
])
y = np.array([0]*80 + [1]*20)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("Train feature means (raw):", X_train.mean(axis=0))
print("Test  feature means (raw):", X_test.mean(axis=0))

# 2) Pipeline = scaler + modello
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000)),
])

# 3) Fit sul train: qui lo scaler impara mean/std SOLO dal train
pipe.fit(X_train, y_train)

scaler = pipe.named_steps["scaler"] # restituisce lo step del processo chiamato scaler
# similmente a: scaler = pipe.steps[0][1] -- steps[0] restituirebbe la prima tupla della lista di cui vogliamo l'elemento 0, lo scaler vero
print("\nScaler learned on TRAIN:")
print("  mean_ :", scaler.mean_) # verifichiamo che questa mean è uguale alla mean di X_train (no leakage)
print("  scale_:", scaler.scale_) # questa è la standard deviation

# 4) Ora trasformiamo train e test e guardiamo le loro medie DOPO scaling
X_train_scaled = scaler.transform(X_train) # questo passaggio sarebbe già incluso nel pipe.fit(X_train, y_train)
X_test_scaled = scaler.transform(X_test) # questo invece sarebbe il primo step di pipe.predict(X_test)

print("\nAfter scaling with TRAIN parameters:")
print("  mean(train_scaled):", X_train_scaled.mean(axis=0))
print("  mean(test_scaled) :", X_test_scaled.mean(axis=0))

# pipe.fit(X_train, y_train) è la somma di: 
# scaler(X_train) 
# X_train_s = scaler.transform(X_train) 
# model.fit(X_train_s, y_train)

# pipe.predict(X_test) è invece la somma di: 
# X_test_s = scaler.transform(X_test) 
# y_pred = model.predict(X_test_s)