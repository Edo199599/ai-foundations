from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ("model", LogisticRegression(max_iter=1000))
])

#print(pipeline.get_params().keys())

param_grid = {
    "model__C": [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3
)

#print(grid)

grid.fit(X_train, y_train)

#print("Best params:", grid.best_params_)
#print("Best CV score:", grid.best_score_)

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("Test accuracy:", test_accuracy)
print("Best estimator:", best_model)