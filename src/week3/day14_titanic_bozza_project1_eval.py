from day14_titanic_bozza_project1_train import load_data, build_training_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier


def evaluate_predictions(y_true, y_pred, model_name):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0)
    }

    metric_names = ["accuracy", "precision", "recall", "f1"]
    for metric in metric_names:
        print(f"{model_name} {metric}: {metrics[metric]:.3f}")

    return metrics


def evaluate_dummy_baseline(X_train, X_test, y_train, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred_dummy = dummy.predict(X_test)
    dummy_metrics = evaluate_predictions(y_test, y_pred_dummy, "Dummy")
    # ho dovuto inserire la zero division perché il modello non prevede nessun positivo essendo la classe negativa la most_freq 
    # avremmo precisione sempre indefinita
    return dummy_metrics


def compare_models(pipeline_metrics, dummy_metrics):
    print("Comparison against dummy baseline:")
    print("-" * 32)

    metric_names = ["accuracy", "precision", "recall", "f1"]
    for metric in metric_names:
        print(f"{metric.capitalize()}: {pipeline_metrics[metric]:.3f} vs {dummy_metrics[metric]:.3f}")
    print()

    if pipeline_metrics["accuracy"] > dummy_metrics ["accuracy"] and pipeline_metrics["f1"] > dummy_metrics["f1"]:
        print("Accuracy e f1 della Pipeline superano le stesse della dummy baseline")
    else:
        print("Accuracy e f1 della Pipeline non superano le stesse della dummy baseline")



def main():
    X_train, X_test, y_train, y_test = load_data()

    pipe = build_training_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    pipeline_metrics = evaluate_predictions(y_test, y_pred, "Pipeline")
    print()
    dummy_metrics = evaluate_dummy_baseline(X_train, X_test, y_train, y_test)
    print()

    compare_models(pipeline_metrics, dummy_metrics)



if __name__ == "__main__":
    main()