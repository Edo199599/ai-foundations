from day14_titanic_bozza_project1_train import load_data, build_training_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold


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

    if pipeline_metrics["accuracy"] > dummy_metrics["accuracy"] and pipeline_metrics["f1"] > dummy_metrics["f1"]:
        print("Accuracy e f1 della Pipeline superano le stesse della dummy baseline")
    else:
        print("Accuracy e f1 della Pipeline non superano le stesse della dummy baseline")


def evaluate_with_cross_validation(X_train, y_train):

    pipe = build_training_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "precision", "recall", "f1"])
    # cross_validate ritorna un dizionario con più dati della semplice cross_val_score che invece ritorna un array degli score
    cv_accuracy = scores["test_accuracy"]
    # mettendo un solo metodo di scoring come "accuracy" la chiave da accedere sarebbe stata "test_score". Passando una lista di accuracy diventano le seguenti
    cv_precision = scores["test_precision"]
    cv_recall = scores["test_recall"]
    cv_f1 = scores["test_f1"]

    return {
        "accuracy_mean": cv_accuracy.mean(),
        "accuracy_std": cv_accuracy.std(),
        "precision_mean": cv_precision.mean(),
        "precision_std": cv_precision.std(),
        "recall_mean": cv_recall.mean(),
        "recall_std": cv_recall.std(),
        "f1_mean": cv_f1.mean(),
        "f1_std": cv_f1.std()
    }

def print_cv_summary(cv_metrics):
    print("Cross-validation summary:")
    print("-" * 30)
    print(f"Accuracy : {cv_metrics['accuracy_mean']:.3f} +/- {cv_metrics['accuracy_std']:.3f}")
    print(f"Precision: {cv_metrics['precision_mean']:.3f} +/- {cv_metrics['precision_std']:.3f}")
    print(f"Recall   : {cv_metrics['recall_mean']:.3f} +/- {cv_metrics['recall_std']:.3f}")
    print(f"F1       : {cv_metrics['f1_mean']:.3f} +/- {cv_metrics['f1_std']:.3f}")


def evaluate_pipeline(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    pipeline_metrics = evaluate_predictions(y_test, y_pred, "Pipeline")
    return pipeline_metrics



def main():
    X_train, X_test, y_train, y_test = load_data()

    cv_metrics = evaluate_with_cross_validation(X_train, y_train)
    print_cv_summary(cv_metrics)
    print()

    pipe = build_training_pipeline()
    pipe.fit(X_train, y_train)

    pipeline_metrics = evaluate_pipeline(pipe, X_test, y_test)

    print()
    dummy_metrics = evaluate_dummy_baseline(X_train, X_test, y_train, y_test)
    print()

    compare_models(pipeline_metrics, dummy_metrics)




if __name__ == "__main__":
    main()



# con random_state di 40 come seed otteniamo un modello non accezionale ma comunque credibile con performance stabili e superiori alla baseline banale
# l'accuracy è quasi coincidente con la media della cross validation, quindi non possiamo considerare lo split fortunato o sfortunato
# anche precision, recall e f1 sono nell'interno delle loro medie da cross validation con piccole variazioni (soprattutto per recall)
# il confronto con la baline dummy suggerisce che il modello non sta facendo predizioni banali ma sra effettivamente imparando dalle feature