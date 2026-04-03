from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
def evaluate_model(model, X_test, y_test):

    y_preds=model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]

    metrics={}

    def accuracy(y_test, y_preds):
        model_accuracy=accuracy_score(y_test, y_preds)
        print("Model accuracy: ", model_accuracy)
        metrics["accuracy"] = model_accuracy

    def precision(y_test, y_preds):
        model_precision=precision_score(y_test, y_preds)
        print("Precision score: ", model_precision)
        metrics["precision"] = model_precision

    def f1(y_test, y_preds):
        model_f1=f1_score(y_test, y_preds)
        print("F1 score: ", model_f1)
        metrics["f1"] = model_f1

    def recall(y_test, y_preds):
        model_recall=recall_score(y_test, y_preds)
        print("Recall score: ", model_recall)
        metrics["recall"] = model_recall

    def roc_auc(y_test, y_probs):
        model_roc_auc=roc_auc_score(y_test, y_probs)
        print("ROC AUC score: ", model_roc_auc)
        metrics["roc_auc"] = model_roc_auc

    def confusion_matrix_fn(y_test, y_preds):
        cf=confusion_matrix(y_test, y_preds);
        print("Confusion matrix:\n", cf)
        metrics["confusion_matrix"] = cf
    
    def classification_report_fn(y_test, y_preds):
        report=classification_report(y_test, y_preds)
        print("Classification report:\n", report)
        metrics["classification_report"] = report

    accuracy(y_test, y_preds)
    precision(y_test, y_preds)
    f1(y_test, y_preds)
    recall(y_test, y_preds)
    roc_auc(y_test, y_probs)
    confusion_matrix_fn(y_test, y_preds)
    classification_report_fn(y_test, y_preds)

    return metrics
