
import pandas as pd
from tqdm.auto import tqdm
from functools import partial
from sklearn.metrics import (
    hamming_loss, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
)
from utils import *

def evaluation(model,X_test,y_test,ifplot):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

    metrics = {
        "accuracy" : accuracy_score,
        "precision" : precision_score,
        "recall" : recall_score,
        "f1_macro" : partial(f1_score, average="macro"),
    }

    moc_results = evaluate_label_wise(y_test, y_pred, metrics)
    print(moc_results)

    moc_results_2 = evaluate(y_test, y_pred, {
        "hamming_loss" : hamming_loss,
        "accuracy" : accuracy_score,
        "f1_macro" : partial(f1_score, average="macro"),
    })

    print(moc_results_2)
    if ifplot:
        plot_results(moc_results)
        plot_confusion_matrices(y_test, y_pred)
    