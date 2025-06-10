
import pandas as pd
from tqdm.auto import tqdm
from functools import partial
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
)
from utils import *
import os,json

def evaluation(model,X_test,y_test,ifplot):
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns)

   
    metrics = {
    "accuracy": accuracy_score,
    "precision": partial(precision_score, zero_division=0),
    "recall": partial(recall_score, zero_division=0),
    #"f1_macro": partial(f1_score, average="macro", zero_division=0),
    "f1_micro": partial(f1_score, average="micro", zero_division=0)         #aggiunta alla fine del progetto! (rimuovi per risolvere vecchie funzionalità)
    }

    moc_results = evaluate_label_wise(y_test, y_pred, metrics)
    print(moc_results)

    '''
    moc_results_2 = evaluate(y_test, y_pred, {
        "hamming_loss" : hamming_loss,
        "accuracy" : accuracy_score,
        "f1_macro" : partial(f1_score, average="macro"),
    })
    print(moc_results_2)
    '''

    if ifplot:
        plot_results(moc_results)
        plot_confusion_matrices(y_test, y_pred)
        
    return moc_results

def log_metrics_callback(model, X_test, y_test, method, it):
    moc_results = evaluation(model, X_test, y_test, False)
    mean_metrics = moc_results.loc["Mean"].to_dict()

    result = {
        "iteration": it,
        "metrics": mean_metrics,
    }

    out_path = os.path.join("result_iterative", f"{method}.json")

    # Se esiste già, carica i risultati precedenti
    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            existing = json.load(f)
    else:
        existing = []

    # Aggiungi il nuovo risultato
    existing.append(result)

    # Riscrivi tutto in append
    with open(out_path, "w") as f:
        json.dump(existing, f, indent=4)
    