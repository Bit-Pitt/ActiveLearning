import kagglehub
import warnings
import os
import re
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import contractions
import xgboost as xgb
import random
import numpy as np
from wordcloud import WordCloud
from tqdm.auto import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier,ClassifierChain
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    hamming_loss, 
    accuracy_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    confusion_matrix,
)

tqdm.pandas()


#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('punkt_tab')

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
sns.set_theme(style="darkgrid", palette="pastel")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path = os.path.abspath("dataset")

dtrain = pd.read_csv(os.path.join(path, "train.csv"))
dtest = pd.read_csv(os.path.join(path, "test.csv"))


rows, columns = dtrain.shape
print(f"The train set contains {rows} rows and {columns} columns")

rows, columns = dtest.shape
print(f"The test set contains {rows} rows and {columns} columns")

print(dtrain.head())
print(dtrain.tail())
print(dtrain.info())
print(dtrain.isna().sum())
print(dtrain.describe())
print(dtrain.duplicated().sum())
print()


ID_COL = "ID"
FEATURES = ["TITLE", "ABSTRACT"]
TARGET_COLS = [col for col in dtrain.columns if col not in [ID_COL] + FEATURES]

if (False):
    labels_counts = dtrain[TARGET_COLS].sum(axis=0)
    labels, counts = labels_counts.index, labels_counts.values
    ax = sns.barplot(x=labels, y=counts)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.set_title("Labels distribution")
    ax.set_ylabel("Counts")
    ax.set_xlabel("Labels")
    plt.tight_layout()  # migliora il layout se le etichette si sovrappongono
    plt.show()  

if (False):
    X = dtrain[TARGET_COLS].values
    X = X.T @ X
    sns.heatmap(X, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.tight_layout()  # migliora il layout se le etichette si sovrappongono
    plt.show() 

if (False):
    X = X / X.sum(axis=1)
    sns.heatmap(X, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.show() 


if (False):
    dtrain[TARGET_COLS].sum(axis=1).value_counts().sort_index().plot(kind="bar")
    plt.tight_layout()
    plt.show() 

if (False):
    dtrain['title_len'] = dtrain['TITLE'].apply(lambda x: len(x.split()))
    sns.histplot(dtrain['title_len'], kde=True) 
    plt.tight_layout()
    plt.show() 

if (False):
    dtrain['abstract_len'] = dtrain['ABSTRACT'].apply(lambda x: len(x.split()))
    sns.histplot(dtrain['abstract_len'], kde=True)
    plt.tight_layout()
    plt.show() 

if (False):
    for title in dtrain['TITLE'].sample(10):
        print(title)
    

if (False):
    for title in dtrain['ABSTRACT'].sample(10):
        print(title)

class Processor:

    def __init__(self):

        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.stemmer = nltk.stem.SnowballStemmer("english")

    def __call__(self, text : str) -> str:

        text = text.lower() # Lowercase
        text = text.replace('-', ' ') # Replace hyphens with spaces
        text = contractions.fix(text) # Expand contractions, for example don't -> do not
        text = re.sub(r"\$.*?\$", "", text) # Replace LaTeX equations
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text) # Remove special characters
        text = re.sub(r"\d+", " ", text) # Remove digits
        text = re.sub(r"\s+", " ", text) # Remove extra whitespaces
        text = nltk.word_tokenize(text) # Tokenize
        text = [word for word in text if word not in self.stopwords] # Remove stopwords
        text = [self.stemmer.stem(word) for word in text if word] # Stem words
        text = " ".join(text)

        return text

if (True):             #text cleaning, creazione di nuove colonne con testi processati
    processor = Processor()
    processor(dtrain['TITLE'].iloc[9])
    dtrain['PROCESSED_TITLE'] = dtrain['TITLE'].progress_apply(processor)
    dtrain['PROCESSED_ABSTRACT'] = dtrain['ABSTRACT'].progress_apply(processor)
    dtrain['PROCESSED_TEXT'] = dtrain['PROCESSED_TITLE'] + " " + dtrain['PROCESSED_ABSTRACT']

def get_vocab(docs : pd.Series) -> set:
    vocab = set()
    for doc in tqdm(docs):
        vocab.update(doc.split())

    return vocab


if (True):
    vocab = get_vocab(dtrain['PROCESSED_TEXT'])
    print(f"The vocabulary contains {len(vocab)} unique words.") 

    #data splitting:  prendo solo una parte di un dataset e lo scompongo
if (True):
    X = dtrain['PROCESSED_TEXT']
    y = dtrain[TARGET_COLS]                                 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.99)
    # test set adesso è il 70% dell'originale, il validation è il rimanente 30 che sarà usata per testare (perchè in realtà test.csv non è etichettato)
    print(f"Train set contains {len(X_train)} samples")
    print(f"Validation set contains {len(X_val)} samples")
    
    
                # MODEL AND TRAINING
    
#UTILS
def evaluate_label_wise(
    y_true : pd.DataFrame,
    y_pred : pd.DataFrame,
    metrics : dict
) -> pd.DataFrame:

    results = []

    for label in y_true.columns:

        results.append({})
        
        for metric_name, metric in metrics.items():
            results[-1][metric_name] = metric(y_true[label], y_pred[label])

    results = pd.DataFrame(results, index=y_true.columns)
    results.loc['Mean'] = results.mean(axis=0)  

    return results

def evaluate(
    y_true : pd.DataFrame,
    y_pred : pd.DataFrame,
    metrics : dict
) -> pd.DataFrame:

    results = {}

    for metric_name, metric in metrics.items():
        results[metric_name] = metric(y_true, y_pred)

    return pd.Series(results)


def plot_confusion_matrices(
    y_true : pd.DataFrame,
    y_pred : pd.DataFrame,
): 
    fig, axes = plt.subplots(2, 3, figsize=(9, 5))

    for i, label in enumerate(y_true.columns):
        cm = confusion_matrix(y_true[label], y_pred[label])
        cm = cm / cm.sum(axis=1)[:, None]
        ax = sns.heatmap(cm, ax=axes[i//3, i%3], annot=True, fmt=".2f", cmap="Blues")
        ax.set_title(f"{label}")

    plt.tight_layout()


def plot_results(results : pd.DataFrame) -> None:

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for i,metric in enumerate(results.columns):
        ax = axes[i//2, i%2]
        ax = sns.barplot(data=results, x=results.index, y=metric, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()

    return ax


def plot_cooccurrence_matrix(y : pd.DataFrame) -> None:
    X = y.T @ y
    X = X / X.sum(axis=1)
    sns.heatmap(X, annot=True, fmt=".2f", cmap="Blues", xticklabels=y.columns, yticklabels=y.columns)


#Modello  MULTI-OUTPUT 
model = Pipeline([
    ("vectorizer", TfidfVectorizer()),
    ("svd", TruncatedSVD(n_components=256)),
    ("classifier", MultiOutputClassifier(estimator=xgb.XGBClassifier()))
]) 

#training
model.fit(X_train, y_train)


#Evaluation
y_pred = model.predict(X_val)
y_pred = pd.DataFrame(y_pred, columns=y_val.columns)

metrics = {
    "accuracy" : accuracy_score,
    "precision" : precision_score,
    "recall" : recall_score,
    "f1_macro" : partial(f1_score, average="macro"),
}

moc_results = evaluate_label_wise(y_val, y_pred, metrics)
print(moc_results)

moc_results_2 = evaluate(y_val, y_pred, {
    "hamming_loss" : hamming_loss,
    "accuracy" : accuracy_score,
    "f1_macro" : partial(f1_score, average="macro"),
})

print(moc_results_2)

plot_results(moc_results)
plot_confusion_matrices(y_val, y_pred)
plt.show()










