import re
import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import contractions
from sklearn.metrics import confusion_matrix


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
