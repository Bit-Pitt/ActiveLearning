import json
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# Percorso e metodi
path = os.path.abspath("results_combined")
methods = ['standard', 'randomAL', 'entropyAL_10it', 'entropyAL_50it', 'ebuAL_10it']
all_data = []


# Caricamento dati
for method in methods:
    path_method = os.path.join(path, f"{method}.json")
    with open(path_method, "r") as f:
        raw = json.load(f)
        for entry in raw:
            entry_flat = {
                "method": method,
                "seed": entry["seed"],
                "train_size": entry["train_size"],
                **entry["metrics"]
            }
            all_data.append(entry_flat)


df = pd.DataFrame(all_data)


# Media dei seeds
grouped = df.groupby(["method", "train_size"]).mean(numeric_only=True).reset_index()


# Lista delle metriche
metrics = ["accuracy", "precision", "recall", "f1_macro"]


# Creazione di un'unica figura con sottoframe
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()  # Trasformiamo in una lista piatta


for i, metric in enumerate(metrics):
    sns.lineplot(data=grouped, x="train_size", y=metric, hue="method", marker="o", ax=axes[i])
    axes[i].set_title(f"{metric.capitalize()} vs Train Size")
    axes[i].set_xlabel("Training Set Size")
    axes[i].set_ylabel(metric.capitalize())
    axes[i].grid(True)


# Layout ordinato e legenda fuori dalla griglia
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, -0.02))
for ax in axes:
    ax.get_legend().remove()


plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.show()



