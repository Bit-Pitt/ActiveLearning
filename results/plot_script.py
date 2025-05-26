import json
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


path = os.path.abspath("results")
#methods = ['standard', 'randomAL', 'entropyAL_10it', 'entropyAL_50it', 'ebuAL_10it']
methods = ['standard','randomAL'] 
all_data = []

#Carichiamo tutti i risultati in all_data
for method in methods:
    path_method = os.path.join(path, f"{method}.json") 
    with open(path_method, "r") as f:
        raw = json.load(f)
        for entry in raw:
            entry_flat = {                        #Il dizionario metriche viene espanso
                "method": method,
                "seed": entry["seed"],
                "train_size": entry["train_size"],
                **entry["metrics"]
            }
            all_data.append(entry_flat)


df = pd.DataFrame(all_data)

# Medie dei seeds (raggruppo per metodo e train_size)
grouped = df.groupby(["method", "train_size"]).mean(numeric_only=True).reset_index()


for metric in ["accuracy", "precision", "recall","f1_macro"]:
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=grouped, x="train_size", y=metric, hue="method", marker="o")
    plt.title(f"{metric.capitalize()} vs Train Size")
    plt.xlabel("Training Set Size")
    plt.ylabel(metric.capitalize())
    plt.grid(True)
    plt.tight_layout()
    plt.show()