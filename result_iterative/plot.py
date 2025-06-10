import json
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Percorso e metodi
base_path = os.path.abspath("result_iterative")
methods = ['randomAL', 'entropyAL', 'ebuAL']
all_data = []


# Caricamento dati
for method in methods:
    path_method = os.path.join(base_path, f"{method}.json")
    with open(path_method, "r") as f:
        raw = json.load(f)
        for entry in raw:
            entry_flat = {
                "method": method,
                "iteration": entry["iteration"],
                **entry["metrics"]
            }
            all_data.append(entry_flat)


df = pd.DataFrame(all_data)

# Ordine per sicurezza non necessario
df = df.sort_values(by=["method", "iteration"])
metrics = ["accuracy", "precision", "recall", "f1_micro"]

# griglia di grafici (2x2)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Un grafico per ogni metrica
for i, metric in enumerate(metrics):
    sns.lineplot(
        data=df,
        x="iteration",
        y=metric,
        hue="method",
        marker="o",
        ax=axes[i]
    )
    axes[i].set_title(f"{metric.capitalize()} vs Iteration")
    axes[i].set_xlabel("Iteration")
    axes[i].set_ylabel(metric.capitalize())
    axes[i].grid(True)

# Sposta la legenda fuori dal grafico
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=len(methods), bbox_to_anchor=(0.5, -0.02))

# Rimuovi legende duplicate
for ax in axes:
    ax.get_legend().remove()

# Imposta layout ordinato
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()
