import json
import numpy as np
from load_dataset import * 

#Setup
path = os.path.abspath("results")
ds_name = "Pubmed_train2.csv"
ds,TARGET_COLS = load_dataset(ds_name)
#methods = ['standard', 'random', 'entropy_10x50', 'entropy_50x10', 'ebu']
methods = ['standard']
seeds = [42, 6, 72, 56, 87, 24, 91, 14]
train_sizes = [500, 3000, 10000, 25000]
model = choose_model(2)

results = {method: [] for method in methods}

for method in methods:
    for seed in seeds:
        
        np.random.seed(seed)

        for size in train_sizes:
            
            dtrain,X_train,y_train,X_pool,y_pool,X_test,y_test = ds_split(ds,size,TARGET_COLS)

            # Allena il modello secondo il metodo
            if method == 'standard':
                model.fit(X_train, y_train)
            elif method == 'random':
                model = random_active_learning(model, X_train, y_train, X_pool, y_pool, iterations=10,k=50)  



            # Evaluation + salvataggio su dizionario
            moc_results = evaluation(model,X_test,y_test,False)
            mean_metrics = moc_results.loc["Mean"].to_dict()
            evaluation_result = {
                "seed": seed,
                "train_size": size,
                "metrics": mean_metrics              # solo la riga "Mean" è stata salvata
            }           
            results[method].append(evaluation_result)

    # Salva i risultati in un file JSON
    path = os.path.join(path, method)
    print(f"Il path è {path}")
    with open(f'{path}.json', 'w') as f:
        json.dump(results[method], f, indent=4)
