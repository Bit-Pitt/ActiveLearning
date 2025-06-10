import json
from load_dataset import * 
import numpy as np
from binary_indip_unc import * 
from utils import *
from models import *
from evaluation import *
from tqdm.auto import tqdm
import os


#Setup
tqdm.pandas()
path_results = os.path.abspath("results_combined")
ds_name = "Pubmed_train2.csv"
ds,TARGET_COLS = carica_ds(ds_name)
#methods = ['standard', 'randomAL', 'entropyAL_10it', 'entropyAL_50it', 'ebuAL_10it']
seeds = [4, 74, 58, 84, 19, 91, 33]
train_sizes = [500, 2000, 10000, 25000]



methods = ['ebuAL_10it']
seeds = [4,74]
train_sizes = [10000]



results = {method: [] for method in methods}


for method in methods:
    for seed in seeds:
        model = choose_model(2,seed)
        np.random.seed(seed)
        print(f"\nImportato il seed: {seed}")

        for size in train_sizes:
            
            dtrain,X_train,y_train,X_pool,y_pool,X_test,y_test = ds_split(ds,size,TARGET_COLS,seed)
            total_samples_to_add = 500 if (0.33 * size) < 500 else int(0.33 * size)

            # Allena il modello secondo il metodo
            if method == 'standard':
                model.fit(X_train, y_train)

            elif method == 'randomAL':
                sample_per_iter = total_samples_to_add // 10
                res = random_active_learning(model, X_train, y_train, X_pool, y_pool, iterations=10,k=sample_per_iter)
                model = res[0] 

            elif method == 'entropyAL_10it':
                sample_per_iter = total_samples_to_add // 10
                res = active_learning(model, X_train, y_train, X_pool, y_pool, iterations=10,k=sample_per_iter)
                model = res[0] 

            elif method == 'entropyAL_50it':
                sample_per_iter = total_samples_to_add // 50
                res = active_learning(model, X_train, y_train, X_pool, y_pool, iterations=50,k=sample_per_iter)
                model = res[0] 

            elif method == 'ebuAL_10it':
                sample_per_iter = total_samples_to_add // 10               
                res = active_learning(model, X_train, y_train, X_pool, y_pool, iterations=10, k=sample_per_iter*3, k_ebu=sample_per_iter, ebu=True)
              
                    
                model = res[0] 


            # Evaluation + salvataggio su dizionario stile Json
            moc_results = evaluation(model,X_test,y_test,False)
            mean_metrics = moc_results.loc["Mean"].to_dict()
            evaluation_result = {
                "seed": seed,
                "train_size": size,
                "metrics": mean_metrics              # solo la riga "Mean" è stata salvata
            }           
            results[method].append(evaluation_result)

    # Salva i risultati in un file JSON
    path_results = os.path.join(path_results, method)
    print(f"Il path è {path_results}")
    with open(f'{path_results}.json', 'w') as f:
        json.dump(results[method], f, indent=4)
